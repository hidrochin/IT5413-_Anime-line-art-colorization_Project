"""CPU integration check on synthetic images; no pretrained weight downloads.

Run with Python containing torch, torchvision, Pillow, matplotlib, tqdm,
scikit-image, nbformat and IPython. Uses a temporary directory and a narrow
U-Net for the training check. Does not claim to validate real-data quality.
"""
import ast
import json
import os
from pathlib import Path
import tempfile
import sys
import shutil
from unittest.mock import patch

os.environ['MPLBACKEND'] = 'Agg'
os.environ['MPLCONFIGDIR'] = str(Path(tempfile.gettempdir()) / 'model-a-matplotlib-cache')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from PIL import Image
import torch
import nbformat

NOTEBOOK = Path(__file__).resolve().parents[1] / 'notebooks/model_a_reference_pix2pix_kaggle.ipynb'
nb = nbformat.read(NOTEBOOK, as_version=4)
nbformat.validate(nb)
CELL_NAMES = ['imports', 'config', 'data', 'dataset', 'models', 'losses', 'validation',
              'checkpoint', 'train', 'plots', 'inference', 'export', 'metrics', 'timing']
# Stable cell IDs keep tests independent of heading numbers and Markdown layout.
by_id = {cell.id: cell.source for cell in nb.cells if cell.cell_type == 'code'}
sources = dict(zip(CELL_NAMES, [by_id[f'cell-{i:03d}'] for i in [2,6,8,10,12,14,16,18,20,22,24,26,28,30]]))
for source in sources.values():
    ast.parse(source)


def execute(name, namespace):
    print(f'Execute {name}', flush=True)
    source = sources[name]
    exec(compile(source, f'{NOTEBOOK.name}:{name}', 'exec'), namespace)


def namespace_for(data, work, run_name, epochs, resume='', do_train=True, cpu_amp_scalers=False):
    ns = {'__name__': '__main__', 'display': lambda *args, **kwargs: None}
    execute('imports', ns)
    # Replace just the instantiation with a test configuration, then execute the actual cell.
    config_source = sources['config'].replace('cfg = Config()', f'''cfg = Config(
        DATA_ROOT={str(data)!r}, WORK_ROOT={str(work)!r}, RUN_NAME={run_name!r},
        EPOCHS={epochs}, RESUME_PATH={str(resume)!r}, DO_TRAIN={do_train!r}, BATCH_SIZE=1, NUM_WORKERS=0,
        NGF=8, NDF=8, TRAIN_LIMIT=None, LAMBDA_PERCEPTUAL=0,
        AMP=False, USE_LPIPS=False, EXPORT_TEST_LIMIT=None)''')
    exec(compile(config_source, 'test-config', 'exec'), ns)
    for name in ['data','dataset','models','losses','validation','checkpoint','train']:
        if name == 'checkpoint' and cpu_amp_scalers:
            ns['scaler_g'] = torch.amp.GradScaler('cpu', enabled=True)
            ns['scaler_d'] = torch.amp.GradScaler('cpu', enabled=True)
        execute(name, ns)
    return ns


def main():
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix='model-a-check-') as temporary:
        root = Path(temporary)
        data = root / 'data'
        rng = np.random.default_rng(5413)
        for kind in ['sketch', 'reference']:
            (data / kind / '000_character_a').mkdir(parents=True)
        for i in range(20):
            image = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
            Image.fromarray(image).save(data / 'reference/000_character_a' / f'{i}.png')
            Image.fromarray(image.mean(-1).astype(np.uint8)).save(
                data / 'sketch/000_character_a' / f'{i}.png')

        with patch.dict(os.environ, {'HF_TOKEN':'example-test-only-token'}):
            first = namespace_for(data, root, 'resumed', 1)
        saved_config = (first['RUN']/'config.json').read_text(encoding='utf-8')
        saved_checkpoint = torch.load(first['RUN']/'checkpoints/last.pt', map_location='cpu', weights_only=False)
        assert 'HF_TOKEN' not in saved_config and 'example-test-only-token' not in saved_config
        assert 'HF_TOKEN' not in saved_checkpoint['config']
        del saved_checkpoint
        example = first['val_ds'][0]
        assert example['sketch'].shape == (1,256,256)
        assert example['reference'].shape == example['target'].shape == (3,256,256)
        assert torch.equal(example['reference'], first['val_ds'][0]['reference'])
        assert (first['RUN'] / 'checkpoints/best.pt').is_file()
        assert first['state']['global_step'] == len(first['train_ds'])
        checkpoint = first['RUN'] / 'checkpoints/last.pt'

        # Only last.pt is available in a new Kaggle run, with no old preview PNGs.
        checkpoint_folder = root/'input_checkpoint'
        checkpoint_folder.mkdir()
        shutil.copy2(checkpoint, checkpoint_folder/'last.pt')
        recovery = namespace_for(data, root, 'recover_last_only', 1, checkpoint_folder/'last.pt',
                                 do_train=False, cpu_amp_scalers=True)
        assert recovery['scaler_g'].is_enabled() and recovery['scaler_g'].state_dict()
        execute('plots', recovery)
        execute('inference', recovery)
        assert recovery['checkpoint_selection'] == 'last_available'
        assert recovery['best_path'] == checkpoint_folder/'last.pt'
        assert all(torch.equal(v,first['G'].state_dict()[k]) for k,v in recovery['G'].state_dict().items())
        print('PASS: last-only recovery, missing preview, and empty-to-enabled AMP scaler state.', flush=True)

        resumed = namespace_for(data, root, 'resumed', 2, checkpoint)
        baseline = namespace_for(data, root, 'uninterrupted', 2)
        for key, value in resumed['G'].state_dict().items():
            assert torch.equal(value, baseline['G'].state_dict()[key]), f'Resume mismatch: {key}'
        assert resumed['state']['history'] == baseline['state']['history']
        print('PASS: checkpoint resume matches uninterrupted CPU training.', flush=True)

        mid = namespace_for(data, root, 'mid_epoch', 2, do_train=False)
        order = torch.randperm(len(mid['train_ds']), generator=torch.Generator().manual_seed(mid['cfg'].SEED)).tolist()
        iterator = iter(mid['loader'](torch.utils.data.Subset(mid['train_ds'], order), 1,
                                     generator=torch.Generator().manual_seed(mid['cfg'].SEED)))
        batch = next(iterator)
        losses = mid['train_step'](batch)
        mid['state'].update(epoch=0, next_batch=1, global_step=1, epoch_sum=losses, epoch_seen=1)
        mid['save_checkpoint']('last.pt')
        mid_resumed = namespace_for(data, root, 'mid_epoch', 2, mid['RUN']/'checkpoints/last.pt')
        assert all(torch.equal(v, baseline['G'].state_dict()[k])
                   for k,v in mid_resumed['G'].state_dict().items())
        assert mid_resumed['state']['history'] == baseline['state']['history']
        print('PASS: mid-epoch resume matches uninterrupted CPU training.', flush=True)

        # Actual default-width architecture forward check (training above uses width=8).
        full_g = resumed['ReferenceUNet'](64).eval()
        full_d = resumed['PatchDiscriminator'](64).eval()
        item = resumed['val_ds'][0]
        with torch.no_grad():
            out = full_g(item['sketch'][None], item['reference'][None])
            assert out.shape == (1,3,256,256)
            assert torch.isfinite(out).all()
            assert full_d(item['sketch'][None], out).shape == (1,1,30,30)
        del full_g, full_d
        print('PASS: default-width G/D forward shapes.', flush=True)

        # Exercise real VGG19 feature/loss/gradient code without downloading weights.
        # This checks differentiability, not the meaning of pretrained features.
        tv_models = resumed['models']
        original_vgg = tv_models.vgg19
        try:
            tv_models.vgg19 = lambda **kwargs: original_vgg(weights=None)
            perceptual = resumed['VGGPerceptual']()
        finally:
            tv_models.vgg19 = original_vgg
        fake = (torch.rand(1,3,64,64)*2-1).requires_grad_()
        loss = perceptual(fake, torch.zeros_like(fake))
        loss.backward()
        assert fake.grad is not None and torch.isfinite(fake.grad).all() and fake.grad.abs().sum() > 0
        assert all(not p.requires_grad for p in perceptual.parameters())
        del fake, loss
        print('PASS: VGG19 perceptual gradient path (random weights, offline).', flush=True)

        # The actual GAN step must also backpropagate VGG features into G.
        mid['perceptual'] = perceptual
        mid['cfg'].LAMBDA_PERCEPTUAL = 10.0
        batch = next(iter(mid['loader'](mid['train_ds'], 1)))
        before = next(mid['G'].parameters()).detach().clone()
        values = mid['train_step'](batch)
        assert values['perceptual'] > 0 and all(np.isfinite(list(values.values())))
        assert not torch.equal(before,next(mid['G'].parameters()).detach())
        mid['perceptual'] = None
        del perceptual, before, batch
        print('PASS: combined GAN + L1 + actual VGG19-feature backward/optimizer step.', flush=True)

        for name in ['plots','inference','export','metrics','timing']:
            execute(name, resumed)
        summary = json.loads((resumed['RUN'] / 'results/summary_self.json').read_text())
        assert summary['n_inputs'] == 2 and summary['psnr'] is not None
        assert len(list(resumed['OUTPUT_DIR'].glob('*.png'))) == 10
        # Test the real LPIPS metric pipeline offline, using a random AlexNet backbone.
        import lpips
        original_lpips = lpips.LPIPS
        resumed['cfg'].USE_LPIPS = True
        with patch.object(lpips, 'LPIPS', side_effect=lambda **kwargs:original_lpips(pnet_rand=True, **kwargs)):
            execute('metrics', resumed)
        assert np.isfinite(resumed['summary']['lpips'])
        assert np.isfinite(resumed['summary']['diversity_lpips'])
        resumed['cfg'].USE_LPIPS = False
        print('PASS: actual LPIPS ground-truth/diversity metric paths (random backbone, offline).', flush=True)
        item = resumed['export_ds'][0]
        saved_ref = resumed['read_rgb01'](resumed['INPUT_DIR'] / f'{item["image_id"]}_reference.png')
        assert torch.allclose(saved_ref*2-1, item['reference'], atol=1e-6)
        # Verify stochastic sampling with frozen BatchNorm, and same-seed determinism.
        G = resumed['G']
        before = {k:v.clone() for k,v in G.state_dict().items()}
        colorizer = resumed['colorizer']
        a = colorizer.colorize(item['sketch'][None], item['reference'][None], 11)
        repeat = colorizer.colorize(item['sketch'][None], item['reference'][None], 11)
        assert torch.equal(a, repeat), 'Same seed must reproduce the same output.'
        b = colorizer.colorize(item['sketch'][None], item['reference'][None], 22)
        assert not torch.equal(a,b), 'Dropout should introduce a stochastic path.'
        assert all(torch.equal(v, G.state_dict()[k]) for k,v in before.items())
        # Run cross-reference export and ensure paired metrics are intentionally absent.
        resumed['cfg'].EVAL_MODE = 'cross'
        execute('export', resumed)
        execute('metrics', resumed)
        cross = json.loads((resumed['RUN'] / 'results/summary_cross.json').read_text())
        assert cross['n_inputs'] == 2 and cross['psnr'] is None
        assert len(resumed['metric_rows']) == 0
        # Automatic cross-reference must use another target within test.
        test_targets = {row['target'] for row in resumed['SPLITS']['test']}
        for row in resumed['SPLITS']['test']:
            ref = resumed['cross_refs'][row['image_id']]
            assert ref in test_targets and ref != row['target']
        print('PASS: train/backward, validation, checkpoint/resume, self/cross export, PNG metrics, '
              'seed determinism, dropout, frozen BatchNorm, exact shared-reference round trip.', flush=True)

        # Exercise the selected Hugging Face layout and source-group split directly.
        hf_data = root / 'hf_data'
        keys = [f'000_character/face_{i}' for i in range(12)]
        keys += [f'AnimeDiffusion_{i}/{i}{suffix}' for i in range(5)
                 for suffix in ['', '_aug1', '_aug2']]
        for key in keys:
            image = rng.integers(0,256,(32,32,3),dtype=np.uint8)
            for folder in ['reference', 'sketch']:
                dest = hf_data / folder / (key+'.png')
                dest.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(image).save(dest)
        # A partial download may contain unpaired files on either side.
        for folder, name in [('sketch','missing_color'), ('reference','missing_sketch')]:
            dest = hf_data / folder / '000_character' / f'{name}.png'
            Image.fromarray(image).save(dest)
        hf_ns = dict(resumed)
        config_source = sources['config'].replace('cfg = Config()', f'''cfg = Config(
            DATA_ROOT={str(hf_data)!r}, WORK_ROOT={str(root)!r}, RUN_NAME='hf_layout',
            NUM_WORKERS=0)''')
        exec(compile(config_source, 'hf-test-config', 'exec'), hf_ns)
        execute('data', hf_ns)
        assert hf_ns['data_summary']['paired'] == len(keys)
        assert hf_ns['data_summary']['sketch_only'] == 1
        assert hf_ns['data_summary']['reference_only'] == 1
        assert len(hf_ns['rows']) == len(keys)
        assert all('missing_' not in row['sketch'] and 'missing_' not in row['target'] for row in hf_ns['rows'])
        assert (hf_data/'sketch/000_character/missing_color.png').is_file()
        assert (hf_data/'reference/000_character/missing_sketch.png').is_file()
        assert all(hf_ns['SPLITS'][split] for split in ['train','val','test'])
        for i in range(5):
            group = [r for r in hf_ns['rows'] if f'AnimeDiffusion_{i}' in Path(r['target']).parts]
            assert len(group) == 3 and len({r['split'] for r in group}) == 1
            assert all(not r['character_id'] for r in group)
        assert all(r['character_id'] == '000_character' for r in hf_ns['rows']
                   if '000_character' in Path(r['target']).parts)
        signature = hf_ns['DATA_SIGNATURE']
        execute('data', hf_ns)
        assert signature == hf_ns['DATA_SIGNATURE']
        print('PASS: Hugging Face pairing, unpaired files skipped without deletion, reproducible grouped splits, '
              'augmentation isolation and source IDs excluded from character IDs.', flush=True)


if __name__ == '__main__':
    main()
