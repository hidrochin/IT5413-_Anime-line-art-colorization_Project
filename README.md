# Reference-Based Anime Line-Art Colorization

A controlled comparison of a GAN, a diffusion-type model trained from scratch, and a large pretrained
diffusion model on the same task, data, and metrics: given a line-art sketch and a color reference
image of the same character, color the sketch so it matches the reference.

**Status:** planning stage (checkpoint 1, 25 Oct 2026). Problem, dataset, models, and protocol are
designed; a Model A training notebook is now available, with no real-data training results yet.
See [`Reference-Based Anime Line-Art Colorization.md`](<Reference-Based Anime Line-Art Colorization.md>)
for the full design document — task definition, dataset, model architectures, evaluation plan, task
assignment, and timeline.

## At a glance

| | |
| --- | --- |
| Task | Line-art sketch + color reference → colored anime face |
| Data | Selected: [AnimeFaceColorization (Hugging Face)](https://huggingface.co/datasets/ComputerVisionAnimeProject/AnimeFaceColorization), resized to 256×256; fixed source-group train/val/test split |
| Model A | Reference-conditioned pix2pix GAN, trained from scratch |
| Model B | Latent flow-matching model with DINOv2 reference cross-attention, trained from scratch |
| Model C | Stable Diffusion 1.5 + ControlNet (line-art) + IP-Adapter, pretrained, zero-shot / optional LoRA |
| Main question | Do diffusion-type models trade fidelity for diversity better than a GAN? |
| Compute | Free Kaggle GPUs (T4/P100), 3 team members |

## Model A: Kaggle notebook

Open [`notebooks/model_a_reference_pix2pix_kaggle.ipynb`](notebooks/model_a_reference_pix2pix_kaggle.ipynb)
in Kaggle and attach the paired dataset with **Add Input**. Set `DATA_ROOT` to
the directory containing `sketch/` and `reference/`, or directly to
`AnimeFaceColorization_paired.zip`. ZIP input is extracted once into Kaggle working
storage and reused in the current session.

The notebook reads attached data only. It contains no Hugging Face dataset download,
token, login, or Secrets configuration. Internet is needed initially for pretrained
VGG19/LPIPS weights if they are not already cached.

Defaults train on 1,000 pairs for two epochs and export eight inputs with five seeds.
After inspecting images/losses, use a new `RUN_NAME`, `TRAIN_LIMIT=None`, and
`EPOCHS=30` for full training. Checkpoints support resuming between Kaggle sessions.

The loader skips unpaired files and exports `splits/data_summary.json` plus shared
manifest/split files. The approximately 80/10/10 split keeps source-image variants
together. These are splits of the selected paired subset, not the official
AnimeDiffusion benchmark. Notebook cells include Vietnamese explanations for the
U-Net/PatchGAN, TPS references, losses, training, sampling and PNG-based evaluation.

For downloading locally to `D:\AnimeFaceColorization`, run
`python scripts/download_dataset.py` in an environment with `huggingface_hub`
and `tqdm`. The local script uses its `HF_TOKEN` constant, validates it, then lists and downloads
`sketch/` and `reference/` separately instead of scanning the whole repository root.
It reuses the same local download cache when restarted. Windows must permit
writing to the destination (use an Administrator PowerShell if required).
The local downloader is excluded from Git because it contains a hardcoded credential.

To package the selected pairs for Kaggle and remove unused local files, stop any
active download and run `python scripts/package_paired_dataset.py` with write
permission to `D:\AnimeFaceColorization`. It creates
`D:\AnimeFaceColorization\AnimeFaceColorization_paired.zip`, verifies its file list,
sizes and CRC, then deletes unmatched images and the `.cache` download directory.
Matched raw images and README remain available. No source files are deleted if
ZIP verification fails or the source changes during packaging.

## Planned repository structure

Apart from the Model A notebook, this is the target layout as it gets built out (see Section 10 of the design doc):

```text
.
├── README.md
├── requirements.txt
├── configs/                  # gan.yaml, flow.yaml, controlnet.yaml
├── splits/                   # train.txt, val.txt, test.txt, cross_ref_pairs.csv
├── src/
│   ├── data/                 # Dataset class, reference warping
│   ├── models/
│   │   ├── gan/              # Model A
│   │   ├── flow/             # Model B
│   │   └── controlnet/       # Model C
│   ├── colorizer.py          # shared Colorizer interface
│   └── utils/                # checkpointing, logging, seeding
├── scripts/
│   ├── cache_latents.py
│   ├── train_gan.py
│   ├── train_flow.py
│   ├── tune_controlnet.py
│   ├── sample.py              # writes outputs/<model>/<mode>/...
│   └── evaluate.py            # all metrics -> results/*.csv
├── notebooks/                 # Kaggle entry notebooks, one per model
├── results/                   # CSV tables and figures (committed)
└── outputs/                   # generated images (not committed)
```

## Team

Each member owns one model end to end plus one shared technical component:

| Member | Model | Shared component |
| --- | --- | --- |
| Member 1 | Model A — Reference GAN | Data pipeline (download, splits, augmented self-reference) |
| Member 2 | Model B — Latent flow matching | Latent caching + sampling harness |
| Member 3 | Model C — SD1.5 + ControlNet + IP-Adapter | Evaluation script + plots |

## References

The full reading list (task/data, model, and evaluation papers) is in Section 12 of the design
document.
