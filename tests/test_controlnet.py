"""Model C wiring against a fake pipeline, plus the real diffusers pipeline with tiny random parts
(no SD weights are downloaded)."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from diffusers.models.embeddings import ImageProjection
from torch import nn

from src.models.controlnet.colorizer import (ControlNetColorizer, ControlNetConfig, downsample, encode_text,
                                             keep_per_sample_ip_embeds, prepare_control)


class FakeTokenizer:
    model_max_length = 8

    def __call__(self, text, padding, max_length, truncation, return_tensors):
        ids = [min(ord(c), 99) for c in text][:max_length]
        return SimpleNamespace(input_ids=torch.tensor([ids + [0] * (max_length - len(ids))]))


def tiny_text_encoder():
    """A real CLIPTextModel (random weights) with the installed transformers' module layout."""
    from transformers import CLIPTextConfig, CLIPTextModel

    torch.manual_seed(0)
    return CLIPTextModel(CLIPTextConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=3, num_attention_heads=2,
                                        vocab_size=100, max_position_embeddings=8, bos_token_id=0, eos_token_id=1,
                                        pad_token_id=0)).eval()


class FakePipe:
    def __init__(self):
        proj = ImageProjection(image_embed_dim=8, cross_attention_dim=8, num_image_text_embeds=1)
        self.unet = SimpleNamespace(encoder_hid_proj=SimpleNamespace(image_projection_layers=[proj]))
        self.device = torch.device("cpu")
        self.tokenizer, self.text_encoder = FakeTokenizer(), tiny_text_encoder()
        self.ip_scale = None
        self.calls = []

    def set_ip_adapter_scale(self, scale):
        self.ip_scale = scale

    def encode_image(self, images, device, n, hidden):
        assert not hidden and len(images) > 0
        pos = torch.ones(len(images), 8)
        return pos, torch.zeros_like(pos)

    def __call__(self, **kw):
        assert "clip_skip" not in kw and "prompt" not in kw  # diffusers' clip_skip path breaks on transformers 5
        self.calls.append(kw)
        b = kw["prompt_embeds"].shape[0]
        assert kw["negative_prompt_embeds"].shape == kw["prompt_embeds"].shape
        return SimpleNamespace(images=torch.rand(b, 3, 512, 512))


def make(**cfg):
    pipe = FakePipe()
    return ControlNetColorizer(ControlNetConfig(device="cpu", **cfg), pipe=pipe), pipe


def test_prepare_control_inverts_and_upsamples():
    sketch = torch.ones(2, 1, 256, 256)  # white paper
    sketch[:, :, 100:102, :] = -1  # one black line
    control = prepare_control(sketch, invert=True)
    assert control.shape == (2, 3, 512, 512)
    assert control[0, 0, 0, 0] == pytest.approx(0.0)  # background -> black
    assert control[0, 0, 202, 50] > 0.9  # line -> white
    assert prepare_control(sketch, invert=False)[0, 0, 0, 0] == pytest.approx(1.0)


def test_downsample_shape():
    assert downsample(torch.rand(1, 3, 512, 512)).shape == (1, 3, 256, 256)


def test_colorize_contract():
    colorizer, pipe = make(ip_adapter_scale=0.7, guidance_scale=5.0, steps=12)
    assert pipe.ip_scale == 0.7
    out = colorizer.colorize(torch.zeros(3, 1, 256, 256), torch.zeros(3, 3, 256, 256), seed=4)
    assert out.shape == (3, 3, 256, 256) and out.min() >= -1 and out.max() <= 1
    kw = pipe.calls[-1]
    assert kw["num_inference_steps"] == 12 and kw["guidance_scale"] == 5.0
    assert len(kw["generator"]) == 3 and kw["image"].shape == (3, 3, 512, 512)
    (embeds,) = kw["ip_adapter_image_embeds"]
    assert embeds.shape == (6, 1, 8)  # CFG: negatives first, one reference per sample
    assert embeds[:3].abs().sum() == 0 and (embeds[3:] == 1).all()
    colorizer.colorize(torch.zeros(1, 1, 256, 256), torch.zeros(1, 3, 256, 256), seed=0, steps=3)
    assert pipe.calls[-1]["num_inference_steps"] == 3


def test_encode_text_clip_skip():
    pipe = FakePipe()
    enc = pipe.text_encoder
    ids = FakeTokenizer()("anime", None, 8, True, "pt").input_ids
    with torch.no_grad():
        hidden = enc(ids, output_hidden_states=True).hidden_states
        last = enc(ids)[0]
    norm = enc.text_model.final_layer_norm if hasattr(enc, "text_model") else enc.final_layer_norm
    assert torch.allclose(encode_text(pipe, "anime", None), last)
    assert torch.allclose(encode_text(pipe, "anime", 1), norm(hidden[-2]))  # CLIP skip 2 = penultimate layer
    assert not torch.allclose(encode_text(pipe, "anime", 1), last)
    # transformers 4 layout (final_layer_norm under .text_model) gives the same result
    class Enc4:
        device = enc.device
        text_model = SimpleNamespace(final_layer_norm=norm)

        def __call__(self, *args, **kwargs):
            return enc(*args, **kwargs)

    v4 = SimpleNamespace(tokenizer=pipe.tokenizer, text_encoder=Enc4())
    assert torch.allclose(encode_text(v4, "anime", 1), norm(hidden[-2]))


def test_text_embeds_cached_and_batched():
    colorizer, pipe = make()
    colorizer.colorize(torch.zeros(3, 1, 256, 256), torch.zeros(3, 3, 256, 256), seed=0)
    kw = pipe.calls[-1]
    assert kw["prompt_embeds"].shape == (3, 8, 16)
    assert torch.equal(kw["prompt_embeds"][0], kw["prompt_embeds"][2])
    assert not torch.equal(kw["prompt_embeds"], kw["negative_prompt_embeds"])
    assert len(colorizer._text) == 2  # prompt + negative prompt, encoded once
    colorizer.colorize(torch.zeros(1, 1, 256, 256), torch.zeros(1, 3, 256, 256), seed=1)
    assert len(colorizer._text) == 2


def test_no_cfg_embeds():
    colorizer, pipe = make(guidance_scale=1.0)
    colorizer.colorize(torch.zeros(2, 1, 256, 256), torch.zeros(2, 3, 256, 256), seed=0)
    assert pipe.calls[-1]["ip_adapter_image_embeds"][0].shape == (2, 1, 8)


def test_set_params():
    colorizer, pipe = make()
    colorizer.set_params(ip_adapter_scale=0.9, controlnet_scale=0.5)
    assert pipe.ip_scale == 0.9 and colorizer.cfg.controlnet_scale == 0.5
    with pytest.raises(ValueError, match="new pipeline"):
        colorizer.set_params(base_model="x")


def test_check_controlnet_script(tmp_path, monkeypatch):
    import sys

    import pandas as pd

    import scripts.check_controlnet as check

    class EchoPipe(FakePipe):
        """'Colors' each sample with the mean of its IP-Adapter embedding row, so outputs follow references."""

        def encode_image(self, images, device, n, hidden):
            pos = torch.stack([torch.tensor(np.asarray(im), dtype=torch.float32).mean() for im in images])[:, None] / 255
            return pos.repeat(1, 8), torch.zeros(len(images), 8)

        def __call__(self, **kw):
            super().__call__(**kw)
            emb = kw["ip_adapter_image_embeds"][0]
            pos = emb[emb.shape[0] // 2:] if kw["guidance_scale"] > 1 else emb
            level = pos[:, 0, 0].clamp(0, 1) * (self.ip_scale or 0)
            return SimpleNamespace(images=level.view(-1, 1, 1, 1).expand(-1, 3, 512, 512).clone())

    monkeypatch.setattr(check, "ControlNetColorizer",
                        lambda cfg: ControlNetColorizer(ControlNetConfig.from_dict({**vars(cfg), "device": "cpu"}),
                                                        pipe=EchoPipe()))
    monkeypatch.setattr(sys, "argv", ["check", "--dataset", "tests.fake_data:build_dataset", "--num-images", "4",
                                      "--batch-size", "2", "--results-dir", str(tmp_path), "--set", "device=cpu"])
    check.main()
    row = pd.read_csv(tmp_path / "check_controlnet.csv").iloc[-1]
    assert row["batch_diff"] < 1e-6 and row["n"] == 4
    assert (tmp_path / "figures/check_controlnet_default.jpg").exists()
    with pytest.raises(ValueError, match="val only"):
        monkeypatch.setattr(sys, "argv", ["check", "--split", "test"])
        check.main()


def tiny_pipeline():
    """The real diffusers StableDiffusionControlNetPipeline with tiny random parts and an IP-Adapter attached the
    way `load_ip_adapter` does it (attention processors + `encoder_hid_proj`). No weights are downloaded."""
    from diffusers import (AutoencoderKL, ControlNetModel, DPMSolverMultistepScheduler, StableDiffusionControlNetPipeline,
                           UNet2DConditionModel)
    from diffusers.models.attention_processor import AttnProcessor2_0, IPAdapterAttnProcessor2_0
    from diffusers.models.embeddings import MultiIPAdapterImageProjection
    from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModelWithProjection

    torch.manual_seed(0)
    cross, widths = 16, (16, 32)
    unet = UNet2DConditionModel(block_out_channels=widths, layers_per_block=1, sample_size=64, cross_attention_dim=cross,
                                down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
                                up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"), norm_num_groups=8,
                                attention_head_dim=2)
    controlnet = ControlNetModel.from_unet(unet)
    vae = AutoencoderKL(block_out_channels=(8, 8, 8, 8), norm_num_groups=8, latent_channels=4,
                        down_block_types=("DownEncoderBlock2D",) * 4, up_block_types=("UpDecoderBlock2D",) * 4)
    procs = {}
    for name in unet.attn_processors:
        if name.endswith("attn1.processor"):
            procs[name] = AttnProcessor2_0()
            continue
        if name.startswith("mid_block"):
            hidden = widths[-1]
        else:
            hidden = (list(reversed(widths)) if name.startswith("up_blocks") else widths)[int(name.split(".")[1])]
        procs[name] = IPAdapterAttnProcessor2_0(hidden, cross, num_tokens=(4,), scale=1.0)
    unet.set_attn_processor(procs)
    for p in unet.attn_processors.values():
        for m in getattr(p, "to_k_ip", []) + getattr(p, "to_v_ip", []):
            nn.init.normal_(m.weight, std=0.5)
    unet.encoder_hid_proj = MultiIPAdapterImageProjection([ImageProjection(8, cross, 4)])
    unet.config.encoder_hid_dim_type = "ip_image_proj"
    image_encoder = CLIPVisionModelWithProjection(CLIPVisionConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                                                                   num_attention_heads=2, image_size=32, patch_size=8,
                                                                   projection_dim=8))
    pipe = StableDiffusionControlNetPipeline(
        vae=vae, text_encoder=tiny_text_encoder(), tokenizer=FakeTokenizer(), unet=unet, controlnet=controlnet,
        scheduler=DPMSolverMultistepScheduler(algorithm_type="dpmsolver++"), safety_checker=None,
        feature_extractor=CLIPImageProcessor(size={"shortest_edge": 32}, crop_size={"height": 32, "width": 32}),
        image_encoder=image_encoder, requires_safety_checker=False)
    pipe.set_progress_bar_config(disable=True)
    return pipe


@pytest.mark.parametrize("guidance", [5.0, 1.0])
def test_real_pipeline_one_reference_per_sample(guidance):
    """Regression: diffusers repeats ip_adapter_image_embeds by the batch size, so without
    keep_per_sample_ip_embeds every sample attended to every reference in its batch (Kaggle v5)."""
    colorizer = ControlNetColorizer(ControlNetConfig(device="cpu", dtype="float32", steps=2, guidance_scale=guidance),
                                    pipe=tiny_pipeline())
    torch.manual_seed(1)
    sketch = torch.rand(3, 1, 256, 256) * 2 - 1
    reference = torch.stack([torch.full((3, 256, 256), v) for v in (-0.9, 0.0, 0.9)])
    batched = colorizer.colorize(sketch, reference, seed=0)
    alone = torch.cat([colorizer.colorize(sketch[i : i + 1], reference[i : i + 1], seed=0) for i in range(3)])
    assert torch.allclose(batched, alone, atol=1e-3)  # float noise ~1e-5; the leak was ~0.1

    del colorizer.pipe.prepare_ip_adapter_image_embeds  # back to diffusers' own: the leak returns
    assert not torch.allclose(colorizer.colorize(sketch, reference, seed=0), alone, atol=1e-3)
    keep_per_sample_ip_embeds(colorizer.pipe)
    colorizer.set_params(ip_adapter_scale=0.0)
    assert not torch.allclose(colorizer.colorize(sketch, reference, seed=0), batched, atol=1e-3)  # the reference is used


def test_config_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown"):
        ControlNetConfig.from_dict({"attention_slicing": True})
