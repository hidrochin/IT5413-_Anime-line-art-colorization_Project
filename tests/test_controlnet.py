"""Model C wiring against a fake pipeline (no SD weights are downloaded)."""

from types import SimpleNamespace

import pytest
import torch
from diffusers.models.embeddings import ImageProjection

from src.models.controlnet.colorizer import ControlNetColorizer, ControlNetConfig, downsample, prepare_control


class FakePipe:
    def __init__(self):
        proj = ImageProjection(image_embed_dim=8, cross_attention_dim=8, num_image_text_embeds=1)
        self.unet = SimpleNamespace(encoder_hid_proj=SimpleNamespace(image_projection_layers=[proj]))
        self.device = torch.device("cpu")
        self.ip_scale = None
        self.calls = []

    def set_ip_adapter_scale(self, scale):
        self.ip_scale = scale

    def encode_image(self, images, device, n, hidden):
        assert not hidden and len(images) > 0
        pos = torch.ones(len(images), 8)
        return pos, torch.zeros_like(pos)

    def __call__(self, **kw):
        self.calls.append(kw)
        b = len(kw["prompt"])
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


def test_config_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown"):
        ControlNetConfig.from_dict({"attention_slicing": True})
