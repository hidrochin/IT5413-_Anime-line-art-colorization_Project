# Reference-Based Anime Line-Art Colorization

A controlled comparison of a GAN, a diffusion-type model trained from scratch, and a large pretrained
diffusion model on the same task, data, and metrics: given a line-art sketch and a color reference
image of the same character, color the sketch so it matches the reference.

**Status (26 Oct 2026):** after checkpoint 1. Setup, Model C (SD1.5 + ControlNet + IP-Adapter),
the shared evaluation suite, tuning/LoRA scripts, plots and the user-study tooling are written and
unit-tested (`uv run pytest -q`). Nothing has run on Kaggle yet. The shared data loader, Models A/B
and `sample.py` are still to do. No results yet. See [`Reference-Based Anime Line-Art Colorization.md`](<Reference-Based Anime Line-Art Colorization.md>)
for the full design document — task definition, dataset, model architectures, evaluation plan, task
assignment, and timeline.

## At a glance

| | |
| --- | --- |
| Task | Line-art sketch + color reference → colored anime face |
| Data | [AnimeDiffusion benchmark](https://xq-meng.github.io/projects/AnimeDiffusion/): 31,696 train / 579 test pairs, 256×256 |
| Model A | Reference-conditioned pix2pix GAN, trained from scratch |
| Model B | Latent flow-matching model with DINOv2 reference cross-attention, trained from scratch |
| Model C | Stable Diffusion 1.5 + ControlNet (line-art) + IP-Adapter, pretrained, zero-shot / optional LoRA |
| Main question | Do diffusion-type models trade fidelity for diversity better than a GAN? |
| Compute | Free Kaggle GPUs (T4/P100), 3 team members |

## Planned repository structure

No code exists yet; this is the target layout as it gets built out (see Section 10 of the design doc):

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
