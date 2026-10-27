# Reference-Based Anime Line-Art Colorization

A controlled comparison of a GAN, a diffusion-type model trained from scratch, and a large pretrained
diffusion model on the same task, data, and metrics: given a line-art sketch and a color reference
image of the same character, color the sketch so it matches the reference.

**Status (27 Oct 2026):** after checkpoint 1. Setup, Model C (SD1.5 + ControlNet + IP-Adapter),
the shared evaluation suite, tuning/LoRA scripts, plots, the user-study tooling and all shared
infrastructure (data loader, splits/pairs, sketch clean-up, checkpointing, `sample.py`, latent cache)
are written and unit-tested (`uv run pytest -q`), and the data pipeline was checked on a small local
sample of the dataset. **The full dataset, splits and pair files are ready on Kaggle** (see
[Using the dataset](#using-the-dataset)). Models A and B are still to do; no model has run with real
weights yet, so there are no results yet. See [`Reference-Based Anime Line-Art Colorization.md`](<Reference-Based Anime Line-Art Colorization.md>)
for the full design document — task definition, dataset, model architectures, evaluation plan, task
assignment, and timeline.

## Using the dataset

The full dataset lives on Kaggle as the output of the private notebook
[`hydrochii/anime-colorization-data`](https://www.kaggle.com/code/hydrochii/anime-colorization-data)
(version 3, run on 27 Oct 2026). It downloaded the Hugging Face mirror
`ComputerVisionAnimeProject/AnimeFaceColorization` once, made the splits, and saved everything as the
notebook's output. You never need to download the mirror again.

**Access:** the notebook is private. Send your Kaggle username to the notebook owner, who adds you under
*Share → Collaborators*. After that it shows up when you add inputs to your own notebooks.

### What is in the output

| File | What it is |
| --- | --- |
| `AnimeFaceColorization.tar` | The whole mirror (8.6 GB, uncompressed tar): `sketch/<group>/…` and `reference/<group>/…`, 256×256 |
| `splits/train.txt`, `val.txt`, `test.txt`, `test_cross.txt` | Image ids per split |
| `splits/cross_ref_pairs.csv`, `shuffled_ref_pairs.csv` | Which reference goes with which sketch in `cross_ref` / `shuffled_ref` mode |
| `stats.txt`, `previews/*.png` | Dataset counts, and sketch / target / reference grids for each mode |

What the data contains:

| | Count |
| --- | --- |
| Sketch–color pairs | 106,134 in 30,752 groups (a group is one folder) |
| AnimeDiffusion groups | 30,579 (each: 1 original drawing + 2 deformed copies) |
| moeimouto characters | 173 (14,397 drawings; 32–161 per character, median 80) |

The splits (made by `scripts/make_splits.py` with a fixed seed; never regenerate them yourself):

| Split | Images | Used for |
| --- | --- | --- |
| `train` | 98,123 (29,133 groups, deformed copies included) | Training |
| `val` | 1,000 | All tuning and checkpoint selection |
| `test` | 579 AnimeDiffusion originals | `self_ref` and `shuffled_ref` evaluation |
| `test_cross` | 600 (40 held-out moeimouto characters × 15 drawings) | `cross_ref` evaluation |

A whole group always stays in one split, so deformed copies of a test image never appear in training,
and none of the 40 `test_cross` characters is in training.

`test` holds 579 randomly chosen AnimeDiffusion originals, not the official AnimeDiffusion test list (we
do not have it yet). When it arrives the splits are remade once with `--test-ids` and the team is told.

### In a Kaggle notebook

1. In the notebook editor: *Add Input → Notebooks →* search `anime-colorization-data` → *Add*.
2. Turn on *Internet* (needed for `git clone` and `pip`).
3. Run this setup cell first:

```python
import glob, os

!git clone -q -b feat/nhpthinh https://github.com/hidrochin/IT5413-_Anime-line-art-colorization_Project.git /kaggle/working/repo
%cd /kaggle/working/repo
!pip install -q -r requirements.txt

# Unpack the data to /kaggle/temp (not saved with the notebook, so it does not fill your output).
TAR = glob.glob("/kaggle/input/**/AnimeFaceColorization.tar", recursive=True)[0]
!mkdir -p /kaggle/temp && tar -xf {TAR} -C /kaggle/temp
os.environ["ANIME_DATA_ROOT"] = "/kaggle/temp/AnimeFaceColorization"
os.environ["ANIME_SPLITS_DIR"] = os.path.join(os.path.dirname(TAR), "splits")
```

Set the two environment variables **before** importing anything from `src`: the loader reads them at
import time. They are also passed on to `!python -m scripts.…` commands run from the same notebook.
`ANIME_SPLITS_DIR` points at the splits in the attached output; once `splits/` is committed to the repo
you can drop that line and the committed copy (the same files) is used.

### Loading samples

Every script and model uses the same loader, `build_dataset(split, mode)`:

```python
from src.data import build_dataset

train = build_dataset("train", "train")      # reference = target with a random warp + flip + crop
val = build_dataset("val", "self_ref")       # fixed warp per image: use for validation
test = build_dataset("test", "self_ref")     # also "shuffled_ref"; test_cross uses "cross_ref"

x = train[0]
# x["id"]: str; x["sketch"]: (1,256,256); x["target"]: (3,256,256); x["reference"]: (3,256,256); all in [-1, 1]
```

Wrap it in a normal `torch.utils.data.DataLoader`. Every sketch already goes through the shared clean-up
(black lines on white, `src/data/sketch.py`), so do not add your own. To see what the loader returns:
`python -m scripts.preview_dataset --split train --mode train`.

### Working locally

Do not download the 8.6 GB tar to your laptop. For smoke tests, download a few folders of the mirror into
`data/AnimeFaceColorization/` (the default data root) and run against them, or run `uv run pytest -q`, which
uses fake data and downloads nothing.

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
