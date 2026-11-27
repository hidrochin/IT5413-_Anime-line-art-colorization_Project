# Model A notebook validation

The notebook has 31 cells: 17 Markdown explanations and 14 Python code cells.
Its only dataset source is attached Kaggle Input: an extracted directory or a ZIP.
There is no HF dataset download, token, Secrets or authentication code.

Validated with `python tests/test_notebook_input.py`:

- Directory input: correct paired intersection and grouped splits; source files untouched.
- ZIP input: extraction, matching split signature, and reuse without re-extraction.
- An incorrect input path produces an actionable Kaggle Input error.
- Dataset download/authentication code is absent from the notebook.

Validated with `python tests/smoke_model_a_notebook.py` on CPU synthetic data:

- Notebook schema/syntax and default-width U-Net/PatchGAN forward shapes.
- GAN backward/optimizer steps and validation on a smaller 8-channel network.
- Epoch-boundary and mid-epoch resume match uninterrupted training with zero workers.
- Recovery in a new run with only `last.pt`, no previous preview PNGs, and
  empty saved scaler state loaded into an enabled CPU GradScaler.
- Real VGG19 feature/loss gradient code using random weights, without downloads.
- Combined GAN + L1 + VGG19-feature loss backward and optimizer update.
- Seed reproducibility, dropout randomness and frozen BatchNorm during inference.
- Self/cross-reference PNG export, exact shared-reference round trip and PSNR/SSIM.
- Real `lpips` 0.1.4 ground-truth/diversity computation, with a random AlexNet
  backbone to avoid downloading weights; numerical pipeline testing only.
- Unpaired files excluded without deletion, source variants grouped in one split,
  source-image folder IDs excluded from character identities.

The selected local subset was inspected on 13 Nov 2026: 80,345 matching pairs,
with source-group splits of 64,143 train / 7,996 validation / 8,206 test.
The notebook recomputes counts from the attached dataset and exports the shared splits.

Environment used: Python 3.13.12, torch 2.14.1+cpu, torchvision 0.29.1+cpu.
Not verified here: GPU/CUDA AMP execution, Kaggle VRAM/throughput, pretrained
VGG19/LPIPS backbone weight downloads or quality scores, multiworker bitwise replay, and
model quality on the real dataset. Synthetic scores are not research results.

Review fixes: skip displaying an unavailable preview from a previous run;
initialize a fresh AMP scaler when the saved scaler state is empty; and use an
available last checkpoint when the best checkpoint was not retained. The latter
is explicitly recorded as `checkpoint_selection: last_available` in metric output.
Normal runs still prefer `best.pt`. The checkpoint filename/selection is recorded
so a last checkpoint is not silently reported as the historical best model.
