# MobileNet-v2 compression | CS6886 Assignment 2

Train MobileNet-v2 on CIFAR-10, compare manual quantization settings, then test one chosen model.

**Test accuracy:** baseline 92.64%; W4A8 90.13%. **W4A8 size:** 1.305497 MB, with 6.96x model compression, 7.12x weights-only compression and 3.90x estimated activation compression.

Start with the [submission report](report/Assignment_2_Report.pdf) or [results](results/README.md), then read the source in this order:

| File or folder | Purpose |
|---|---|
| `src/data.py` | Prepare the split and normalization |
| `src/train.py` | Build, train and resume the baseline |
| `src/compression.py` | Quantize, fold BN and pack bits |
| `src/evaluate.py` | Compare settings (`sweep`) or test the chosen model (`final`) |
| `src/report.py` | Check results and publish the W&B chart |
| `tests/` | Quantization, file loading and checkpoint tests |
| `models/` | One baseline and one chosen compressed model |
| `results/` | Measurements, curves, split indices and verification |
| `report/` | Submission PDF |

## Setup and quick check

Tested on Windows, Python 3.14.3 and an RTX 3050 Laptop GPU (4 GB). Dependencies are pinned in `requirements.txt`, including torch 2.14.0+cu126, torchvision 0.29.0+cu126 and numpy 2.5.2. A shallow clone skips older experiment files.

```powershell
git clone --depth 1 https://github.com/harshith0518/cs6886-assignment-2-mobilenetv2.git
cd cs6886-assignment-2-mobilenetv2
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m src.data --download
python -m unittest discover -s tests -v
python -m src.report
python -m src.evaluate final
```

Run commands from the repository root. If activation is unavailable, replace `python` with `.\.venv\Scripts\python.exe`. Evaluation detects CUDA; `--device cpu` also works. Setup and the first dataset download need internet. Experiments then work offline.

New experiments go to ignored `runs/`. A completed final evaluation is reused. For a fresh check, add `--output-dir runs/final_recheck` with a new folder name.

## Reproduce Q3 and Q4

Reuse the supplied baseline; no retraining is needed. The grid tests weight and activation bits of 8, 4 and 2. It calibrates on 1,000 training images and evaluates all 5,000 validation images, balanced by class.

```powershell
python -m src.evaluate sweep
python -m src.report --results-dir runs/sweep --check-files
python -m src.evaluate final --sweep-dir runs/sweep --output-dir runs/final_reproduced
```

Each sweep needs a new output folder; use `--output-dir` to name it. All nine packed models are regenerated there. For a smaller experiment, pass `--settings 8/8 4/8`. Other options include `--batch-size`, `--cpu-threads`, `--quantize-edges` and `--no-fold-bn`. The report checker expects the full assignment grid and its original layer settings.

<details>
<summary>Train or resume a new baseline</summary>

```powershell
python -m src.train --epochs 300 --batch-size 128 --lr 0.1 --amp
# Continue an interrupted run:
python -m src.train --epochs 300 --resume
```

Training defaults to `runs/baseline`. Resume restores the model, optimizer, AMP scaler and random states. Keep the original total of 300 epochs. After training, add `--run-dir runs/baseline` to each Q3/Q4 command and use new output folders.

</details>

<details>
<summary>View or publish the native W&B chart</summary>

The [native Parallel Coordinates chart](https://wandb.ai/harshith7946-indian-institute-of-technology-madras/cs6886-assignment2-q3/reports/Q3---MobileNet-v2-compression-results--VmlldzoxNzg4ODQ3OQ==) contains the original ten runs. To publish a regenerated sweep:

```powershell
wandb login
python -m src.report --results-dir runs/sweep --check-files --wandb-mode online --entity YOUR_WANDB_ENTITY
```

Replace `YOUR_WANDB_ENTITY` with your account or team. Add `--run-dir runs/baseline` for a new baseline. `--wandb-mode offline` saves local logs; publishing needs internet. TAs need access to both private projects.

</details>

<details>
<summary>Settings and measurement definitions</summary>

- **Data:** 45,000 training / 5,000 validation / 10,000 test images. Split and calibration seed 42; validation order seed 43. Normalization uses training pixels only.
- **Training:** width 1.0, dropout 0.2, first convolution stride 1; BN epsilon 1e-5 and momentum 0.1. SGD uses momentum 0.9, Nesterov, weight decay 5e-4 excluding BN/bias, label smoothing 0.05 and gradient clipping at 5. Learning rate 0.1 warms up for 5 epochs, then follows cosine decay to 1e-5. Full settings: [baseline.json](results/baseline.json). cuDNN is deterministic, with benchmarking disabled; other hardware or versions can change results.
- **Compression:** symmetric per-output-channel weights; affine min/max activations at Conv/Linear inputs. First convolution and classifier weights/inputs, and all biases, stay FP32. BN folding and bit packing are manual. No compression API or fine-tuning is used.
- **Selection:** Q3 uses training calibration and validation comparisons. Q4 records the smallest model with at least 90% validation accuracy before testing it.
- **Model ratio:** original parameters and buffers / complete packed file bytes. Q3 calls this `weight_compression_ratio`. Q4 separately measures Conv/Linear weights, counting scales, FP32 edge weights and the whole header. MB means 1,000,000 bytes.
- **Activation ratio:** summed packed Conv/Linear input occurrences per image, plus 8 bytes per quantized location for scale and zero point. Repeated occurrences count. This estimates storage, not peak GPU memory. Inference decodes weights and simulates activation quantization in FP32.

</details>

References: [MobileNet-v2](https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py), [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), [quantization](https://arxiv.org/abs/1712.05877), [W&B panels](https://docs.wandb.ai/models/app/features/panels/parallel-coordinates).
