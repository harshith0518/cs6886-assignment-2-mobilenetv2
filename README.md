# MobileNet-v2 compression: CS6886 Assignment 2

Baseline: **92.64% test accuracy**. Chosen W4A8 model: **90.13%**, **1.305497 MB**, **6.96x model compression**. Its weights-only ratio is **7.12x** and estimated activation ratio is **3.90x**, including the stated overheads.

## Read the code in this order

| File | What it does |
|---|---|
| `prepare_data.py` | Split CIFAR-10 and calculate training-only normalization |
| `train_baseline.py` | Build MobileNet-v2, train, save and resume checkpoints |
| `manual_compression.py` | Quantization equations, BN folding and bit packing |
| `run_compression.py` | Calibrate and compare the nine compression settings |
| `evaluate_final.py` | Select one setting using validation and evaluate its test accuracy |
| `report_q3.py` | Verify results and optionally publish the native W&B chart |

`artifacts/` contains the measured results, curves, split indices, **one baseline checkpoint** and **one final compressed model**. Intermediate models are regenerated when needed. The two `test_*.py` files check quantization, serialization and safe checkpoint writes. The PDF remains local for review.

## Setup and check the supplied model

Tested on Windows, Python **3.14.3**, RTX 3050 Laptop GPU (4 GB), torch **2.14.0+cu126**, torchvision **0.29.0+cu126** and numpy **2.5.2**. All directly used dependencies are pinned in `requirements.txt`. A shallow clone downloads the compact version without the older experiment files:

```powershell
git clone --depth 1 https://github.com/harshith0518/cs6886-assignment-2-mobilenetv2.git
cd cs6886-assignment-2-mobilenetv2
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python prepare_data.py --download
python -m unittest test_manual_compression test_training_io
python report_q3.py
python evaluate_final.py --device cuda --output-dir artifacts/final_recheck
```

Use `.\.venv\Scripts\python.exe` instead of `python` if virtual-environment activation is unavailable. Evaluation also supports `--device cpu`. Use a new evaluation output directory for a fresh run; a matching completed result is otherwise reused. Installation and the first dataset download need internet; the model experiments then work offline.

## Reproduce Q3 and Q4 without retraining

The full 3 x 3 grid is the default: weight bits and activation bits each vary over 8, 4 and 2. Calibration uses 100 training images per class; evaluation uses all 500 validation images per class.

```powershell
python run_compression.py --output-dir artifacts/compression/reproduced --device cuda --batch-size 128 --cpu-threads 4
python report_q3.py --results-dir artifacts/compression/reproduced --check-files
python evaluate_final.py --sweep-dir artifacts/compression/reproduced --output-dir artifacts/final_reproduced --device cuda
```

The compression output directory must be new. `--check-files` requires all nine regenerated model files; the default report command checks saved metrics and the retained W4A8 file. For a smaller experiment, pass `--settings 8/8 4/8`. `--quantize-edges` and `--no-fold-bn` change the layer exceptions and folding choice.

## Train a new baseline (optional)

```powershell
python train_baseline.py --epochs 300 --batch-size 128 --lr 0.1 --amp --run-dir artifacts/training/reproduced
# To continue that run after an interruption:
python train_baseline.py --epochs 300 --resume --run-dir artifacts/training/reproduced
```

After training finishes, use the Q3/Q4 commands above with a new output folder and add `--run-dir artifacts/training/reproduced` to **each** command. Resume preserves model, optimizer, AMP scaler and random-generator states; keep the original total of 300 epochs.

## Results and W&B

See [the complete comparison](artifacts/compression/q3_final_grid/Q3_RESULTS.md), [final measurements](artifacts/final/results.json), and the [native W&B Parallel Coordinates chart](https://wandb.ai/harshith7946-indian-institute-of-technology-madras/cs6886-assignment2-q3/reports/Q3---MobileNet-v2-compression-results--VmlldzoxNzg4ODQ3OQ==). To publish a chart for a regenerated sweep:

```powershell
wandb login
python report_q3.py --results-dir artifacts/compression/reproduced --check-files --wandb-mode online --entity YOUR_WANDB_ENTITY
```

Replace `YOUR_WANDB_ENTITY` with your account/team. Add `--run-dir` as above if using a new baseline. `--wandb-mode offline` records local logs; creating the native online panel needs internet. Give the evaluator access to the private GitHub repository and W&B project before submission.

## Settings and measurement definitions

- Training/data split seed: **42**. Calibration seed: **42**. Validation order seed: **43**. Split sizes: **45,000 / 5,000 / 10,000**. cuDNN deterministic mode is enabled for baseline training; benchmark mode is disabled. Hardware/version differences can affect exact reproduction.
- MobileNet-v2: width 1.0, dropout 0.2, first convolution stride 1, BN epsilon 1e-5 and momentum 0.1. SGD uses momentum 0.9, Nesterov, weight decay 5e-4 excluding BN/bias, label smoothing 0.05, and a 5-epoch warmup followed by cosine decay. Full settings are saved in the training configuration.
- Weights use symmetric per-output-channel quantization; activations use affine min/max quantization at Conv/Linear inputs. The first convolution and classifier weights/inputs and all biases stay FP32. BN is folded manually. There is no compression API or fine-tuning.
- Q3 uses training-only calibration and validation comparisons. Q4 saves its choice before test evaluation: the smallest model retaining at least 90% validation accuracy. No test result chooses the configuration.
- Model ratio = original parameters and buffers / whole packed file bytes. The Q3 JSON/W&B name `weight_compression_ratio` refers to this model ratio. Q4 separately counts Conv/Linear weights, scales, FP32 edge weights and the whole header for its weights-only ratio. MB means 1,000,000 bytes.
- Activation ratio estimates the sum of packed Conv/Linear input occurrences for one image, including 8 bytes per quantized location for scale and zero point. Repeated occurrences are counted. It is not peak GPU memory. Inference decodes weights and simulates activation quantization in FP32.

References: [MobileNet-v2 source](https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py), [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), [quantization paper](https://arxiv.org/abs/1712.05877), [W&B parallel coordinates](https://docs.wandb.ai/models/app/features/panels/parallel-coordinates).
