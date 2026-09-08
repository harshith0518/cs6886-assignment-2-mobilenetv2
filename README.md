# CS6886 Assignment 2: MobileNet-v2 on CIFAR-10

This project trains a MobileNet-v2 baseline and compresses its weights and activations using manually written quantization and bit packing. The editable Q1-Q5 draft is in `output/Assignment_2_Answers.md`. The PDF is kept locally for review and will be added after approval.

The baseline reached **92.64% test accuracy**. The final model, **W4A8**, reached **90.13%** on the same 10,000 test images and occupies **1.305497 MB**. It was chosen as the smallest saved model with at least 90% validation accuracy, before evaluating its test accuracy.

| Final measurement | Value |
|---|---:|
| Validation accuracy | 90.96% |
| Test accuracy | 90.13% |
| Weights-only compression, including scales and all header overhead | 7.12x |
| Complete model storage compression | 6.96x |
| Estimated activation compression | 3.90x |
| Test accuracy drop from baseline | 2.51 percentage points |

[Native W&B Parallel Coordinates chart](https://wandb.ai/harshith7946-indian-institute-of-technology-madras/cs6886-assignment2-q3/reports/Q3---MobileNet-v2-compression-results--VmlldzoxNzg4ODQ3OQ==)

## Files

| File | Purpose |
|---|---|
| `prepare_data.py` | Download CIFAR-10, reproduce the class-balanced split and training-only RGB statistics |
| `train_baseline.py` | Train, save checkpoints and resume an interrupted run |
| `manual_compression.py` | Quantization, BatchNorm folding, activation hooks and packed file I/O |
| `run_compression.py` | Calibrate and compare compression settings on validation images |
| `evaluate_final.py` | Record the Q4 choice and evaluate the chosen file on the test set |
| `report_q3.py` | Check saved measurements, draw figures and publish the native W&B panel |
| `build_report.py` | Generate the PDF and its editable Markdown answer file |
| `test_manual_compression.py`, `test_training_io.py` | Numerical, serialization and checkpoint regression checks |
| `cifar10_mobilenetv2_exploration.ipynb` | Original data and model exploration |

Saved checkpoints, split indices, training history, all nine packed Q3 models and the Q4 results are in `artifacts/`. CIFAR-10, the virtual environment, personal credentials, temporary files and W&B internal logs are excluded. `last.pt` is also excluded from the repository; it is only needed to resume an unfinished training run.

## Environment

The reported run used Windows, Python **3.14.3**, an NVIDIA GeForce RTX 3050 Laptop GPU (4 GB), and CUDA-enabled PyTorch. Training used batch size 128, four CPU threads and zero data-loader workers. The commands below use PowerShell from the repository root.

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-notebook.txt -r requirements-q3.txt -r requirements-report.txt
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

The requirement files pin the directly used packages, including torch 2.14.0+cu126, torchvision 0.29.0+cu126, numpy 2.5.2, matplotlib 3.11.1, wandb 0.29.0 and wandb-workspaces 0.4.11. `environment-lock.txt` records the full installed environment. If activation is disabled by local PowerShell policy, use `.\.venv\Scripts\python.exe` in place of `python` without changing the policy.

## Check the supplied results

Internet is needed for package installation, the first dataset download and online W&B logging. Evaluation and compression can run offline once the dependencies and data are present.

```powershell
python prepare_data.py --download
python -m unittest test_manual_compression test_training_io
python report_q3.py --wandb-mode none
python evaluate_final.py --device cuda --output-dir artifacts/final_recheck
```

The last command reloads the saved W4A8 file and evaluates all 10,000 test images. Use `--device cpu` if CUDA is unavailable; CPU evaluation is slower. An existing output folder with matching final results is reused. Choose a new output folder when an actual fresh evaluation is wanted. The recorded submission results are in `artifacts/final/results.json` and are not replaced by this command.

## Reproduce the full experiment

Run the environment and data setup above first. The following commands write new results into separate folders and preserve the supplied measurements.

```powershell
python train_baseline.py --epochs 300 --batch-size 128 --lr 0.1 --amp --run-dir artifacts/training/reproduced
```

If that training run is interrupted, resume it with the same target epoch count:

```powershell
python train_baseline.py --epochs 300 --resume --run-dir artifacts/training/reproduced
```

The resume file includes model weights, optimizer, AMP scaler, learning-rate progress and random generator states. Training chooses `best.pt` using validation accuracy, then evaluates the test set after epoch 300. Compression reuses that checkpoint; it does not retrain the baseline.

```powershell
python run_compression.py --run-dir artifacts/training/reproduced --output-dir artifacts/compression/reproduced --device cuda --batch-size 128 --cpu-threads 4 --calibration-per-class 100 --validation-per-class 500 --settings 8/8 8/4 8/2 4/8 4/4 4/2 2/8 2/4 2/2
python report_q3.py --run-dir artifacts/training/reproduced --results-dir artifacts/compression/reproduced --wandb-mode none
python evaluate_final.py --run-dir artifacts/training/reproduced --sweep-dir artifacts/compression/reproduced --output-dir artifacts/final_reproduced --min-validation-accuracy 90 --device cuda
```

The compression output directory must be new. The Q3 reporting script checks the full 3 x 3 grid, 1,000 calibration images and all 5,000 validation images. For a smaller exploratory sweep, `run_compression.py` also accepts settings such as `--settings 8/8 4/8`, but this does not constitute the complete Q3 comparison.

To publish a native chart for the new sweep, log into your own W&B account:

```powershell
wandb login
python report_q3.py --run-dir artifacts/training/reproduced --results-dir artifacts/compression/reproduced --wandb-mode online --entity YOUR_WANDB_ENTITY --project cs6886-assignment2-q3
```

Use the team/account name in place of `YOUR_WANDB_ENTITY`. Offline logging is available with `--wandb-mode offline`, but the native online chart requires connectivity. The existing submission chart is already published at the link above.

## Seeds and measurement details

The data split uses NumPy seed 42, keeping 4,500 training and 500 validation images per class. Training sets Python, NumPy, PyTorch, CUDA and loader seeds to 42. cuDNN deterministic mode is enabled and benchmark mode is disabled. Compression calibration uses seed 42; validation order uses seed 43. Exact results can still vary with hardware or package versions.

Q3 calibrates on 1,000 training images without augmentation and compares all settings on the same 5,000 validation images. It uses no test images. Q4 first saves its selection rule and chosen checkpoint hash, then evaluates that one compressed model on the 10,000-image test set. No compression fine-tuning was done.

Weights use symmetric per-output-channel quantization. Activations use affine min/max quantization at Conv/Linear inputs. The first convolution and final classifier weights and inputs remain FP32; biases remain FP32. BatchNorm is folded manually. `--quantize-edges` and `--no-fold-bn` allow experiments with these choices. No ready-made compression or quantization API is used.

The `.mq` files actually pack weight codes into bytes. Their reported sizes count scales, raw tensors and the complete header. The Q3 field named `weight_compression_ratio` is the original model's parameters and buffers divided by the whole packed file; it is a model storage ratio. Q4 also reports a distinct weights-only ratio, conservatively charging the whole header to the weights. MB means 1,000,000 bytes.

Activation compression is an estimated storage ratio: sum the Conv/Linear input occurrences for one image, compare their FP32 bytes with packed bytes, and add eight bytes per quantized location for a scale and zero point. Repeated occurrences are counted. This is not peak RAM or GPU memory. Inference decodes weights to FP32 and simulates activation quantization in FP32; no integer-kernel speed or live-memory reduction is claimed.

## Rebuild the submission

```powershell
python build_report.py
```

This uses the supplied measurements to write a local draft at `output/pdf/Assignment_2_Report.pdf` and update `output/Assignment_2_Answers.md`. PDF files are currently excluded from Git while the report is being reviewed. `submission_info.json` holds the author and repository URL. The PDF is the file to upload to Moodle. If the GitHub repository or W&B project is private, give the evaluator access before submission.

## References

- [torchvision MobileNet-v2 implementation](https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py)
- [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877)
- [W&B Parallel Coordinates](https://docs.wandb.ai/models/app/features/panels/parallel-coordinates)

PyTorch and torchvision provide the model and tensor operations. The quantization, folding and packing routines are implemented in `manual_compression.py`.
