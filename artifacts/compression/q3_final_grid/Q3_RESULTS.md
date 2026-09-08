# Question 3 - Compression results

## (a) Setup

The grid uses 8, 4 and 2 bits for weights and activations, giving nine compressed models and one FP32 baseline. All settings use the same checkpoint, chosen at epoch 284. There is no fine-tuning.

The first convolution and final classifier weights and inputs stay FP32. The other Conv/Linear weights use per-channel symmetric quantization, after manually folding BatchNorm. Their inputs use affine quantization. Biases stay FP32.

Calibration uses 1,000 training images (100 per class, seed 42) without augmentation. Evaluation uses all 5,000 validation images (500 per class, order seed 43). The saved indices are disjoint. No test images are used for Q3.

## (b) Accuracy comparison

| Setting | Validation accuracy (%) | Drop (pp) | Model size (MB) | Model ratio | Activation ratio |
|---|---:|---:|---:|---:|---:|
| FP32 | 93.88 | 0.00 | 9.083592 | 1.00x | 1.00x |
| W8A8 | 93.88 | 0.00 | 2.399973 | 3.78x | 3.90x |
| W8A4 | 90.16 | 3.72 | 2.399343 | 3.79x | 7.57x |
| W8A2 | 10.00 | 83.88 | 2.399330 | 3.79x | 14.25x |
| W4A8 | 90.96 | 2.92 | 1.305497 | 6.96x | 3.90x |
| W4A4 | 86.86 | 7.02 | 1.304877 | 6.96x | 7.57x |
| W4A2 | 10.00 | 83.88 | 1.304871 | 6.96x | 14.25x |
| W2A8 | 10.26 | 83.62 | 0.758247 | 11.98x | 3.90x |
| W2A4 | 9.94 | 83.94 | 0.757684 | 11.99x | 7.57x |
| W2A2 | 10.04 | 83.84 | 0.757675 | 11.99x | 14.25x |

The drop is measured against FP32 on the same validation split, in percentage points. A larger compression ratio is only useful if enough accuracy is retained.

![Validation comparison](q3_accuracy_compression.png)

Model ratio = original parameters and buffers in bytes / complete packed file bytes. This includes the effect of BatchNorm folding, scales, unchanged tensors and the JSON header. The legacy JSON/W&B metric name is `weight_compression_ratio`; this is a whole-model storage measurement. Q4 separately calculates a weights-only ratio.

Activation ratio compares the sum of FP32 Conv/Linear input occurrences per image with estimated packed inputs plus 8 bytes per quantized location for scale and zero point. Repeated occurrences are included. It is not a measurement of peak GPU memory. Inference still uses FP32 arithmetic with quantize/dequantize steps.

All nine files were reloaded and matched their pre-export logits exactly on a validation batch. The report script also checks the source checkpoint hash, saved split membership, file sizes and ratio arithmetic. Results come from one training run and one calibration selection.

## W&B Parallel Coordinates chart

[Native W&B Parallel Coordinates chart](https://wandb.ai/harshith7946-indian-institute-of-technology-madras/cs6886-assignment2-q3/reports/Q3---MobileNet-v2-compression-results--VmlldzoxNzg4ODQ3OQ==)

The native panel has six axes: weight bits, activation bits, model ratio, activation ratio, model size and validation accuracy. Each setting is a separate W&B run. The image below is a local companion using the same saved measurements.

![Local companion chart](q3_parallel_coordinates_local.png)

## Commands

See README.md for the full sweep command and environment setup. To check the saved run and redraw the figures:

```powershell
python report_q3.py --results-dir artifacts/compression/q3_final_grid --wandb-mode none
# To publish a new native chart after logging in:
wandb login
python report_q3.py --results-dir artifacts/compression/q3_final_grid --wandb-mode online --entity YOUR_WANDB_ENTITY
```

For a new training run, also pass `--run-dir` with that run's folder. Online logging sends numerical results and configuration, not images or checkpoints.

Baseline SHA-256: `d25c1a9daa9de1da2101ae2dd4ec9b7a2524b4828e69803273c6d67d861ad06d`.
