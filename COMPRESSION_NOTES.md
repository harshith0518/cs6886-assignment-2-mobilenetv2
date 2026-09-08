# Manual compression: what we have implemented

This is post-training quantization: take a trained model, reduce its numerical precision, and measure what changed. The baseline training job does not need to restart.

## Files to use

- `manual_compression.py`: our quantization equations, BatchNorm folding, bit packing, export and reload.
- `run_compression.py`: calibration and validation experiments using a snapshot of the baseline.
- `test_manual_compression.py`: checks for numerical errors, bit packing, folding and export/reload.
- `cifar10_mobilenetv2_exploration.ipynb`: short examples and controls, with code cells only.

PyTorch supplies tensors, layers and hooks. NumPy supplies arrays. We do **not** call a ready-made quantization, compression, pruning or model-fusion API.

## How it works

1. **Copy the saved baseline.** Each experiment starts from the same snapshot. It never changes the weights being trained.
2. **Fold BatchNorm into the previous convolution for inference.** If BN normally computes `gamma * (conv(x) - mean) / sqrt(var + eps) + beta`, we put this multiplication and addition into the convolution's weights and bias. Our own implementation handles adjacent Conv/BN pairs inside MobileNet's sequential blocks. Tests check that predictions remain close before quantization.
3. **Quantize convolution and classifier weights per output channel.** Each channel gets its own scale: `max(abs(weight)) / (2**(bits-1)-1)`. We round `weight / scale` and clamp it to the signed range. A zero channel uses scale 1 and stays zero. One negative code is unused so the range is symmetric.
4. **Observe activation ranges on training images.** The images use the saved normalization, with no random augmentation. We record the minimum and maximum at each Conv/Linear input after weight quantization. Each weight setting gets its own calibration.
5. **Quantize those activations with a scale and zero point.** Include zero in the observed range, divide the range into `2**bits-1` steps, then round and clamp. Values outside the calibration range saturate at an endpoint.
6. **Evaluate on validation images.** The preview uses 20 images per class; a full sweep uses all 5,000. The test dataset is never loaded by this runner.

The scale sets the spacing between representable values. Fewer bits reduce storage but increase rounding error.

## Settings

The default sweep is **W8A8, W4A8 and W4A4**. W means weight bits; A means activation bits. The functions also support 2 bits, and 32 means bypass quantization.

By default, the **first convolution and final classifier weights and inputs stay in FP32**. They can be sensitive to reduced precision. All other Conv/Linear weights and inputs use the requested bits. Biases stay FP32; any unfused BatchNorm parameters and buffers retain their original types. Residual additions and other arithmetic still execute in floating point.

CLI switches:

- `--settings 8/8 4/8 4/4`: choose the experiments.
- `--quantize-edges`: also quantize the first convolution and final classifier.
- `--no-fold-bn`: leave BatchNorm separate.
- `--calibration-per-class 100`: use 1,000 training images to set ranges.
- `--validation-per-class 500`: evaluate all 5,000 validation images.

## What the compression numbers mean

**Weight/model storage is measured from a real file.** Our `.mq` file stores a short JSON header, packed integer weight codes, FP32 per-channel scales, and unchanged tensors. At 4 bits, two weight codes share one byte; at 2 bits, four share it. Padding at tensor ends is counted. The header includes shapes, layer names, activation settings, architecture, normalization and checkpoint identity.

`weight_compression_ratio = original parameter-and-buffer bytes / complete .mq file bytes`

This deliberately includes the entire compressed file, including scales, biases, buffers, activation calibration metadata and header overhead. The uncompressed reference counts the original state tensors, not PyTorch checkpoint-container overhead. Model sizes use decimal MB: 1 MB = 1,000,000 bytes. BatchNorm folding is part of the reported method and can change the number of stored tensors.

**Activation storage is an estimate.** We count each Conv/Linear input occurrence for one 32x32 image. For the same locations in both models, compare FP32 bytes against the estimated packed bytes. The estimate includes unchanged FP32 input locations and one FP32 scale plus one int32 zero point (8 bytes) per quantized location.

`activation_compression_ratio = summed FP32 input bytes / (summed packed input bytes + scale/zero-point bytes)`

This counts tensor occurrences, so a tensor consumed at multiple locations can be counted more than once. Other intermediate tensors and allocator/workspace costs are excluded. It is **not peak GPU memory**. Static scales are conservatively charged once per location in this one-image estimate; they are also included in the model file's metadata.

**The inference simulation still uses FP32 tensors.** Packed weights are decoded when loaded, and activations are rounded then converted back to approximate floating-point values. This measures quantization's effect on predictions, but does not provide integer kernels, reduced live activation memory, or a promised speedup. The actual saving demonstrated here is the stored model file.

## Commands

From this project directory, use the existing environment:

```powershell
# Quick checks, CPU only.
.\.venv\Scripts\python.exe test_manual_compression.py

# Small interim check while the GPU trains: 100 calibration + 200 validation images.
.\.venv\Scripts\python.exe run_compression.py --preview --device cpu

# After the 300-epoch baseline finishes: 1,000 calibration + 5,000 validation images.
.\.venv\Scripts\python.exe run_compression.py --device cuda --batch-size 128
```

Each invocation creates a new folder under `artifacts/compression/`. It saves the exact baseline snapshot, subset indices, packed models, `results.json` and `comparison.csv`. Every exported model must reproduce the pre-export model's predictions exactly on a validation batch before it is evaluated. The runner refuses GPU evaluation while the local baseline process is running.

## Completed results

- **Q3 full sweep completed on 8 September 2026:** nine combinations of 2/4/8-bit weights and activations, plus FP32, using the epoch-284 baseline, 1,000 training calibration images, and all 5,000 validation images. Read [Q3 results and analysis](artifacts/compression/q3_final_grid/Q3_RESULTS.md). W8A8 retained 93.88% validation accuracy; W4A8 reached 90.96%; W4A4 reached 86.86%. All configurations using 2 bits reached approximately chance accuracy. All nine export/reload checks passed.
- **Q3 W&B publication completed and verified.** All ten runs are uploaded, their configurations and numeric metrics match the local results, and the native six-axis Parallel Coordinates panel was saved and reloaded successfully. Open the [Q3 W&B report](https://wandb.ai/harshith7946-indian-institute-of-technology-madras/cs6886-assignment2-q3/reports/Q3---MobileNet-v2-compression-results--VmlldzoxNzg4ODQ3OQ==). The written Q3 section includes this link. To reproduce publication from saved measurements, use `.\.venv\Scripts\python.exe report_q3.py --wandb-mode online`; no model experiments need rerunning. The publisher normalizes report URLs on Windows and resolves the API endpoint before connecting.
- **Q4 complete:** W4A8 was chosen using validation accuracy, then scored 90.13% on all 10,000 test images. It occupies 1.305497 MB, with 7.12x weights-only compression, 6.96x complete model compression and 3.90x estimated activation compression. See `artifacts/final/results.json`.
- The complete Q1-Q5 answers are in `output/pdf/Assignment_2_Report.pdf`, with editable text in `output/Assignment_2_Answers.md`. `README.md` gives the exact setup and reproduction commands.

## References consulted

- [Jacob et al., Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877): reference for scale/zero-point quantization. Our implementation is a post-training FP32 simulation, not the paper's full integer inference system.
- [PyTorch Module documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html): input hooks and module traversal.

This is a simple min/max calibration method. It does not search clipping thresholds or train with quantization enabled. Lower-bit MobileNet configurations can therefore lose substantial accuracy; report the measured results honestly.
