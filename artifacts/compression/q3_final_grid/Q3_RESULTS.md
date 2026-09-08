# Q3 results

Checkpoint epoch 284; 1,000 training calibration images; 5,000 validation images; no test images or fine-tuning.

| Setting | Val. accuracy (%) | Drop (pp) | Model MB | Model ratio | Activation ratio |
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

Model ratios include the complete packed file. Activation ratios estimate packed Conv/Linear input occurrences plus scale/zero-point bytes; they are not peak GPU memory. Inference arithmetic remains FP32.

[Native W&B Parallel Coordinates chart](https://wandb.ai/harshith7946-indian-institute-of-technology-madras/cs6886-assignment2-q3/reports/Q3---MobileNet-v2-compression-results--VmlldzoxNzg4ODQ3OQ==)

The compact repository retains the baseline and chosen W4A8 model. Intermediate model files can be regenerated with the full-sweep command in README.md.

This check verified the saved numerical results and 1 available packed file(s). Use --check-files after regenerating the sweep to require all nine files.
