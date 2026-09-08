# Assignment 2: what each question asks

The task is to train MobileNet-v2 on CIFAR-10, compress the trained model using manually written code, and measure the change in accuracy and storage. Submit one PDF with answers, figures, tables and the GitHub link.

## Q1: baseline training

Use a separate training, validation and test split. Apply augmentation only during training, and calculate normalization from training images. State the model settings, optimizer, learning-rate schedule and regularization. Report the best checkpoint's test accuracy and show the learning curves.

The completed baseline has **92.64% test accuracy**. Its best checkpoint was selected at epoch 284 using validation accuracy. Training finished all 300 epochs after resuming the interrupted run. A 99% training score alone does not prove overfitting; compare the held-out results and curves.

## Q2: compression implementation

Weights are learned parameters. Activations are the intermediate values produced while processing an image. Both are quantized here, using fewer numerical levels. The compression code handles scales, rounding, clipping and packing itself. The first convolution and final classifier stay FP32 in the reported experiments.

A compression ratio is the original byte count divided by the compressed byte count. Include scales and metadata rather than just dividing 32 by the chosen bit-width. A smaller saved model and lower peak runtime memory are different measurements.

## Q3: compare settings

The complete grid uses 2, 4 and 8 bits independently for weights and activations. Each setting uses the same trained checkpoint and validation split. Calibration ranges come from training images. The test set is not used to compare these settings.

The full comparison and native W&B chart are linked in `artifacts/compression/q3_final_grid/Q3_RESULTS.md`. W8A8 retained 93.88% validation accuracy. W4A8 reached 90.96% with a smaller model. Settings involving 2 bits were near chance accuracy.

## Q4: report one final choice

Use one configuration and give its weight ratio, activation ratio and measurement method, test accuracy, and final model size. The chosen W4A8 model is the smallest saved model that met the 90% validation threshold. It scored **90.13% test accuracy** and occupies **1.305497 MB**.

Its weight ratio is **7.12x**, including scales and all header overhead. Its complete model ratio is **6.96x**. The activation ratio is an estimated **3.90x**, based on Conv/Linear input occurrences; it is not measured peak GPU memory. Inference currently runs in FP32 with quantize/dequantize steps.

## Q5: reproducibility

Keep data preparation, training, compression and evaluation in separate files. The README must give exact commands, dependencies, environment and seeds. Include the GitHub repository link in the PDF and ensure the evaluator can access it.

The baseline does not need to be trained again for Q3 or Q4. To reproduce everything or evaluate the supplied compressed model, follow `README.md`. The final answers are in `output/pdf/Assignment_2_Report.pdf`.
