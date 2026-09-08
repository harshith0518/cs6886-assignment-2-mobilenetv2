"""Build the assignment PDF and Markdown answers from the saved results."""
import csv
import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

ROOT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def build():
    info = read_json("submission_info.json")
    base = read_json("artifacts/training/mobilenetv2_300/results.json")
    sweep = read_json("artifacts/compression/q3_final_grid/results.json")
    final = read_json("artifacts/final/results.json")
    manifest = read_json("artifacts/compression/q3_final_grid/wandb_online_manifest.json")
    with (ROOT / "artifacts/training/mobilenetv2_300/history.csv").open() as stream:
        history = list(csv.DictReader(stream))
    output = ROOT / "output"
    pdf_dir = output / "pdf"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BodyTextSmall", fontName="Helvetica", fontSize=10.2, leading=13.5,
                              spaceAfter=7, textColor=colors.HexColor("#26313d")))
    styles.add(ParagraphStyle(name="Sub", fontName="Helvetica-Bold", fontSize=11.3, leading=15,
                              spaceBefore=8, spaceAfter=6, textColor=colors.HexColor("#183650")))
    styles.add(ParagraphStyle(name="CaptionSmall", fontName="Helvetica", fontSize=8.4, leading=11,
                              spaceAfter=8, textColor=colors.HexColor("#52606e")))
    styles.add(ParagraphStyle(name="TableCell", fontName="Helvetica", fontSize=9, leading=12))
    styles.add(ParagraphStyle(name="CodeSmall", fontName="Courier", fontSize=8.4, leading=12,
                              backColor=colors.HexColor("#f0f3f6"), borderPadding=6, spaceAfter=9))
    styles["Title"].fontSize = 21
    styles["Title"].leading = 25
    styles["Title"].textColor = colors.HexColor("#183650")
    styles["Heading1"].fontSize = 17
    styles["Heading1"].leading = 21
    styles["Heading1"].textColor = colors.HexColor("#183650")
    story, markdown = [], []
    width = A4[0] - 88

    def paragraph(text, style="BodyTextSmall"):
        story.append(Paragraph(escape(text), styles[style]))
        markdown.extend([text, ""])

    def heading(text, top=False):
        story.append(Paragraph(escape(text), styles["Heading1" if top else "Sub"]))
        markdown.extend([("## " if top else "### ") + text, ""])

    def table(headers, rows, widths=None):
        cells = [[Paragraph(escape(str(x)), styles["TableCell"]) for x in row]
                 for row in [headers] + rows]
        t = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e4edf3")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f8fa")]),
            ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor("#aec2d0"))]))
        story.extend([t, Spacer(1, 9)])
        markdown.append("| " + " | ".join(headers) + " |")
        markdown.append("| " + " | ".join("---" for _ in headers) + " |")
        markdown.extend("| " + " | ".join(str(x) for x in row) + " |" for row in rows)
        markdown.append("")

    def picture(path, caption, max_height=250):
        im = Image(str(ROOT / path))
        factor = min(width / im.imageWidth, max_height / im.imageHeight)
        im.drawWidth, im.drawHeight = im.imageWidth * factor, im.imageHeight * factor
        story.append(im)
        markdown.extend([f"![{caption}](../{path})", ""])
        paragraph(caption, "CaptionSmall")

    def link(label, url):
        story.append(Paragraph(f'<link href="{escape(url, {chr(34): "&quot;"})}" color="#17628a"><u>{escape(label)}</u></link>', styles["BodyTextSmall"]))
        markdown.extend([f"[{label}]({url})", ""])

    def code(lines):
        story.append(Paragraph("<br/>".join(escape(line).replace(" ", "&#160;") for line in lines), styles["CodeSmall"]))
        markdown.extend(["```powershell", *lines, "```", ""])

    def new_page():
        story.append(PageBreak())

    story.append(Paragraph("MobileNet-v2 on CIFAR-10", styles["Title"]))
    author = info["author"] + (" | " + info["roll_number"] if info.get("roll_number") else "")
    paragraph("CS6886 - System Engineering for Deep Learning | Assignment 2", "CaptionSmall")
    paragraph(author + " | IIT Madras | 8 September 2026", "CaptionSmall")
    markdown[0:0] = ["# MobileNet-v2 on CIFAR-10", ""]
    heading("Q1. Training baseline", True)
    heading("(a) Data preparation")
    paragraph("CIFAR-10 was split into 45,000 training, 5,000 validation and 10,000 test images. A seeded permutation keeps 4,500 training and 500 validation images per class. RGB mean (0.491840, 0.482541, 0.446820) and standard deviation (0.247051, 0.243542, 0.261692) were computed from the training split only.")
    paragraph("Training transforms: reflect padding of 4 pixels, random 32 x 32 crop, horizontal flip (p = 0.5), tensor conversion, normalization and random erasing (p = 0.25, area 0.02-0.15, aspect ratio 0.5-2, fill 0). Validation and test use only tensor conversion and normalization.")
    heading("(b) Model and training strategy")
    table(["Setting", "Value"], [
        ["Model", "torchvision MobileNet-v2, trained from scratch; width 1.0; dropout 0.2; 10 output classes"],
        ["CIFAR-10 change", "First convolution stride 1; remaining stage strides unchanged; 2,236,682 parameters"],
        ["BatchNorm", "epsilon = 1e-5; momentum = 0.1"],
        ["Training", "300 epochs; batch 128; SGD, momentum 0.9, Nesterov"],
        ["Learning rate", "0.1; 5-epoch linear warmup, then cosine decay to 1e-5"],
        ["Regularization", "Weight decay 5e-4, excluding BN/bias; label smoothing 0.05; gradient clipping at norm 5"],
        ["Execution", "Seed 42; AMP during training; FP32 evaluation; deterministic cuDNN"]], [103, width - 103])
    heading("(c) Test accuracy, curves and failure modes")
    paragraph(f"The checkpoint with the highest validation accuracy was epoch {base['best_epoch']}: {base['best_validation_accuracy']:.2f}% validation and {base['test_accuracy']:.2f}% test top-1 accuracy (9,264 / 10,000). Test cross-entropy was {base['test_loss']:.4f}. The interrupted run was resumed with its optimizer and random states.")
    picture("artifacts/training/mobilenetv2_300/learning_curves.png",
            "Figure 1. Training and validation curves over 300 epochs. Both plotted losses use label smoothing 0.05; the final test loss is unsmoothed.", 175)
    paragraph(f"Final-epoch training accuracy was {float(history[-1]['train_accuracy']):.2f}% and validation accuracy was {float(history[-1]['val_accuracy']):.2f}%. This gap suggests some overfitting, although training uses augmentation. A 99% training score alone is insufficient to judge it; the held-out results and curves matter. Only one seed was trained, so run-to-run variation was not measured.")

    new_page()
    heading("Q2. Model compression implementation", True)
    heading("(a) Configurable method and design")
    paragraph("The implementation is in manual_compression.py. Weight and activation precision can each be set to 2, 4, 8 or 32 bits. It uses tensor operations, explicit rounding/clipping and bit shifts; no ready-made compression or quantization API is called. All compressed settings reuse the trained baseline without fine-tuning.")
    paragraph("Weights use a separate symmetric scale for every output channel. This lets channels with different ranges, including depthwise filters, use their available precision. For b-bit weights and output channel c:")
    code(["L = 2**(b - 1) - 1", "s[c] = max(abs(W[c])) / L",
          "q[c] = clip(round(W[c] / s[c]), -L, L)", "W_hat[c] = s[c] * q[c]"])
    paragraph("An all-zero channel uses scale 1. Signed codes are shifted by L before packing. Two 4-bit codes or four 2-bit codes share one byte. This symmetric scheme has three signed levels at 2 bits; it does not use all four possible bit patterns.")
    paragraph("Activation ranges are collected at Conv/Linear inputs using 1,000 training images without augmentation, after weight quantization. Each location uses one affine scale and zero point. The observed range is extended to include zero:")
    code(["Q = 2**b - 1", "s = (high - low) / Q",
          "z = clip(round(-low / s), 0, Q)", "q = clip(round(x / s + z), 0, Q)", "x_hat = (q - z) * s"])
    paragraph("A zero-width range uses scale 1. Min/max calibration is simple to reproduce, but outliers can widen the range and reduce precision. Very low bit-widths can then accumulate rounding and clipping errors.")
    heading("(b) Layers and exceptions")
    paragraph("Adjacent Conv/BatchNorm pairs are folded in evaluation mode: multiply each filter by gamma / sqrt(running_variance + epsilon), and adjust the bias using the running mean and beta. The BatchNorm module is then replaced by an identity. All intermediate Conv/Linear weights are quantized, including pointwise and depthwise convolutions.")
    paragraph("The first convolution (features.0.0) and final classifier (classifier.1) weights and inputs stay FP32. Biases stay FP32. Activation hooks quantize the other Conv/Linear inputs; residual additions and other arithmetic run in FP32. Flags allow quantizing the edge layers or disabling BN folding. These exceptions stay fixed throughout Q3.")
    heading("(c) Storage overhead")
    table(["Part of final W4A8 file", "Bytes"], [
        ["Packed weight codes", f"{final['packed_code_bytes']:,}"],
        ["FP32 per-channel weight scales", f"{final['weight_scale_bytes']:,}"],
        ["Raw FP32 edge weights and biases", f"{final['other_raw_tensor_bytes']:,}"],
        ["Header, including activation ranges, scales and zero points", f"{final['header_bytes']:,}"],
        ["Total actual file size", f"{final['model_file_bytes']:,}"]], [width - 90, 90])
    paragraph("The .mq file contains a 4-byte signature, 8-byte header length, JSON header and tensor payload. All are counted above. Activation storage estimates additionally use a compact 4-byte scale and 4-byte zero point per quantized location. The actual JSON representation is already included in the model file size; the two ratios describe different storage quantities.")

    new_page()
    heading("Q3. Compression results", True)
    heading("(a) Compression levels")
    paragraph(f"The 3 x 3 grid varies weight and activation precision over 8, 4 and 2 bits. All nine settings start from the epoch-{sweep['baseline_epoch']} checkpoint. Calibration uses 1,000 training images, 100 per class (seed 42). Evaluation uses the same full 5,000-image validation split, 500 per class (order seed 43). Calibration and validation indices are saved and disjoint. No test images are used here.")
    heading("(b) Accuracy comparison")
    rows = [[r["setting"], f"{r['validation_accuracy']:.2f}", f"{r['accuracy_drop_pp']:.2f}",
             f"{r['model_size_mb']:.3f}", f"{r['weight_compression_ratio']:.2f}x",
             f"{r['activation_compression_ratio']:.2f}x"] for r in sweep["rows"]]
    table(["Setting", "Val. acc. (%)", "Drop (pp)", "Size (MB)", "Model ratio", "Activation ratio"],
          rows, [64, 83, 70, 82, 99, width - 398])
    paragraph("Model ratio uses 9,083,592 bytes of original parameters and buffers divided by the complete packed file size. It includes BN folding and every storage overhead; it is the metric named weight_compression_ratio in the Q3 JSON/W&B runs. The separate weights-only calculation is given in Q4. Activation ratios are estimates of packed Conv/Linear input occurrences, not peak GPU memory.", "CaptionSmall")
    picture("artifacts/compression/q3_final_grid/q3_accuracy_compression.png",
            "Figure 2. Validation accuracy across the grid and its relation to saved model size.", 210)
    paragraph("W8A8 keeps the measured 93.88% validation accuracy at 2.400 MB. W4A8 gives 90.96% at 1.305 MB, while W8A4 gives 90.16% with a larger reduction in activation storage. W4A4 drops to 86.86%. Every setting using 2-bit weights or activations is close to 10% chance accuracy, so its larger storage reduction is not useful here.")
    paragraph("Each packed file was reloaded before evaluation. Reloaded logits matched the corresponding pre-export model exactly on a validation batch. This checks serialization; it does not mean a compressed model produces the same predictions as FP32.")

    new_page()
    heading("Q3(b). W&B Parallel Coordinates chart", True)
    paragraph("Ten finished runs, one per setting including FP32, are logged to W&B. The report contains a native Parallel Coordinates panel with six axes: weight bits, activation bits, model storage ratio, activation storage ratio, model size in MB and validation accuracy.")
    link("Open the native W&B Parallel Coordinates chart", manifest["report_url"])
    paragraph("Project: " + manifest["entity"] + "/" + manifest["project"], "CaptionSmall")
    picture("artifacts/compression/q3_final_grid/q3_parallel_coordinates_local.png",
            "Figure 3. Local companion plot of the same ten runs. This figure was drawn with Matplotlib; the native W&B panel is available through the link above.", 340)
    paragraph("The uploaded run configurations and numerical results were checked against the local results. The saved W&B report was also reloaded through the Reports API to verify that the native six-axis panel exists. The report URL and run links are recorded in wandb_online_manifest.json.")
    paragraph("Higher compression does not always preserve useful accuracy. The 2-bit settings reduce storage most but perform near chance. Among settings with at least 90% validation accuracy, W4A8 has the smallest stored model. This validation-based rule is used for the one final configuration in Q4.")
    paragraph("These results use one baseline and one calibration subset. No repeated-seed study, integer-kernel timing or runtime-memory benchmark was done. Stored weights are packed, but inference decodes them to FP32 and simulates activation quantization in floating point.")

    new_page()
    heading("Q4. Compression analysis", True)
    paragraph("Chosen configuration: W4A8. It is the smallest saved model among the settings with at least 90% validation accuracy. The choice and file hash were saved in selection.json before the test set was evaluated. The following measurements all refer to this one configuration.")
    heading("(a) Weight compression ratio")
    paragraph("Original Conv/Linear weights occupy 8,810,240 bytes. After compression, their packed codes, scales and FP32 edge weights occupy 1,217,200 bytes. Conservatively charging the entire 20,033-byte file header to weights gives:")
    code(["Weight ratio = 8,810,240 / (1,217,200 + 20,033)", f"             = {final['weight_compression_ratio']:.2f}x"])
    paragraph("This ratio includes weight scales and metadata overhead. BN parameters and biases are excluded from the weights-only numerator and tensor payload. Their contribution is included in the complete model measurement below.")
    heading("(b) Activation compression ratio and measurement")
    paragraph("For one 32 x 32 image, input shapes were recorded at every Conv/Linear location. Summing all occurrences gives 2,177,536 FP32 bytes. Using 8 bits at the compressed locations and 32 bits at the two protected inputs gives 557,440 packed bytes. There are 51 quantized locations, adding 51 x 8 = 408 bytes for scales and zero points.")
    code(["Activation ratio = 2,177,536 / (557,440 + 408)", f"                 = {final['activation_compression_ratio']:.2f}x"])
    paragraph("An input used by multiple layers is counted at each occurrence. This measures estimated aggregate activation storage, not simultaneously live tensors or peak memory. The current inference code keeps arithmetic in FP32, so it does not realize this activation-memory saving at runtime.")
    heading("(c) Quantized model accuracy")
    table(["Measurement", "Value"], [
        ["Validation top-1 accuracy used for selection", f"{final['validation_accuracy']:.2f}%"],
        ["Final test top-1 accuracy", f"{final['test_accuracy']:.2f}% ({final['test_correct']:,} / {final['test_images']:,})"],
        ["Baseline test top-1 accuracy", f"{base['test_accuracy']:.2f}%"],
        ["Test accuracy decrease", f"{final['test_accuracy_drop_pp']:.2f} percentage points"],
        ["Quantized test cross-entropy", f"{final['test_loss']:.4f}"]], [width - 170, 170])
    paragraph("The compressed model's lowest class accuracy is cat (72.3%); automobile is highest (95.3%). The saved predictions and confusion matrix allow checking these class results. No further configuration was selected using these test outcomes.")
    heading("(d) Final model size")
    paragraph(f"The actual packed file is {final['model_file_bytes']:,} bytes, or {final['model_size_mb']:.6f} MB (about 1.31 MB; decimal MB). Compared with 9,083,592 original state-tensor bytes, the complete model storage ratio is {final['model_compression_ratio']:.2f}x. This includes the overhead table in Q2(c), and is separate from the weights-only ratio in Q4(a).")

    new_page()
    heading("Q5. Reproducibility and repository", True)
    heading("(a) Code organization")
    table(["File", "Purpose"], [
        ["prepare_data.py", "Download, class-balanced split and training-only normalization"],
        ["train_baseline.py", "Training, validation selection, checkpointing and resume"],
        ["manual_compression.py", "Quantization, BN folding and packed model save/load"],
        ["run_compression.py", "Training-only calibration and Q3 validation sweep"],
        ["evaluate_final.py", "Record the final selection and evaluate test accuracy"],
        ["report_q3.py / build_report.py", "W&B panel, figures, PDF and editable answers"]], [173, width - 173])
    paragraph("Comments explain the scale calculations, packing, layer exceptions and data separation. The repository includes the best baseline checkpoint, packed models, saved splits, metrics and training curves. Eight regression tests passed, including bit packing, compression reload and checkpoint write handling. The standalone preparation script reproduced every saved split index and the original normalization values exactly.")
    heading("(b) Commands, environment and seeds")
    paragraph("The run used Windows, Python 3.14.3, torch 2.14.0+cu126, torchvision 0.29.0+cu126, NumPy 2.5.2 and an RTX 3050 Laptop GPU (4 GB). W&B uses wandb 0.29.0 and wandb-workspaces 0.4.11. The README gives setup, full training/resume, the nine-setting sweep, final evaluation and chart publication commands. Requirement files pin the direct dependencies; environment-lock.txt records the full environment.")
    paragraph("From the repository root, after creating and activating the virtual environment:")
    code(["python -m pip install -r requirements-notebook.txt",
          "python -m pip install -r requirements-q3.txt",
          "python -m pip install -r requirements-report.txt",
          "python prepare_data.py --download",
          "python -m unittest test_manual_compression test_training_io",
          "python report_q3.py --wandb-mode none",
          "python evaluate_final.py --device cuda `",
          "    --output-dir artifacts/final_recheck"])
    paragraph("The last command reloads and tests the chosen model in a fresh output folder. Training and data splitting use seed 42. Calibration uses seed 42 and validation ordering uses seed 43. cuDNN deterministic mode is enabled; benchmark mode is disabled. Exact results may still vary across hardware and versions.")
    heading("(c) GitHub repository")
    link(info["repository_url"], info["repository_url"])
    paragraph("The PDF is the Moodle submission file. Repository and W&B access must be available to the evaluator if either is private.", "CaptionSmall")
    heading("References")
    link("torchvision: MobileNet-v2 source", "https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py")
    link("CIFAR-10 dataset", "https://www.cs.toronto.edu/~kriz/cifar.html")
    link("Jacob et al.: Quantization and Training of Neural Networks (2018)", "https://arxiv.org/abs/1712.05877")
    link("W&B: Parallel Coordinates panels", "https://docs.wandb.ai/models/app/features/panels/parallel-coordinates")

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#d7e0e7"))
        canvas.line(44, 35, A4[0] - 44, 35)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#607080"))
        canvas.drawString(44, 22, "CS6886 | Assignment 2 | MobileNet-v2 compression")
        canvas.drawRightString(A4[0] - 44, 22, str(doc.page))
        canvas.restoreState()

    destination = pdf_dir / "Assignment_2_Report.pdf"
    doc = SimpleDocTemplate(str(destination), pagesize=A4, rightMargin=44, leftMargin=44,
                            topMargin=34, bottomMargin=48, title="CS6886 Assignment 2 - MobileNet-v2 Compression",
                            author=info["author"])
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    (output / "Assignment_2_Answers.md").write_text("\n".join(markdown), encoding="utf-8")
    print(f"Saved {destination}")


if __name__ == "__main__":
    build()
