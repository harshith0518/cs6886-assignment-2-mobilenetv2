"""Run validation sweeps and final test evaluation without retraining."""
import argparse
import csv
import hashlib
import io
import json
import math
import struct
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.train import ROOT, RESULTS, DatasetView, build_model, evaluate, training_is_running, baseline_files, save_json
from src.compression import (
    activation_plan, activation_storage, add_activation_quantization,
    calibrate_activations, compress_weights, save_compressed, load_compressed, compression_layers,
)

DEFAULT_SETTINGS = [(weight, activation) for weight in (8, 4, 2) for activation in (8, 4, 2)]


def balanced_subset(indices, labels, per_class, seed):
    # Keep the same number of images per class, even for a small preview.
    rng = np.random.default_rng(seed)
    selected = []
    for label in range(10):
        choices = indices[labels[indices] == label]
        if not 0 < per_class <= len(choices):
            raise ValueError(f"Requested {per_class} images per class, but class {label} has {len(choices)}.")
        selected.extend(rng.permutation(choices)[:per_class])
    return rng.permutation(np.array(selected, dtype=np.int64))


def run_experiments(run_dir=None, output_dir=None, preview=False, device="cpu",
                    settings=None, calibration_per_class=None, validation_per_class=None,
                    batch_size=32, cpu_threads=2, keep_first_last=True, fold_bn=True):
    checkpoint_path, baseline_results_path, splits_path = baseline_files(run_dir)
    finished = baseline_results_path.exists()
    if not finished and not preview:
        raise RuntimeError("Baseline isn't finished yet. Use preview=True (CLI: --preview) for a small interim check.")
    if run_dir is not None and training_is_running(run_dir) and device.startswith("cuda"):
        raise RuntimeError("The GPU is training the baseline. Use CPU for this preview.")
    if batch_size < 1 or cpu_threads < 1:
        raise ValueError("Batch size and CPU thread count must be positive.")
    settings = list(DEFAULT_SETTINGS if settings is None else settings)
    if not settings or len(settings) != len(set(settings)):
        raise ValueError("Give at least one setting, without duplicates.")
    calibration_per_class = calibration_per_class if calibration_per_class is not None else (10 if preview else 100)
    validation_per_class = validation_per_class if validation_per_class is not None else (20 if preview else 500)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_dir) if output_dir else ROOT / "runs" / ("preview_" + stamp if preview else "validation_" + stamp)
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(cpu_threads)
    torch.manual_seed(42)

    # Read one whole best.pt version. Training can safely keep replacing its own file.
    snapshot = checkpoint_path.read_bytes()
    checkpoint_hash = hashlib.sha256(snapshot).hexdigest()
    checkpoint = torch.load(io.BytesIO(snapshot), map_location="cpu", weights_only=False)
    # Keep the source checkpoint once; its checksum identifies this sweep.
    preparation = checkpoint["config"]["preparation"]
    baseline = build_model(preparation["model"]).eval()
    baseline.load_state_dict(checkpoint["model"])
    original_bytes = sum(t.numel() * t.element_size() for t in baseline.state_dict().values())

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
        preparation["normalization"]["mean"], preparation["normalization"]["std"])])
    raw = datasets.CIFAR10(str(ROOT / "data"), train=True, download=False)
    labels = np.array(raw.targets)
    with np.load(splits_path) as splits:
        train_indices = splits["train"].copy()
        validation_indices = splits["validation"].copy()
    cal_indices = balanced_subset(train_indices, labels, calibration_per_class, 42)
    val_indices = balanced_subset(validation_indices, labels, validation_per_class, 43)
    if np.intersect1d(cal_indices, val_indices).size:
        raise ValueError("Calibration and validation must be disjoint.")
    np.savez(output_dir / "compression_indices.npz", calibration=cal_indices, validation=val_indices)
    cal_loader = DataLoader(DatasetView(raw, cal_indices, transform), batch_size=batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(DatasetView(raw, val_indices, transform), batch_size=batch_size, shuffle=False, num_workers=0)
    stage = "intermediate checkpoint" if not finished else "completed baseline"
    print(f"{'PREVIEW' if preview else 'VALIDATION SWEEP'} | {stage}, best epoch {checkpoint['epoch']} | {device}", flush=True)
    print(f"Calibration: {len(cal_indices)} training images | validation: {len(val_indices)} images | test set untouched", flush=True)

    baseline.to(device)
    start = time.perf_counter()
    baseline_loss, baseline_accuracy = evaluate(baseline, val_loader, nn.CrossEntropyLoss(), device)
    baseline_seconds = time.perf_counter() - start
    baseline.cpu()
    rows = [{"setting": "FP32", "weight_bits": 32, "activation_bits": 32,
             "validation_accuracy": baseline_accuracy, "validation_loss": baseline_loss,
             "accuracy_drop_pp": 0.0, "model_size_mb": original_bytes / 1e6,
             "weight_compression_ratio": 1.0, "activation_compression_ratio": 1.0,
             "evaluation_seconds": baseline_seconds, "size_basis": "original state tensors"}]
    print(f"FP32: {baseline_accuracy:.2f}% validation accuracy", flush=True)

    for weight_bits, act_bits in settings:
        name = f"W{weight_bits}A{act_bits}"
        started = time.perf_counter()
        model, weight_records = compress_weights(baseline, weight_bits, keep_first_last, fold_bn)
        model.to(device)
        ranges, seen = calibrate_activations(model, cal_loader)
        plan = activation_plan(model, ranges, act_bits, keep_first_last)
        add_activation_quantization(model, plan)
        activation_sizes = activation_storage(plan)
        metadata = {"model_config": preparation["model"], "fold_bn": fold_bn,
                    "keep_first_last": keep_first_last, "weight_bits": weight_bits,
                    "activation_bits": act_bits, "activations": plan,
                    "original_state_bytes": original_bytes,
                    "normalization": preparation["normalization"], "classes": preparation["classes"],
                    "calibration_images": seen, "calibration_source": "training split, no augmentation",
                    "baseline_epoch": checkpoint["epoch"], "baseline_sha256": checkpoint_hash,
                    "preview": preview, "baseline_stage": stage,
                    "inference": "packed weights decoded to FP32; activation quantization simulated in FP32"}
        packed_path = output_dir / (name + ".mq")
        sizes = save_compressed(packed_path, model, weight_records, metadata)

        # Check the exported file really restores the same predictions before using it.
        restored, _ = load_compressed(packed_path, build_model)
        restored.to(device)
        example_images, _ = next(iter(val_loader))
        with torch.inference_mode():
            expected = model(example_images.to(device))
            actual = restored(example_images.to(device))
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        del model, weight_records
        eval_start = time.perf_counter()
        loss, accuracy = evaluate(restored, val_loader, nn.CrossEntropyLoss(), device)
        eval_seconds = time.perf_counter() - eval_start
        del restored
        row = {"setting": name, "weight_bits": weight_bits, "activation_bits": act_bits,
               "validation_accuracy": accuracy, "validation_loss": loss,
               "accuracy_drop_pp": baseline_accuracy - accuracy,
               **sizes, **activation_sizes, "evaluation_seconds": eval_seconds,
               "total_setting_seconds": time.perf_counter() - started,
               "size_basis": "actual packed file including all metadata", "export_roundtrip_passed": True}
        rows.append(row)
        print(f"{name}: {accuracy:.2f}% | file {sizes['model_size_mb']:.3f} MB | storage {sizes['weight_compression_ratio']:.2f}x | activations {activation_sizes['activation_compression_ratio']:.2f}x", flush=True)

    result = {"preview": preview, "baseline_stage": stage, "baseline_epoch": checkpoint["epoch"],
              "baseline_sha256": checkpoint_hash, "calibration_images": len(cal_indices),
              "validation_images": len(val_indices), "test_images_used": 0,
              "keep_first_last_fp32": keep_first_last, "fold_bn": fold_bn,
              "seed": 42, "validation_seed": 43, "device": device,
              "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
              "weight_measurement": "original parameters and buffers / complete packed model file bytes",
              "activation_measurement": "estimated packed Conv/Linear inputs per image, including scale/zero-point bytes; not runtime memory",
              "rows": rows}
    (output_dir / "compression.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {output_dir}", flush=True)
    return output_dir, result


def evaluate_final(args):
    source = Path(args.sweep_dir).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    results_path = output / "final.json"
    sweep = json.loads((source / "compression.json").read_text())
    checkpoint_path, baseline_results_path, _ = baseline_files(args.run_dir)
    baseline_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if baseline_hash != sweep["baseline_sha256"]:
        raise ValueError("The baseline folder does not match the compression sweep.")
    choices = [row for row in sweep["rows"] if row["setting"] != "FP32"
               and row["validation_accuracy"] >= args.min_validation_accuracy]
    if not choices:
        raise ValueError("No compressed setting meets the validation accuracy limit.")
    chosen = min(choices, key=lambda row: (row["model_size_mb"], -row["validation_accuracy"]))
    model_file = source / (chosen["setting"] + ".mq")
    # The compact repository keeps only the chosen model from the original sweep.
    if source == RESULTS and not model_file.exists():
        model_file = ROOT / "models" / model_file.name
    packed = model_file.read_bytes()
    model_hash = hashlib.sha256(packed).hexdigest()
    if results_path.exists():
        previous = json.loads(results_path.read_text())
        if previous["model_sha256"] != model_hash:
            raise ValueError("This output folder already contains results for a different model.")
        print(json.dumps(previous, indent=2))
        return

    selection = {"setting": chosen["setting"], "min_validation_accuracy": args.min_validation_accuracy,
                 "rule": "smallest stored model among settings meeting the validation accuracy limit",
                 "validation_accuracy": chosen["validation_accuracy"], "model_sha256": model_hash,
                 "baseline_sha256": sweep["baseline_sha256"], "selected_before_test": True,
                 "selected_at": datetime.now().isoformat(timespec="seconds")}
    selection_path = output / "selection.json"
    if selection_path.exists():
        old = json.loads(selection_path.read_text())
        if old["model_sha256"] != model_hash:
            raise ValueError("The recorded selection cannot be changed in this output folder.")
    else:
        save_json(selection_path, selection)
    model_path = output / (chosen["setting"] + ".mq")
    model_path.write_bytes(packed)
    print(f"Selected {chosen['setting']} using validation only: {chosen['validation_accuracy']:.2f}%", flush=True)

    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model, metadata = load_compressed(model_path, build_model)
    assert metadata["baseline_sha256"] == sweep["baseline_sha256"]
    model.to(args.device).eval()
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
        metadata["normalization"]["mean"], metadata["normalization"]["std"])])
    test_data = datasets.CIFAR10(str(ROOT / "data"), train=False, transform=transform, download=False)
    loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
    confusion = torch.zeros((10, 10), dtype=torch.int64)
    predictions, labels_saved = [], []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(args.device))
            predicted = logits.argmax(1).cpu()
            total_loss += criterion(logits, labels.to(args.device)).item()
            confusion += torch.bincount(labels * 10 + predicted, minlength=100).reshape(10, 10)
            predictions.append(predicted.numpy())
            labels_saved.append(labels.numpy())
    correct = int(confusion.diag().sum())
    np.savez(output / "test_predictions.npz", labels=np.concatenate(labels_saved),
             predictions=np.concatenate(predictions), confusion=confusion.numpy())

    header_length = struct.unpack("<Q", packed[4:12])[0]
    header = json.loads(packed[12:12 + header_length])
    original = build_model(metadata["model_config"])
    weight_names = {name + ".weight" for name in compression_layers(original)}
    original_weights = sum(t.numel() * t.element_size() for name, t in original.state_dict().items() if name in weight_names)
    weight_payload = sum(entry["bytes"] for entry in header["tensors"] if entry["name"] in weight_names)
    header_bytes = header_length + 12
    scales_bytes = sum(entry.get("scale_count", 0) * 4 for entry in header["tensors"])
    codes_bytes = sum(entry.get("codes_bytes", 0) for entry in header["tensors"])
    original_state_bytes = metadata["original_state_bytes"]
    baseline_results = json.loads(baseline_results_path.read_text())
    final = {"setting": chosen["setting"], "weight_bits": chosen["weight_bits"],
             "activation_bits": chosen["activation_bits"], "validation_accuracy": chosen["validation_accuracy"],
             "test_accuracy": 100 * correct / len(test_data), "test_loss": total_loss / len(test_data),
             "test_correct": correct, "test_images": len(test_data), "baseline_test_accuracy": baseline_results["test_accuracy"],
             "test_accuracy_drop_pp": baseline_results["test_accuracy"] - 100 * correct / len(test_data),
             "model_sha256": model_hash, "baseline_sha256": metadata["baseline_sha256"],
             "model_file_bytes": len(packed), "model_size_mb": len(packed) / 1e6,
             "original_state_bytes": original_state_bytes, "model_compression_ratio": original_state_bytes / len(packed),
             "original_conv_linear_weight_bytes": original_weights, "packed_weight_payload_bytes": weight_payload,
             "weight_bytes_with_all_header_overhead": weight_payload + header_bytes,
             "weight_compression_ratio": original_weights / (weight_payload + header_bytes),
             "weight_ratio_definition": "Conv/Linear weights only, including scales and FP32 edge weights; entire file header charged to weights",
             "packed_code_bytes": codes_bytes, "weight_scale_bytes": scales_bytes,
             "other_raw_tensor_bytes": len(packed) - header_bytes - codes_bytes - scales_bytes,
             "header_bytes": header_bytes, **activation_storage(metadata["activations"]),
             "per_class_accuracy": {name: 100 * int(confusion[i, i]) / int(confusion[i].sum())
                                    for i, name in enumerate(metadata["classes"])},
             "device": args.device, "evaluated_at": datetime.now().isoformat(timespec="seconds")}
    assert math.isclose(final["model_compression_ratio"], chosen["weight_compression_ratio"])
    assert math.isclose(final["activation_compression_ratio"], chosen["activation_compression_ratio"])
    save_json(results_path, final)
    print(json.dumps(final, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Compare compression settings or test the chosen model.")
    commands = parser.add_subparsers(dest="command", required=True)
    sweep = commands.add_parser("sweep", help="Compare weight/activation bit settings on validation images")
    final = commands.add_parser("final", help="Choose on validation, then evaluate the test set")
    for command in (sweep, final):
        command.add_argument("--run-dir", help="Use a new training run instead of the supplied baseline")
        command.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        command.add_argument("--batch-size", type=int, default=128)
    sweep.add_argument("--output-dir", default=str(ROOT / "runs/sweep"))
    sweep.add_argument("--preview", action="store_true")
    sweep.add_argument("--settings", nargs="+", default=[f"{w}/{a}" for w, a in DEFAULT_SETTINGS])
    sweep.add_argument("--calibration-per-class", type=int)
    sweep.add_argument("--validation-per-class", type=int)
    sweep.add_argument("--cpu-threads", type=int, default=4)
    sweep.add_argument("--quantize-edges", action="store_true")
    sweep.add_argument("--no-fold-bn", action="store_true")
    final.add_argument("--sweep-dir", default=str(RESULTS))
    final.add_argument("--output-dir", default=str(ROOT / "runs/final"))
    final.add_argument("--min-validation-accuracy", type=float, default=90.0)
    args = parser.parse_args()
    if args.command == "final":
        evaluate_final(args)
    else:
        settings = [tuple(map(int, pair.split("/"))) for pair in args.settings]
        if any(len(pair) != 2 for pair in settings):
            parser.error("Use weight/activation bits, such as 4/8.")
        run_experiments(args.run_dir, args.output_dir, args.preview, args.device, settings,
                        args.calibration_per_class, args.validation_per_class,
                        args.batch_size, args.cpu_threads, not args.quantize_edges, not args.no_fold_bn)


if __name__ == "__main__":
    main()
