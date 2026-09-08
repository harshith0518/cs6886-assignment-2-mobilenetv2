"""Compare manual compression settings on validation images, without retraining."""
import argparse
import csv
import hashlib
import io
import json
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from train_baseline import ROOT, DEFAULT_RUN, DatasetView, build_model, evaluate, training_is_running
from manual_compression import (
    activation_plan, activation_storage, add_activation_quantization,
    calibrate_activations, compress_weights, save_compressed, load_compressed,
)

DEFAULT_SETTINGS = [(8, 8), (4, 8), (4, 4)]


def balanced_subset(indices, labels, per_class, seed):
    # Same number of cats, dogs, etc. A tiny preview shouldn't contain one class only.
    rng = np.random.default_rng(seed)
    selected = []
    for label in range(10):
        choices = indices[labels[indices] == label]
        if not 0 < per_class <= len(choices):
            raise ValueError(f"Requested {per_class} images per class, but class {label} has {len(choices)}.")
        selected.extend(rng.permutation(choices)[:per_class])
    return rng.permutation(np.array(selected, dtype=np.int64))


def run_experiments(run_dir=DEFAULT_RUN, output_dir=None, preview=False, device="cpu",
                    settings=None, calibration_per_class=None, validation_per_class=None,
                    batch_size=32, cpu_threads=2, keep_first_last=True, fold_bn=True):
    run_dir = Path(run_dir)
    finished = (run_dir / "results.json").exists()
    if not finished and not preview:
        raise RuntimeError("Baseline isn't finished yet. Use preview=True (CLI: --preview) for a small interim check.")
    if training_is_running(run_dir) and device.startswith("cuda"):
        raise RuntimeError("The GPU is training the baseline. Use CPU for this preview.")
    if batch_size < 1 or cpu_threads < 1:
        raise ValueError("Batch size and CPU thread count must be positive.")
    settings = list(DEFAULT_SETTINGS if settings is None else settings)
    if not settings or len(settings) != len(set(settings)):
        raise ValueError("Give at least one setting, without duplicates.")
    calibration_per_class = calibration_per_class if calibration_per_class is not None else (10 if preview else 100)
    validation_per_class = validation_per_class if validation_per_class is not None else (20 if preview else 500)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_dir) if output_dir else ROOT / "artifacts/compression" / ("preview_" + stamp if preview else "validation_" + stamp)
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(cpu_threads)
    torch.manual_seed(42)

    # Read one whole best.pt version. Training can safely keep replacing its own file.
    snapshot = (run_dir / "best.pt").read_bytes()
    checkpoint_hash = hashlib.sha256(snapshot).hexdigest()
    checkpoint = torch.load(io.BytesIO(snapshot), map_location="cpu", weights_only=False)
    (output_dir / "baseline_snapshot.pt").write_bytes(snapshot)
    preparation = checkpoint["config"]["preparation"]
    baseline = build_model(preparation["model"]).eval()
    baseline.load_state_dict(checkpoint["model"])
    original_bytes = sum(t.numel() * t.element_size() for t in baseline.state_dict().values())

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
        preparation["normalization"]["mean"], preparation["normalization"]["std"])])
    raw = datasets.CIFAR10(str(ROOT / "data"), train=True, download=False)
    labels = np.array(raw.targets)
    with np.load(run_dir / "split_indices.npz") as splits:
        train_indices = splits["train"].copy()
        validation_indices = splits["validation"].copy()
    cal_indices = balanced_subset(train_indices, labels, calibration_per_class, 42)
    val_indices = balanced_subset(validation_indices, labels, validation_per_class, 43)
    if np.intersect1d(cal_indices, val_indices).size:
        raise ValueError("Calibration and validation must be disjoint.")
    np.savez(output_dir / "evaluation_indices.npz", calibration=cal_indices, validation=val_indices)
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
    (output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {output_dir}", flush=True)
    return output_dir, result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN))
    parser.add_argument("--output-dir")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--settings", nargs="+", default=["8/8", "4/8", "4/4"], help="weight/activation bits, e.g. 8/8 4/8")
    parser.add_argument("--calibration-per-class", type=int)
    parser.add_argument("--validation-per-class", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--quantize-edges", action="store_true", help="Also quantize the first convolution and classifier")
    parser.add_argument("--no-fold-bn", action="store_true")
    args = parser.parse_args()
    settings = [tuple(map(int, pair.split("/"))) for pair in args.settings]
    if any(len(pair) != 2 for pair in settings):
        parser.error("Each setting needs weight/activation bits, e.g. 4/8.")
    run_experiments(args.run_dir, args.output_dir, args.preview, args.device, settings,
                    args.calibration_per_class, args.validation_per_class,
                    args.batch_size, args.cpu_threads, not args.quantize_edges, not args.no_fold_bn)
