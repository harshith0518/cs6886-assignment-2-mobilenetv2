"""Choose the Q4 setting using validation results, then evaluate its test accuracy."""
import argparse
import hashlib
import json
import math
import struct
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from manual_compression import activation_storage, compression_layers, load_compressed
from train_baseline import ROOT, DEFAULT_RUN, build_model, save_json


def run(args):
    source = Path(args.sweep_dir).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    results_path = output / "results.json"
    sweep = json.loads((source / "results.json").read_text())
    run_dir = Path(args.run_dir).resolve()
    baseline_hash = hashlib.sha256((run_dir / "best.pt").read_bytes()).hexdigest()
    if baseline_hash != sweep["baseline_sha256"]:
        raise ValueError("The baseline folder does not match the compression sweep.")
    choices = [row for row in sweep["rows"] if row["setting"] != "FP32"
               and row["validation_accuracy"] >= args.min_validation_accuracy]
    if not choices:
        raise ValueError("No compressed setting meets the validation accuracy limit.")
    chosen = min(choices, key=lambda row: (row["model_size_mb"], -row["validation_accuracy"]))
    packed = (source / (chosen["setting"] + ".mq")).read_bytes()
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
    baseline_results = json.loads((run_dir / "results.json").read_text())
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", default=str(ROOT / "artifacts/compression/q3_final_grid"))
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN))
    parser.add_argument("--output-dir", default=str(ROOT / "artifacts/final"))
    parser.add_argument("--min-validation-accuracy", type=float, default=90.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    run(parser.parse_args())
