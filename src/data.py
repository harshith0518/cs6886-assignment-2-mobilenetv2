"""Prepare the class-balanced CIFAR-10 split and training-only normalization."""
import argparse
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
from torchvision.datasets import CIFAR10

from src.train import ROOT, build_model, save_json


def prepare(output_dir, download=False):
    train = CIFAR10(str(ROOT / "data"), train=True, download=download)
    test = CIFAR10(str(ROOT / "data"), train=False, download=download)
    labels = np.array(train.targets)
    rng = np.random.default_rng(42)
    train_parts, validation_parts = [], []
    for class_id in range(10):
        indices = rng.permutation(np.flatnonzero(labels == class_id))
        validation_parts.append(indices[:500])
        train_parts.append(indices[500:])
    train_indices = rng.permutation(np.concatenate(train_parts))
    validation_indices = rng.permutation(np.concatenate(validation_parts))

    # Use training pixels only. Small chunks keep the RAM use manageable.
    channel_sum = np.zeros(3, dtype=np.float64)
    squared_sum = np.zeros(3, dtype=np.float64)
    count = 0
    for start in range(0, len(train_indices), 256):
        pixels = train.data[train_indices[start:start + 256]].astype(np.float64) / 255.0
        channel_sum += pixels.sum(axis=(0, 1, 2))
        squared_sum += np.square(pixels).sum(axis=(0, 1, 2))
        count += pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
    mean = channel_sum / count
    std = np.sqrt(np.maximum(squared_sum / count - mean ** 2, 0))
    model_config = {"architecture": "MobileNet-v2", "pretrained": False,
                    "num_classes": 10, "width_mult": 1.0, "dropout": 0.2,
                    "first_convolution_stride": 1, "other_stage_strides": "torchvision defaults",
                    "batch_norm_eps": 1e-5, "batch_norm_momentum": 0.1}
    model = build_model(model_config)
    model_config["parameter_count"] = sum(p.numel() for p in model.parameters())
    model_config["uncompressed_tensor_bytes"] = sum(t.numel() * t.element_size() for t in model.state_dict().values())
    packages = ["torch", "torchvision", "numpy", "matplotlib", "pillow", "psutil"]
    config = {"seed": 42, "dataset": "CIFAR-10", "image_size": 32, "classes": train.classes,
              "splits": {"train": len(train_indices), "validation": len(validation_indices), "test": len(test)},
              "split_method": "seeded permutation within each class; 500 validation examples per class",
              "normalization": {"mean": mean.tolist(), "std": std.tolist(),
                                "computed_from": "training split only, before augmentation"},
              "training_augmentation": {"random_crop_size": 32, "padding": 4, "padding_mode": "reflect",
                                        "horizontal_flip_probability": 0.5},
              "loader": {"batch_size": 32, "num_workers": 0},
              "model": model_config, "python": platform.python_version(),
              "packages": {name: version(name) for name in packages}}
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / "split_indices.npz", train=train_indices, validation=validation_indices)
    save_json(output_dir / "config.json", config)
    print(f"Saved 45,000 training and 5,000 validation indices to {output_dir}")
    print(f"RGB mean: {mean}; standard deviation: {std}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Download CIFAR-10 if it is missing")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/prepared")
    args = parser.parse_args()
    prepare(args.output_dir, args.download)
