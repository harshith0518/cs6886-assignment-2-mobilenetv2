"""Train the CIFAR-10 baseline. Example: python train_baseline.py --epochs 300"""
import argparse
import csv
import ctypes
import json
import math
import os
import random
import time
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import psutil
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v2

ROOT = Path(__file__).resolve().parent
DEFAULT_RUN = ROOT / "artifacts" / "training" / "mobilenetv2_300"


def replace_with_retry(source, destination, timeout=30.0):
    # Windows readers/scanners can briefly prevent an atomic file replacement.
    # Keep the previous complete file intact and retry, never truncate it.
    deadline = time.monotonic() + timeout
    while True:
        try:
            source.replace(destination)
            return
        except OSError as error:
            if getattr(error, "winerror", None) not in (5, 32, 33):
                raise
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.1)


def save_json(path, data):
    # Write a temporary file first, so a progress read never sees half a JSON file.
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    replace_with_retry(temporary, path)


def save_status(path, data):
    # Progress reporting must not discard an epoch if a reader holds its file.
    # Checkpoints, configuration, and final results still require successful writes.
    try:
        save_json(path, data)
    except OSError as error:
        print(f"Warning: could not update progress file: {error}", flush=True)


def save_checkpoint(path, checkpoint):
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    replace_with_retry(temporary, path)


def training_is_running(run_dir=DEFAULT_RUN):
    marker = Path(run_dir) / "process.json"
    if not marker.exists():
        return False
    try:
        info = json.loads(marker.read_text())
        if not isinstance(info.get("pid"), int):
            return False
        process = psutil.Process(info["pid"])
        return process.is_running() and any("train_baseline.py" in arg for arg in process.cmdline())
    except (psutil.Error, OSError, ValueError, KeyError):
        return False


class DatasetView(Dataset):
    def __init__(self, source, indices, transform):
        self.source, self.indices, self.transform = source, indices, transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, label = self.source[int(self.indices[index])]
        return self.transform(image), label


def build_model(model_config):
    model = mobilenet_v2(
        weights=None,
        num_classes=model_config["num_classes"],
        width_mult=model_config["width_mult"],
        dropout=model_config["dropout"],
        norm_layer=partial(nn.BatchNorm2d, eps=1e-5, momentum=0.1),
    )
    model.features[0][0].stride = (1, 1)
    return model


def learning_rate(epoch, config):
    # Small steps for the first 5 epochs, then slowly reduce the step size.
    warmup = config["warmup_epochs"]
    if epoch < warmup:
        return config["learning_rate"] * (epoch + 1) / warmup
    progress = (epoch - warmup) / max(1, config["epochs"] - warmup - 1)
    return config["min_lr"] + 0.5 * (config["learning_rate"] - config["min_lr"]) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, loader, criterion, device, channels_last=False):
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        if channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        # Evaluate in FP32 even if mixed precision is used during training.
        logits = model(images)
        total_loss += criterion(logits, labels).item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        seen += labels.size(0)
    return total_loss / seen, 100 * correct / seen


def plot_history(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for split in ["train", "val"]:
        axes[0].plot(epochs, [row[f"{split}_loss"] for row in history], label=split)
        axes[1].plot(epochs, [row[f"{split}_accuracy"] for row in history], label=split)
    axes[0].set_ylabel("Cross-entropy loss (label smoothing 0.05)")
    axes[1].set_ylabel("Accuracy (%)")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_training(args):
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if training_is_running(run_dir):
        raise RuntimeError("Training is already running in this folder.")
    if (run_dir / "last.pt").exists() and not args.resume:
        raise RuntimeError("A checkpoint already exists. Use --resume to continue it.")

    save_json(run_dir / "process.json", {"pid": os.getpid(), "started": datetime.now().isoformat(timespec="seconds")})
    config_path = run_dir / "config.json"
    if args.resume:
        config = json.loads(config_path.read_text())
        if args.epochs != config["epochs"]:
            raise ValueError("Resume with the original epoch total so the LR schedule stays consistent.")
    else:
        preparation = json.loads((ROOT / "artifacts/preparation/config.json").read_text())
        config = {
            "epochs": args.epochs, "batch_size": args.batch_size,
            "learning_rate": args.lr, "min_lr": 1e-5, "warmup_epochs": 5,
            "optimizer": "SGD", "momentum": 0.9, "nesterov": True,
            "weight_decay": 5e-4, "weight_decay_on_bn_and_bias": False,
            "label_smoothing": 0.05, "gradient_clip_norm": 5.0,
            "random_erasing_probability": 0.25, "random_erasing_scale": [0.02, 0.15],
            "amp": args.amp, "channels_last": args.channels_last,
            "num_workers": 0, "cudnn_benchmark": args.fast_cudnn,
            "cpu_threads": 4,
            "cudnn_deterministic": not args.fast_cudnn,
            "validation_precision": "FP32", "target_accuracy_percent": [90, 93],
            "selection": "highest validation accuracy; test once after training",
            "preparation": preparation,
        }
        save_json(config_path, config)
        split_bytes = (ROOT / "artifacts/preparation/split_indices.npz").read_bytes()
        (run_dir / "split_indices.npz").write_bytes(split_bytes)
        (run_dir / "training_source.py").write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")

    seed = config["preparation"]["seed"]
    torch.set_num_threads(config["cpu_threads"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = config["cudnn_benchmark"]
    torch.backends.cudnn.deterministic = config["cudnn_deterministic"]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Check the .venv kernel and GPU setup.")
    device = torch.device("cuda")

    mean = config["preparation"]["normalization"]["mean"]
    std = config["preparation"]["normalization"]["std"]
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(mean, std),
        # Hide a small patch sometimes, so the model doesn't depend on one tiny detail.
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.5, 2.0), value=0),
    ])
    eval_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    raw = datasets.CIFAR10(str(ROOT / "data"), train=True, download=False)
    splits = np.load(run_dir / "split_indices.npz")
    train_data = DatasetView(raw, splits["train"], train_transform)
    val_data = DatasetView(raw, splits["validation"], eval_transform)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True,
                              generator=generator, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=config["batch_size"], shuffle=False,
                            num_workers=0, pin_memory=True)
    model = build_model(config["preparation"]["model"]).to(device)
    if config["channels_last"]:
        model = model.to(memory_format=torch.channels_last)

    # Weight decay on conv/linear weights; leave BN parameters and biases alone.
    decay = [p for p in model.parameters() if p.ndim > 1]
    no_decay = [p for p in model.parameters() if p.ndim <= 1]
    optimizer = torch.optim.SGD(
        [{"params": decay, "weight_decay": config["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=config["learning_rate"], momentum=0.9, nesterov=True,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
    scaler = torch.amp.GradScaler("cuda", enabled=config["amp"])
    start_epoch, best_accuracy, best_epoch, history = 0, -1.0, 0, []

    if args.resume:
        # This is our own local checkpoint, including optimizer and random states.
        checkpoint = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"]
        best_accuracy, best_epoch = checkpoint["best_accuracy"], checkpoint["best_epoch"]
        history = checkpoint["history"]
        random.setstate(checkpoint["python_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        generator.set_state(checkpoint["loader_rng"])
        del checkpoint

    status = {"state": "running", "pid": os.getpid(), "gpu": torch.cuda.get_device_name(0),
              "epoch": start_epoch, "total_epochs": config["epochs"],
              "best_val_accuracy": max(best_accuracy, 0), "best_epoch": best_epoch}
    save_status(run_dir / "status.json", status)
    print(f"Training on {status['gpu']} | epochs={config['epochs']} | batch={config['batch_size']} | AMP={config['amp']}", flush=True)
    print("Test set is reserved until the best validation checkpoint is selected.", flush=True)
    if os.name == "nt":
        # Keep the machine awake while this run is active; the display can still turn off.
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)

    try:
        for epoch in range(start_epoch, config["epochs"]):
            epoch_start = time.perf_counter()
            lr = learning_rate(epoch, config)
            for group in optimizer.param_groups:
                group["lr"] = lr
            model.train()
            total_loss, correct, seen = 0.0, 0, 0
            status.update(epoch=epoch + 1, phase="training", batch=0, batches=len(train_loader), lr=lr)
            save_status(run_dir / "status.json", status)

            for batch, (images, labels) in enumerate(train_loader, 1):
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                if config["channels_last"]:
                    images = images.contiguous(memory_format=torch.channels_last)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16, enabled=config["amp"]):
                    logits = model(images)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"])
                scaler.step(optimizer)
                scaler.update()
                loss_value = loss.detach().item()
                if not math.isfinite(loss_value):
                    raise RuntimeError("Loss became non-finite; stopping instead of wasting the remaining epochs.")
                total_loss += loss_value * labels.size(0)
                correct += (logits.detach().argmax(1) == labels).sum().item()
                seen += labels.size(0)
                if batch % 25 == 0:
                    status.update(batch=batch, train_loss=total_loss / seen, train_accuracy=100 * correct / seen,
                                  updated=datetime.now().isoformat(timespec="seconds"))
                    save_status(run_dir / "status.json", status)

            status.update(phase="validation", batch=len(train_loader))
            save_status(run_dir / "status.json", status)
            val_loss, val_accuracy = evaluate(model, val_loader, criterion, device, config["channels_last"])
            row = {"epoch": epoch + 1, "lr": lr, "train_loss": total_loss / seen,
                   "train_accuracy": 100 * correct / seen, "val_loss": val_loss,
                   "val_accuracy": val_accuracy, "seconds": time.perf_counter() - epoch_start}
            history.append(row)
            if val_accuracy > best_accuracy:
                best_accuracy, best_epoch = val_accuracy, epoch + 1
                save_checkpoint(run_dir / "best.pt", {"model": model.state_dict(), "epoch": best_epoch,
                                                       "val_accuracy": best_accuracy, "config": config})

            save_checkpoint(run_dir / "last.pt", {
                "epoch": epoch + 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(), "best_accuracy": best_accuracy, "best_epoch": best_epoch,
                "history": history, "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                "loader_rng": generator.get_state(),
            })
            with (run_dir / "history.csv.tmp").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(history)
            replace_with_retry(run_dir / "history.csv.tmp", run_dir / "history.csv")
            recent_seconds = np.mean([entry["seconds"] for entry in history[-5:]])
            status.update(phase="epoch_finished", **row, best_val_accuracy=best_accuracy,
                          best_epoch=best_epoch, estimated_hours_remaining=float(recent_seconds * (config["epochs"] - epoch - 1) / 3600))
            save_status(run_dir / "status.json", status)
            print(f"Epoch {epoch + 1:3}/{config['epochs']} | lr {lr:.5f} | train {row['train_accuracy']:.2f}% | val {val_accuracy:.2f}% | best {best_accuracy:.2f}% | {row['seconds']:.1f}s", flush=True)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                plot_history(history, run_dir / "learning_curves.png")

        # One final test evaluation, using the checkpoint chosen only on validation.
        best = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(best["model"])
        raw_test = datasets.CIFAR10(str(ROOT / "data"), train=False, transform=eval_transform, download=False)
        test_loader = DataLoader(raw_test, batch_size=config["batch_size"], num_workers=0, pin_memory=True)
        test_loss, test_accuracy = evaluate(model, test_loader, nn.CrossEntropyLoss(), device, config["channels_last"])
        results = {"best_epoch": best_epoch, "best_validation_accuracy": best_accuracy,
                   "test_accuracy": test_accuracy, "test_loss": test_loss,
                   "completed_epochs": config["epochs"], "target_at_least_90_met": test_accuracy >= 90,
                   "model_tensor_mb": sum(t.numel() * t.element_size() for t in model.state_dict().values()) / 1e6}
        save_json(run_dir / "results.json", results)
        plot_history(history, run_dir / "learning_curves.png")
        status.update(state="completed", phase="completed", **results)
        save_status(run_dir / "status.json", status)
        print("Finished:", json.dumps(results), flush=True)
    except BaseException as error:
        status.update(state="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        save_status(run_dir / "status.json", status)
        raise
    finally:
        if os.name == "nt":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        save_json(run_dir / "process.json", {"pid": None, "ended": datetime.now().isoformat(timespec="seconds")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--fast-cudnn", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN))
    run_training(parser.parse_args())
