"""Verify Q3 results, draw figures, and log the saved measurements to W&B.

This script never trains a model or evaluates the test set. Online publication
uses the official W&B SDK and Reports API; the compression itself is our code.
"""
import argparse
import hashlib
import json
import math
import socket
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from train_baseline import ROOT, DEFAULT_RUN, save_json

DEFAULT_RESULTS = ROOT / "artifacts/compression/q3_final_grid"
DEFAULT_PROJECT = "cs6886-assignment2-q3"
METRICS = ["validation_accuracy", "validation_loss", "accuracy_drop_pp",
           "model_size_mb", "weight_compression_ratio", "activation_compression_ratio",
           "model_file_bytes", "tensor_payload_bytes", "header_bytes",
           "activation_fp32_bytes_per_image", "activation_packed_bytes_per_image",
           "activation_metadata_bytes", "evaluation_seconds", "export_roundtrip_passed"]


def verify_results(folder, run_dir=DEFAULT_RUN):
    raw = (folder / "results.json").read_bytes()
    result = json.loads(raw)
    assert result["preview"] is False
    assert result["baseline_stage"] == "completed baseline"
    assert result["test_images_used"] == 0
    assert result["validation_images"] == 5000
    assert result["calibration_images"] == 1000
    assert result["keep_first_last_fp32"] and result["fold_bn"]
    snapshot_hash = hashlib.sha256((folder / "baseline_snapshot.pt").read_bytes()).hexdigest()
    assert snapshot_hash == result["baseline_sha256"]
    assert snapshot_hash == hashlib.sha256((run_dir / "best.pt").read_bytes()).hexdigest()
    baseline_results = json.loads((run_dir / "results.json").read_text())
    assert result["baseline_epoch"] == baseline_results["best_epoch"]
    with np.load(folder / "evaluation_indices.npz") as indices, np.load(run_dir / "split_indices.npz") as splits:
        cal, val = indices["calibration"], indices["validation"]
        assert len(np.unique(cal)) == 1000 and len(np.unique(val)) == 5000
        assert not np.intersect1d(cal, val).size
        assert np.isin(cal, splits["train"]).all()
        assert np.array_equal(np.sort(val), np.sort(splits["validation"]))
    rows = result["rows"]
    assert len(rows) == 10 and rows[0]["setting"] == "FP32"
    assert rows[0]["validation_accuracy"] == baseline_results["best_validation_accuracy"]
    assert {(r["weight_bits"], r["activation_bits"]) for r in rows[1:]} == {
        (w, a) for w in (2, 4, 8) for a in (2, 4, 8)}
    original_bytes = round(rows[0]["model_size_mb"] * 1e6)
    for row in rows:
        assert 0 <= row["validation_accuracy"] <= 100
        assert math.isfinite(row["validation_loss"])
        assert math.isclose(row["accuracy_drop_pp"], rows[0]["validation_accuracy"] - row["validation_accuracy"], abs_tol=1e-10)
        if row["setting"] == "FP32":
            continue
        file_bytes = (folder / (row["setting"] + ".mq")).stat().st_size
        assert row["export_roundtrip_passed"] is True
        assert file_bytes == row["model_file_bytes"] == row["tensor_payload_bytes"] + row["header_bytes"]
        assert math.isclose(row["model_size_mb"], file_bytes / 1e6)
        assert math.isclose(row["weight_compression_ratio"], original_bytes / file_bytes)
        assert math.isclose(row["activation_compression_ratio"], row["activation_fp32_bytes_per_image"] /
                            (row["activation_packed_bytes_per_image"] + row["activation_metadata_bytes"]))
    result["experiment_id"] = "q3-" + hashlib.sha256(raw).hexdigest()[:16]
    return result


def figures(result, folder):
    rows = result["rows"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    by_setting = {r["setting"]: r for r in rows}
    bits = [8, 4, 2]
    matrix = np.array([[by_setting[f"W{w}A{a}"]["validation_accuracy"] for a in bits] for w in bits])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), gridspec_kw={"width_ratios": [1, 1.25]})
    fig.suptitle("Q3 | MobileNet-v2 compression on CIFAR-10", x=0.06, ha="left", fontsize=19, fontweight="bold")
    fig.text(0.06, 0.89, "5,000 validation images per setting  |  1,000 training images for calibration  |  No fine-tuning", color="#465366")
    heat = axes[0].imshow(matrix, cmap="viridis", vmin=0, vmax=100)
    axes[0].set(xticks=range(3), xticklabels=bits, yticks=range(3), yticklabels=bits,
                xlabel="Activation bits", ylabel="Weight bits", title="Validation accuracy (%)")
    for i in range(3):
        for j in range(3):
            axes[0].text(j, i, f"{matrix[i, j]:.2f}%", ha="center", va="center",
                         fontsize=15, fontweight="bold", color="#102537" if matrix[i, j] > 60 else "white")
    fig.colorbar(heat, ax=axes[0], shrink=0.8, label="Accuracy (%)")
    colors = {8: "#247f70", 4: "#2774ae", 2: "#ba5639", 32: "#334155"}
    for w in [32, 8, 4, 2]:
        selected = [r for r in rows if r["weight_bits"] == w]
        axes[1].scatter([r["model_size_mb"] for r in selected], [r["validation_accuracy"] for r in selected],
                        s=60, c=colors[w], label="FP32" if w == 32 else f"{w}-bit weights", zorder=4)
    offsets = {"FP32": (-51, 10), "W8A8": (8, 9), "W8A4": (8, -17),
               "W4A8": (-54, 11), "W4A4": (-54, -17), "W8A2": (8, 8),
               "W4A2": (8, 8), "W2A8": (8, 18)}
    for name, offset in offsets.items():
        row = by_setting[name]
        axes[1].annotate(name if name != "W2A8" else "All W2 settings", (row["model_size_mb"], row["validation_accuracy"]),
                         xytext=offset, textcoords="offset points", fontsize=9)
    axes[1].axhline(rows[0]["validation_accuracy"], color="#64748b", linewidth=1, linestyle="--", alpha=0.55)
    axes[1].set(xscale="log", ylim=(0, 105), xlim=(0.6, 12), xlabel="Stored model size (MB, log scale)",
                ylabel="Validation accuracy (%)", title="Accuracy versus stored model size")
    axes[1].grid(alpha=0.15)
    axes[1].legend(loc="center right", frameon=False, fontsize=9)
    fig.text(0.06, 0.035, f"FP32: {rows[0]['validation_accuracy']:.2f}%. First convolution and final classifier stay FP32. Packed-file sizes include metadata.\nActivation precision is simulated in FP32; this experiment does not measure an integer-kernel speedup.", fontsize=9, color="#465366")
    fig.subplots_adjust(top=0.78, bottom=0.18, left=0.06, right=0.97, wspace=0.3)
    fig.savefig(folder / "q3_accuracy_compression.png", dpi=180)
    plt.close(fig)

    dimensions = [
        ("weight_bits", "Weight bits", [2, 4, 8, 32], True),
        ("activation_bits", "Activation bits", [2, 4, 8, 32], True),
        ("weight_compression_ratio", "Weight/model\nstorage ratio", [1, 4, 8, 12], False),
        ("activation_compression_ratio", "Activation\nsize ratio", [1, 4, 8, 12, 16], False),
        ("model_size_mb", "Stored model\nsize (MB)", [0, 2, 4, 6, 8, 10], False),
        ("validation_accuracy", "Validation\naccuracy (%)", [0, 20, 40, 60, 80, 100], False),
    ]
    fig, ax = plt.subplots(figsize=(15, 7.2))
    ax.set(xlim=(-0.5, 5.55), ylim=(-0.1, 1.13))
    ax.axis("off")
    fig.suptitle("Q3 | Parallel coordinates: nine compression settings and FP32", x=0.055, ha="left", fontsize=18, fontweight="bold")
    fig.text(0.055, 0.905, "Local companion figure. See Q3_RESULTS.md for the native W&B report and its publication status.", fontsize=10, color="#465366")

    def position(value, ticks, log):
        values = np.log2([value, ticks[0], ticks[-1]]) if log else [value, ticks[0], ticks[-1]]
        return (values[0] - values[1]) / (values[2] - values[1])

    for i, (_, title, ticks, log) in enumerate(dimensions):
        ax.plot([i, i], [0, 1], color="#a5b2c3", linewidth=1)
        ax.text(i, 1.085, title, ha="center", va="center", fontweight="bold", fontsize=11)
        for tick in ticks:
            y = position(tick, ticks, log)
            ax.text(i - 0.04, y, f"{tick:g}", ha="right", va="center", fontsize=9,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5}, zorder=5)
    cmap = plt.get_cmap("viridis")
    for row in sorted(rows, key=lambda r: r["validation_accuracy"]):
        ys = [position(row[key], ticks, log) for key, _, ticks, log in dimensions]
        ax.plot(range(len(dimensions)), ys, "o-", linewidth=1.6, markersize=3.5,
                color=cmap(row["validation_accuracy"] / 100), alpha=0.85,
                label=f"{row['setting']} ({row['validation_accuracy']:.2f}%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=5, frameon=False, fontsize=9)
    fig.text(0.055, 0.025, f"Bit-width axes use log2 spacing. Activation ratios estimate packed Conv/Linear input occurrences; they are not peak GPU memory.\nAll runs use the same epoch-{result['baseline_epoch']} checkpoint and validation split.", fontsize=9, color="#465366")
    fig.subplots_adjust(left=0.045, right=0.98, top=0.83, bottom=0.2)
    fig.savefig(folder / "q3_parallel_coordinates_local.png", dpi=180)
    plt.close(fig)


def payloads(result):
    payload = []
    for row in result["rows"]:
        config = {key: result[key] for key in ["experiment_id", "baseline_epoch", "baseline_sha256",
                  "calibration_images", "validation_images", "test_images_used", "keep_first_last_fp32",
                  "seed", "validation_seed", "device", "python", "torch", "numpy"]}
        config.update(setting=row["setting"], weight_bits=row["weight_bits"], activation_bits=row["activation_bits"],
                      model="MobileNet-v2", dataset="CIFAR-10", method="manual post-training quantization",
                      fine_tuning_epochs=0, fold_bn=row["setting"] != "FP32", split="validation",
                      size_basis=row["size_basis"], activation_size_is_estimate=True)
        payload.append({"name": row["setting"], "id": result["experiment_id"] + "-" + row["setting"].lower(),
                        "config": config, "metrics": {key: row[key] for key in METRICS if key in row}})
    return payload


def native_panel():
    import wandb_workspaces.reports.v2 as wr
    column = wr.ParallelCoordinatesPlotColumn
    return wr.ParallelCoordinatesPlot(
        title="Q3: compression settings and validation performance",
        layout=wr.Layout(x=0, y=0, w=24, h=10), font_size="medium",
        columns=[column(wr.Config("weight_bits"), display_name="Weight bits", log=True),
                 column(wr.Config("activation_bits"), display_name="Activation bits", log=True),
                 column("weight_compression_ratio", display_name="Weight/model storage ratio"),
                 column("activation_compression_ratio", display_name="Activation size ratio (estimate)"),
                 column("model_size_mb", display_name="Stored model size (MB)"),
                 column("validation_accuracy", display_name="Validation accuracy (%)")],
        gradient=[wr.GradientPoint(color="#440154", offset=0), wr.GradientPoint(color="#21918c", offset=50),
                  wr.GradientPoint(color="#fde725", offset=100)],
    )


def log_wandb(result, folder, mode, project, entity):
    import wandb
    import wandb_workspaces.reports.v2 as wr
    items = payloads(result)
    if mode == "online":
        # Resolve the cloud endpoint explicitly before starting the SDK service.
        # This also fails early if the laptop has lost its internet connection.
        socket.getaddrinfo("api.wandb.ai", 443, type=socket.SOCK_STREAM)
        from wandb.sdk.internal.internal_api import Api
        if not Api().api_key:
            raise RuntimeError("W&B login is required. Run .\\.venv\\Scripts\\wandb.exe login privately, then rerun with --wandb-mode online.")
        api = wandb.Api(timeout=60)
        entity = entity or api.default_entity
        if not entity:
            raise RuntimeError("Pass --entity with your W&B username or team.")
    manifest_path = folder / f"wandb_{mode}_manifest.json"
    manifest = {"experiment_id": result["experiment_id"], "mode": mode, "entity": entity,
                "project": project, "runs": []}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["experiment_id"] == result["experiment_id"] and previous.get("entity") == entity and previous["project"] == project:
            manifest = previous
    logged_ids = {item["id"] for item in manifest["runs"]}
    log_root = folder / f"wandb_{mode}"
    log_root.mkdir(exist_ok=True)
    for item in items:
        if item["id"] in logged_ids:
            continue
        options = {"project": project, "entity": entity, "id": item["id"], "name": item["name"],
                   "group": result["experiment_id"], "job_type": "compression-evaluation", "config": item["config"],
                   "tags": ["q3", "validation", "manual-quantization"], "mode": mode, "dir": str(log_root),
                   "settings": wandb.Settings(disable_git=True, disable_code=True, x_disable_stats=True,
                                              console="off", silent=True)}
        if mode == "online":
            options["resume"] = "allow"
        with wandb.init(**options) as run:
            run.log(item["metrics"], step=0)
            run.summary.update(item["metrics"])
            record = {"name": item["name"], "id": item["id"], "directory": str(Path(run.dir).parent)}
            if mode == "online":
                record["url"] = run.url
        manifest["runs"].append(record)
        save_json(manifest_path, manifest)
        print(f"W&B {mode}: {item['name']}", flush=True)
    if mode == "offline":
        manifest["native_chart_published"] = False
        manifest["next_step"] = "Login, then run report_q3.py --wandb-mode online. No model experiments need rerunning."
        save_json(manifest_path, manifest)
        return

    # Release the logging service before public-API verification. A busy service
    # immediately after finishing several runs can otherwise time out on reads.
    wandb.teardown()
    socket.getaddrinfo("api.wandb.ai", 443, type=socket.SOCK_STREAM)
    api = wandb.Api(timeout=60)
    uploaded = list(api.runs(f"{entity}/{project}", filters={"config.experiment_id": result["experiment_id"]}))
    by_id = {r.id: r for r in uploaded}
    assert set(by_id) == {item["id"] for item in items}, "Uploaded run set is incomplete. Rerun to finish publication."
    for item in items:
        remote = by_id[item["id"]]
        assert remote.state == "finished"
        for key in ("weight_bits", "activation_bits", "baseline_sha256"):
            assert remote.config[key] == item["config"][key]
        for key, value in item["metrics"].items():
            assert math.isclose(remote.summary[key], value, rel_tol=1e-9, abs_tol=1e-9), key
    report = wr.Report.from_url(manifest["report_url"].replace("\\", "/")) if manifest.get("report_url") else wr.Report(
        project=project, entity=entity, title="Q3 - MobileNet-v2 compression results", width="fluid",
        description="Nine manual quantization settings and FP32, evaluated on 5,000 CIFAR-10 validation images.")
    report.blocks = [
        wr.P(f"All experiments reuse the same epoch-{result['baseline_epoch']} baseline. Calibration uses 1,000 training images; validation uses all 5,000 held-out validation images. No test images or fine-tuning are used in Q3."),
        wr.PanelGrid(runsets=[wr.Runset(entity=entity, project=project,
                     filters=f"Config('experiment_id') == '{result['experiment_id']}'")], panels=[native_panel()]),
        wr.P(" | ".join(f"{row['setting']}: {row['validation_accuracy']:.2f}% validation accuracy, {row['weight_compression_ratio']:.2f}x model compression" for row in result["rows"])),
        wr.P("The first convolution and final classifier weights and inputs remain FP32. File sizes include packed weight codes, scales, unchanged tensors and metadata. Activation ratios are packed-size estimates for Conv/Linear input occurrences, not peak GPU memory. Inference arithmetic remains FP32."),
    ]
    report.save()
    # This Reports SDK version builds its URL using os.path.join; on Windows
    # normalize the separators before storing a link or passing it back to W&B.
    report_url = report.url.replace("\\", "/")
    manifest.update(report_url=report_url, project_url=f"https://wandb.ai/{entity}/{project}",
                    native_chart_published=True, uploaded_metrics_verified=True)
    save_json(manifest_path, manifest)
    restored = wr.Report.from_url(report_url)
    panels = [p for b in restored.blocks if isinstance(b, wr.PanelGrid) for p in b.panels]
    assert any(isinstance(p, wr.ParallelCoordinatesPlot) and len(p.columns) == 6 for p in panels)
    manifest["native_chart_roundtrip_verified"] = True
    save_json(manifest_path, manifest)
    print(f"Verified W&B report: {report_url}", flush=True)


def write_answer(result, folder):
    rows = result["rows"]
    table = ["| Setting | Validation accuracy (%) | Drop (pp) | Model size (MB) | Model ratio | Activation ratio |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        table.append(f"| {row['setting']} | {row['validation_accuracy']:.2f} | {row['accuracy_drop_pp']:.2f} | {row['model_size_mb']:.6f} | {row['weight_compression_ratio']:.2f}x | {row['activation_compression_ratio']:.2f}x |")
    manifest_path = folder / "wandb_online_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    chart = (f"[Native W&B Parallel Coordinates chart]({manifest['report_url']})"
             if manifest.get("native_chart_roundtrip_verified") else
             "The native W&B chart still needs online publication using the command below.")
    answer = "\n".join([
        "# Question 3 - Compression results", "", "## (a) Setup", "",
        f"The grid uses 8, 4 and 2 bits for weights and activations, giving nine compressed models and one FP32 baseline. All settings use the same checkpoint, chosen at epoch {result['baseline_epoch']}. There is no fine-tuning.", "",
        "The first convolution and final classifier weights and inputs stay FP32. The other Conv/Linear weights use per-channel symmetric quantization, after manually folding BatchNorm. Their inputs use affine quantization. Biases stay FP32.", "",
        "Calibration uses 1,000 training images (100 per class, seed 42) without augmentation. Evaluation uses all 5,000 validation images (500 per class, order seed 43). The saved indices are disjoint. No test images are used for Q3.", "",
        "## (b) Accuracy comparison", "", *table, "",
        "The drop is measured against FP32 on the same validation split, in percentage points. A larger compression ratio is only useful if enough accuracy is retained.", "",
        "![Validation comparison](q3_accuracy_compression.png)", "",
        "Model ratio = original parameters and buffers in bytes / complete packed file bytes. This includes the effect of BatchNorm folding, scales, unchanged tensors and the JSON header. The legacy JSON/W&B metric name is `weight_compression_ratio`; this is a whole-model storage measurement. Q4 separately calculates a weights-only ratio.", "",
        "Activation ratio compares the sum of FP32 Conv/Linear input occurrences per image with estimated packed inputs plus 8 bytes per quantized location for scale and zero point. Repeated occurrences are included. It is not a measurement of peak GPU memory. Inference still uses FP32 arithmetic with quantize/dequantize steps.", "",
        "All nine files were reloaded and matched their pre-export logits exactly on a validation batch. The report script also checks the source checkpoint hash, saved split membership, file sizes and ratio arithmetic. Results come from one training run and one calibration selection.", "",
        "## W&B Parallel Coordinates chart", "", chart, "",
        "The native panel has six axes: weight bits, activation bits, model ratio, activation ratio, model size and validation accuracy. Each setting is a separate W&B run. The image below is a local companion using the same saved measurements.", "",
        "![Local companion chart](q3_parallel_coordinates_local.png)", "",
        "## Commands", "", "See README.md for the full sweep command and environment setup. To check the saved run and redraw the figures:", "", "```powershell",
        "python report_q3.py --results-dir artifacts/compression/q3_final_grid --wandb-mode none",
        "# To publish a new native chart after logging in:", "wandb login",
        "python report_q3.py --results-dir artifacts/compression/q3_final_grid --wandb-mode online --entity YOUR_WANDB_ENTITY",
        "```", "",
        "For a new training run, also pass `--run-dir` with that run's folder. Online logging sends numerical results and configuration, not images or checkpoints.", "",
        f"Baseline SHA-256: `{result['baseline_sha256']}`.", "",
    ])
    (folder / "Q3_RESULTS.md").write_text(answer, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--wandb-mode", choices=["none", "offline", "online"], default="none")
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--entity")
    args = parser.parse_args()
    folder = args.results_dir.resolve()
    result = verify_results(folder, args.run_dir.resolve())
    figures(result, folder)
    save_json(folder / "wandb_payload.json", payloads(result))
    save_json(folder / "wandb_parallel_coordinates_spec.json", native_panel()._to_model().model_dump(mode="json", by_alias=True))
    sources = folder / "source"
    sources.mkdir(exist_ok=True)
    hashes = {}
    for name in ("manual_compression.py", "run_compression.py", "train_baseline.py", "report_q3.py"):
        content = (ROOT / name).read_bytes()
        target = sources / name
        if not target.exists():
            target.write_bytes(content)
        hashes[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    save_json(folder / "verification.json", {"passed": True, "configurations": 10,
              "compressed_export_roundtrips": 9, "baseline_sha256": result["baseline_sha256"],
              "calibration_validation_disjoint": True, "full_validation_split": True,
              "test_images_used": 0, "file_size_and_ratio_checks": True, "source_sha256": hashes})
    write_answer(result, folder)
    if args.wandb_mode != "none":
        log_wandb(result, folder, args.wandb_mode, args.project, args.entity)
        write_answer(result, folder)
    print(f"Verified Q3 results and figures: {folder}", flush=True)


if __name__ == "__main__":
    main()
