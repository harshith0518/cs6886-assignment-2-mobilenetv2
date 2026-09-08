"""Check the Q3 measurements and optionally publish the native W&B chart."""
import argparse
import hashlib
import json
import math
import socket
from pathlib import Path

import numpy as np

from train_baseline import ROOT, DEFAULT_RUN, save_json

DEFAULT_RESULTS = ROOT / "artifacts/compression/q3_final_grid"
DEFAULT_PROJECT = "cs6886-assignment2-q3"
METRICS = ["validation_accuracy", "validation_loss", "accuracy_drop_pp",
           "model_size_mb", "weight_compression_ratio", "activation_compression_ratio",
           "model_file_bytes", "tensor_payload_bytes", "header_bytes",
           "activation_fp32_bytes_per_image", "activation_packed_bytes_per_image",
           "activation_metadata_bytes", "evaluation_seconds", "export_roundtrip_passed"]


def verify_results(folder, run_dir=DEFAULT_RUN, check_files=False):
    raw = (folder / "results.json").read_bytes()
    result = json.loads(raw)
    assert result["preview"] is False
    assert result["baseline_stage"] == "completed baseline"
    assert result["test_images_used"] == 0
    assert result["validation_images"] == 5000
    assert result["calibration_images"] == 1000
    assert result["keep_first_last_fp32"] and result["fold_bn"]
    baseline_hash = hashlib.sha256((run_dir / "best.pt").read_bytes()).hexdigest()
    assert baseline_hash == result["baseline_sha256"]
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
    checked_files = 0
    for row in rows:
        assert 0 <= row["validation_accuracy"] <= 100
        assert math.isfinite(row["validation_loss"])
        assert math.isclose(row["accuracy_drop_pp"], rows[0]["validation_accuracy"] - row["validation_accuracy"], abs_tol=1e-10)
        if row["setting"] == "FP32":
            continue
        packed = folder / (row["setting"] + ".mq")
        if folder == DEFAULT_RESULTS and not packed.exists():
            packed = ROOT / "artifacts/final" / packed.name
        if packed.exists():
            assert packed.stat().st_size == row["model_file_bytes"]
            checked_files += 1
        elif check_files:
            raise FileNotFoundError(f"Regenerate the full sweep to check {packed.name}.")
        file_bytes = row["model_file_bytes"]
        assert row["export_roundtrip_passed"] is True
        assert file_bytes == row["model_file_bytes"] == row["tensor_payload_bytes"] + row["header_bytes"]
        assert math.isclose(row["model_size_mb"], file_bytes / 1e6)
        assert math.isclose(row["weight_compression_ratio"], original_bytes / file_bytes)
        assert math.isclose(row["activation_compression_ratio"], row["activation_fp32_bytes_per_image"] /
                            (row["activation_packed_bytes_per_image"] + row["activation_metadata_bytes"]))
    result["experiment_id"] = "q3-" + hashlib.sha256(raw).hexdigest()[:16]
    result["packed_files_checked"] = checked_files
    return result


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
    lines = ["# Q3 results", "",
             f"Checkpoint epoch {result['baseline_epoch']}; 1,000 training calibration images; 5,000 validation images; no test images or fine-tuning.", "",
             "| Setting | Val. accuracy (%) | Drop (pp) | Model MB | Model ratio | Activation ratio |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in result["rows"]:
        lines.append(f"| {row['setting']} | {row['validation_accuracy']:.2f} | {row['accuracy_drop_pp']:.2f} | {row['model_size_mb']:.6f} | {row['weight_compression_ratio']:.2f}x | {row['activation_compression_ratio']:.2f}x |")
    lines += ["", "Model ratios include the complete packed file. Activation ratios estimate packed Conv/Linear input occurrences plus scale/zero-point bytes; they are not peak GPU memory. Inference arithmetic remains FP32.", ""]
    manifest = folder / "wandb_online_manifest.json"
    if manifest.exists():
        url = json.loads(manifest.read_text()).get("report_url")
        if url:
            lines += [f"[Native W&B Parallel Coordinates chart]({url})", ""]
    lines += ["The compact repository retains the baseline and chosen W4A8 model. Intermediate model files can be regenerated with the full-sweep command in README.md.", "",
              f"This check verified the saved numerical results and {result['packed_files_checked']} available packed file(s). Use --check-files after regenerating the sweep to require all nine files.", ""]
    (folder / "Q3_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--check-files", action="store_true", help="Require all nine packed files")
    parser.add_argument("--wandb-mode", choices=["none", "offline", "online"], default="none")
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--entity")
    args = parser.parse_args()
    folder = args.results_dir.resolve()
    result = verify_results(folder, args.run_dir.resolve(), args.check_files)
    if args.wandb_mode != "none":
        log_wandb(result, folder, args.wandb_mode, args.project, args.entity)
    write_answer(result, folder)
    print(f"Verified Q3 metrics and {result['packed_files_checked']}/9 packed files. Results: {folder}")


if __name__ == "__main__":
    main()
