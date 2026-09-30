"""Stage-2 validation: seed metrics, schedule trace, sparse correctness, and energy ablation."""
import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from baseline.evaluate import metrics
from baseline.hardware_aware import ternary_quantize
from hardware_model.cost_model import ScampCostModel
from hardware_model.energy import energy_ablation, energy_components
from hardware_model.sparse_schedule_simulator import (
    build_schedule, retained_offsets, sparse_reference_conv, summarize_schedule,
)
from tests.test_mnist import TestMNIST


def load_deploy(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    return state


def signs_and_report(name, state):
    conv = state["conv_weight"]
    fc = state["fc_weight"]
    cs, fs = conv.sign(), fc.sign()
    report = ScampCostModel().analyze_ternary(name, cs, fs, parameters=41242)
    return conv, state["conv_bias"], fc, state["fc_bias"], cs, fs, report


def dense_masked_conv(x, weight, bias, offsets):
    mask = torch.zeros_like(weight)
    for i, j in offsets:
        mask[:, :, i, j] = weight[:, :, i, j]
    return F.conv2d(F.pad(x, (1, 2, 1, 2)), mask, bias)


def evaluate_deploy(state, dataset, batch_size=256):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    truth, pred = [], []
    with torch.inference_mode():
        for images, labels in loader:
            conv = F.conv2d(F.pad(images, (1, 2, 1, 2)), state["conv_weight"], state["conv_bias"])
            features = F.max_pool2d(F.relu(conv), 4, 4).flatten(1)
            logits = F.linear(features, state["fc_weight"], state["fc_bias"])
            truth.extend(labels.tolist()); pred.extend(logits.argmax(1).tolist())
    return metrics(truth, pred, 10), truth, pred


def compare_dense_sparse_full(state, dataset, offsets, batch_size=256):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    max_error = 0.0; sum_error = 0.0; count = 0; agreement = 0; total = 0
    with torch.inference_mode():
        for images, _ in loader:
            dense = dense_masked_conv(images, state["conv_weight"], state["conv_bias"], offsets)
            sparse = sparse_reference_conv(images, state["conv_weight"], state["conv_bias"], offsets)
            err = (dense - sparse).abs()
            max_error = max(max_error, float(err.max()))
            sum_error += float(err.sum()); count += err.numel()
            dense_logits = F.linear(F.max_pool2d(F.relu(dense), 4, 4).flatten(1), state["fc_weight"], state["fc_bias"])
            sparse_logits = F.linear(F.max_pool2d(F.relu(sparse), 4, 4).flatten(1), state["fc_weight"], state["fc_bias"])
            agreement += int((dense_logits.argmax(1) == sparse_logits.argmax(1)).sum())
            total += len(images)
    return {
        "max_absolute_error": max_error,
        "mean_absolute_error": sum_error / count,
        "prediction_agreement": agreement / total,
        "samples": total,
    }


def time_conv(fn, x, repeats=3):
    with torch.inference_mode():
        fn(x)
        start = time.perf_counter()
        for _ in range(repeats): fn(x)
        return (time.perf_counter() - start) / repeats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data")
    p.add_argument("--output-dir", default="results")
    p.add_argument("--runtime-samples", type=int, default=32)
    args = p.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    dataset = TestMNIST(args.data_root)
    models = {
        "qat": ("artifacts/checkpoints/mnist_scamp_qat.pt", True),
        "13tap_seed7": ("artifacts/checkpoints/mnist_scamp_offset13.deploy.pt", False),
        "13tap_seed42": ("artifacts/checkpoints/mnist_scamp_offset13_seed42.deploy.pt", False),
        "14tap": ("artifacts/checkpoints/mnist_scamp_offset_gated.deploy.pt", False),
    }
    validation_rows, trace = [], {}
    reports = {}
    for name, path in models.items():
        path, latent_qat = path
        state = load_deploy(path)
        if latent_qat:
            qconv, _, _, _ = ternary_quantize(state["conv_weight"], .05, True)
            qfc, _, _, _ = ternary_quantize(state["fc_weight"], .05, True)
            state = dict(state, conv_weight=qconv.detach(), fc_weight=qfc.detach())
        task, truth, pred = evaluate_deploy(state, dataset)
        conv, bias, fc, fcb, cs, fs, report = signs_and_report(name, state)
        offsets = retained_offsets(cs)
        schedule = build_schedule(cs, offsets)
        summary = summarize_schedule(schedule)
        fc_plus = int((fs > 0).sum()); fc_minus = int((fs < 0).sum()); fc_zero = int((fs == 0).sum())
        summary.update({
            "conv_active_operations": summary["active_operations"],
            "conv_skipped_operations": summary["skipped_operations"],
            "fc_active_operations": fc_plus + fc_minus,
            "fc_skipped_operations": fc_zero,
            "total_active_operations": summary["active_operations"] + fc_plus + fc_minus,
            "within_retained_offset_skipped_operations": summary["skipped_operations"] + fc_zero,
            "removed_offset_candidate_operations": (16 - len(offsets)) * 16 * 64 * 64,
            "total_logical_skipped_operations": (
                summary["skipped_operations"] + fc_zero
                + (16 - len(offsets)) * 16 * 64 * 64
            ),
        })
        task_row = {
            "model": name, "checkpoint": path,
            "accuracy": task["accuracy"], "macro_precision": task["macro_precision"],
            "macro_recall": task["macro_recall"], "macro_f1": task["macro_f1"],
            "logical_active_operations": report.active_terms,
            "logical_skipped_operations": report.skipped_terms,
            "dense_candidate_operations": report.dense_terms,
            "schedule_length": summary["schedule_length"],
            "movement_external_bits": report.external_movement_bits,
            "movement_local_elements": report.local_movement_elements,
            "latency_proxy": report.latency_proxy, "energy_proxy_A": report.energy_proxy,
            "energy_proxy_B": energy_components(report, orthogonal_registers=True)["total"],
            "parameters": report.parameters,
            "software_inference_skip_realized": False,
            "confusion_matrix_json": json.dumps(task["confusion_matrix"], separators=(",", ":")),
        }
        validation_rows.append(task_row); reports[name] = report
        trace[name] = {"retained_offsets": offsets, "records": schedule, "summary": summary}

    with (out / "seed_validation.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(validation_rows[0])); writer.writeheader(); writer.writerows(validation_rows)
    (out / "stage2_schedule_trace.json").write_text(json.dumps(trace, indent=2), encoding="utf-8")

    # Dense masked versus explicit sparse reference on one deterministic batch.
    state = load_deploy(models["13tap_seed7"][0])
    conv, bias, _, _, cs, _, report = signs_and_report("13tap_seed7", state)
    offsets = retained_offsets(cs)
    x, labels = dataset[0]
    x = x.unsqueeze(0)
    dense = dense_masked_conv(x, conv, bias, offsets)
    sparse = sparse_reference_conv(x, conv, bias, offsets)
    err = (dense - sparse).abs()
    dense_logits = F.linear(F.max_pool2d(F.relu(dense), 4, 4).flatten(1), state["fc_weight"], state["fc_bias"])
    sparse_logits = F.linear(F.max_pool2d(F.relu(sparse), 4, 4).flatten(1), state["fc_weight"], state["fc_bias"])
    correctness = {
        "max_absolute_error": float(err.max()), "mean_absolute_error": float(err.mean()),
        "prediction_agreement": float((dense_logits.argmax(1) == sparse_logits.argmax(1)).float().mean()),
        "label": int(labels), "dense_prediction": int(dense_logits.argmax(1)),
        "sparse_prediction": int(sparse_logits.argmax(1)),
    }
    correctness.update({f"full_test_{k}": v for k, v in compare_dense_sparse_full(
        state, dataset, offsets).items()})
    (out / "sparse_correctness.json").write_text(json.dumps(correctness, indent=2), encoding="utf-8")

    # Runtime is a PC reference experiment only; use a small deterministic batch.
    xbatch = torch.stack([dataset[i][0] for i in range(min(args.runtime_samples, len(dataset)))])
    dense_time = time_conv(lambda z: dense_masked_conv(z, conv, bias, offsets), xbatch)
    sparse_time = time_conv(lambda z: sparse_reference_conv(z, conv, bias, offsets), xbatch)
    (out / "sparse_runtime.json").write_text(json.dumps({
        "samples_per_call": len(xbatch), "repeats": 3,
        "dense_masked_conv_seconds": dense_time,
        "sparse_reference_conv_seconds": sparse_time,
        "measurement_boundary": "CPU Python/PyTorch reference runtime; not SCAMP latency",
    }, indent=2), encoding="utf-8")

    energy_rows = []
    for name, report in reports.items():
        values = energy_ablation(report)
        energy_rows.append({"model": name, **values})
    with (out / "energy_ablation.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(energy_rows[0])); writer.writeheader(); writer.writerows(energy_rows)
    print(json.dumps({"seed_validation": str(out / "seed_validation.csv"),
                      "schedule_trace": str(out / "stage2_schedule_trace.json"),
                      "correctness": correctness, "dense_seconds": dense_time,
                      "sparse_seconds": sparse_time}, indent=2))


if __name__ == "__main__":
    main()
