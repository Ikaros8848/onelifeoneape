"""Unified software accuracy and SCAMP proxy benchmark.

Example:
    python benchmark.py --model qat:qat:artifacts/checkpoints/mnist_scamp_qat.pt \
        --model fp32:fp32:artifacts/checkpoints/mnist_scamp_software.pt \
        --output results/comparison.csv

The latency and energy fields are normalized/analytical proxies, not hardware
measurements.
"""
import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from baseline.evaluate import metrics
from baseline.hardware_aware import HardwareAwareCNN, ternary_quantize
from baseline.torch_model import build_model
from hardware_model import ScampCostModel, compare_reports
from optimized.offset_gated import OffsetGatedHardwareAwareCNN
from tests.test_mnist import TestMNIST


def parse_model(value):
    parts = value.split(":", 2)
    if len(parts) != 3 or parts[1] not in {"fp32", "qat", "offset"}:
        raise argparse.ArgumentTypeError("model must be NAME:fp32|qat|offset:CHECKPOINT")
    return tuple(parts)


def load_model(kind, checkpoint, threshold_ratio):
    if kind == "fp32":
        model = build_model()
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        return model
    if kind == "offset":
        model = OffsetGatedHardwareAwareCNN(quantize=True, threshold_ratio=threshold_ratio,
                                           per_channel=True, gate_mode="hard")
    else:
        model = HardwareAwareCNN(quantize=True, threshold_ratio=threshold_ratio, per_channel=True)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return model


def evaluate(model, loader, device):
    truth, pred = [], []
    model.to(device).eval()
    with torch.inference_mode():
        for images, labels in loader:
            out = model(images.to(device)).argmax(1).cpu()
            truth.extend(labels.tolist())
            pred.extend(out.tolist())
    return metrics(truth, pred, 10)


def cost_for(model, kind, name, threshold_ratio):
    if kind == "qat":
        _, conv_sign, _, _ = ternary_quantize(model.conv_weight.detach().cpu(), threshold_ratio, True)
        _, fc_sign, _, _ = ternary_quantize(model.fc_weight.detach().cpu(), threshold_ratio, True)
    elif kind == "offset":
        _, conv_sign, _, _ = ternary_quantize(model.conv_weight.detach().cpu(), threshold_ratio, True)
        _, fc_sign, _, _ = ternary_quantize(model.fc_weight.detach().cpu(), threshold_ratio, True)
        conv_sign = conv_sign * (model.offset_mask(hard=True).detach().cpu() > 0.5).to(conv_sign.dtype).view(1, 1, 4, 4)
    else:
        # The float model is an accuracy/complexity reference. Sign-only dense
        # accounting shows candidate terms but is not a deployable ternary map.
        conv_sign = model.conv.weight.detach().cpu().sign()
        fc_sign = model.fc.weight.detach().cpu().sign()
    report = ScampCostModel().analyze_ternary(
        name, conv_sign, fc_sign, parameters=sum(p.numel() for p in model.parameters()))
    if kind == "fp32":
        report.mapping = "float_dense_reference_not_scamp_deployment"
    return report


def flatten_row(name, kind, checkpoint, task, cost, comparison):
    row = {
        "method": name, "kind": kind, "checkpoint": checkpoint,
        "accuracy": task["accuracy"],
        "precision_macro": task["macro_precision"],
        "recall_macro": task["macro_recall"],
        "f1_macro": task["macro_f1"],
        "parameters": cost.parameters, "dense_terms": cost.dense_terms,
        "active_terms": cost.active_terms, "skipped_terms": cost.skipped_terms,
        # `active_terms` is the logical deployed SCAMP schedule count. The
        # current PyTorch reference still launches dense conv/linear kernels;
        # expose that fact instead of implying CPU zero-skipping.
        "logical_deployment_active_terms": cost.active_terms,
        "actual_software_candidate_terms": cost.dense_terms,
        "software_inference_skip_realized": False,
        "positive_terms": cost.positive_terms, "negative_terms": cost.negative_terms,
        "data_movement_external_bits": cost.external_movement_bits,
        "data_movement_local_elements": cost.local_movement_elements,
        "feature_footprint_elements": cost.feature_footprint_elements,
        "register_pressure_proxy": cost.register_pressure,
        "retained_offsets": cost.retained_offsets,
        "offset_route_depth": cost.offset_route_depth,
        "latency_proxy": cost.latency_proxy, "energy_proxy": cost.energy_proxy,
        "measurement_boundary": "software task metrics; logical sparse schedule and normalized hardware proxies; PyTorch kernels remain dense",
    }
    if comparison:
        row.update({f"normalized_{k}": v for k, v in comparison["normalized"].items()})
        row.update({f"composite_{k}": v for k, v in comparison["composite_cost"].items()})
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", type=parse_model, required=True)
    parser.add_argument("--reference", help="model NAME used to normalize proxy metrics")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--threshold-ratio", type=float, default=0.05)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output", default="results/comparison.csv")
    parser.add_argument("--json-output", default="results/benchmark.json")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    dataset = TestMNIST(args.data_root, args.max_samples)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    records = []
    for name, kind, checkpoint in args.model:
        model = load_model(kind, checkpoint, args.threshold_ratio)
        records.append({
            "name": name, "kind": kind, "checkpoint": checkpoint,
            "task": evaluate(model, loader, device),
            "cost": cost_for(model, kind, name, args.threshold_ratio),
        })

    reference_name = args.reference or next((r["name"] for r in records if r["kind"] == "qat"), records[0]["name"])
    reference = next((r for r in records if r["name"] == reference_name), None)
    if reference is None:
        raise SystemExit(f"reference model {reference_name!r} was not supplied")

    rows, detailed = [], []
    for record in records:
        comparison = compare_reports(record["cost"], reference["cost"])
        rows.append(flatten_row(record["name"], record["kind"], record["checkpoint"],
                                record["task"], record["cost"], comparison))
        detailed.append({
            "name": record["name"], "kind": record["kind"],
            "checkpoint": record["checkpoint"], "task": record["task"],
            "hardware_proxy": record["cost"].to_dict(),
            "normalized_to_reference": comparison,
        })

    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    json_output = Path(args.json_output); json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps({
        "reference": reference_name, "device": str(device),
        "samples": len(dataset), "models": detailed,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"csv": str(output), "json": str(json_output),
                      "reference": reference_name, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
