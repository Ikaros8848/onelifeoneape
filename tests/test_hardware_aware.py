"""Evaluate the final SCAMP-aware ternary QAT model.

Examples:
    python -m baseline.test_hardware_aware
    python -m baseline.test_hardware_aware --max-samples 1000 --cpu
"""
import argparse
import json
import torch
from torch.utils.data import DataLoader

from tests.test_mnist import TestMNIST
from baseline.hardware_aware import HardwareAwareCNN, ternary_quantize


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="artifacts/checkpoints/mnist_scamp_qat.pt")
    p.add_argument("--data-root", default="data")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-samples", type=int)
    p.add_argument("--threshold-ratio", type=float, default=0.05)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model = HardwareAwareCNN(quantize=True, threshold_ratio=args.threshold_ratio,
                             per_channel=True)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()

    dataset = TestMNIST(args.data_root, args.max_samples)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    confusion = torch.zeros((10, 10), dtype=torch.int64)
    correct = total = 0
    with torch.inference_mode():
        for images, labels in loader:
            pred = model(images.to(device)).argmax(1).cpu()
            correct += int((pred == labels).sum())
            total += len(labels)
            for y, p_ in zip(labels, pred):
                confusion[int(y), int(p_)] += 1

    stats = {}
    for name in ("conv_weight", "fc_weight"):
        w = getattr(model, name).detach().cpu()
        _, signs, scales, _ = ternary_quantize(w, args.threshold_ratio, True)
        stats[name] = {
            "plus_fraction": float((signs > 0).float().mean()),
            "zero_fraction": float((signs == 0).float().mean()),
            "minus_fraction": float((signs < 0).float().mean()),
            "scale_mean": float(scales.mean()),
            "scale_min": float(scales.min()),
            "scale_max": float(scales.max()),
        }

    print(json.dumps({
        "model": "SCAMP-aware ternary QAT",
        "checkpoint": args.checkpoint,
        "device": str(device),
        "samples": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "threshold_ratio": args.threshold_ratio,
        "weight_statistics": stats,
        "confusion_matrix": confusion.tolist(),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
