"""Retrain QAT weights under a predeclared fixed random 13-offset mask.

Uses the existing OffsetGatedHardwareAwareCNN and the same task/regularizer
training loop as train_offset_gated.py. Only the offset subset is fixed rather
than learned; the offset logits are frozen and excluded from the optimizer.
"""
import argparse, json, os, random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from baseline.hardware_aware import ternary_quantize
from optimized.offset_gated import OffsetGatedHardwareAwareCNN, export_deploy_state
from scripts.baseline.train_mnist import FirmwareMNIST, run_epoch
from scripts.optimized.train_offset_gated import seed_all


MASK_SEEDS = (101, 202, 303, 404, 505)
K = 13


def make_mask(mask_seed):
    offsets = [(i, j) for i in range(4) for j in range(4)]
    return sorted(random.Random(mask_seed).sample(offsets, K))


def prepare_all(root):
    masks = [make_mask(seed) for seed in MASK_SEEDS]
    if len({tuple(map(tuple, mask)) for mask in masks}) != len(masks):
        raise RuntimeError("predeclared random mask seeds produced a duplicate subset")
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    for seed, mask in zip(MASK_SEEDS, masks):
        folder = root / f"random{seed}_seed7"
        folder.mkdir(parents=True, exist_ok=True)
        record = {
            "method": "retrained_fixed_random_13_offset",
            "mask_seed": seed, "training_seed": 7, "k": K,
            "selected_offsets": mask,
            "training_config": {
                "init_checkpoint": "artifacts/checkpoints/mnist_scamp_qat.pt",
                "epochs": 4, "batch_size": 128, "optimizer": "Adam",
                "base_lr": 1e-3, "gate_lr": 2e-3,
                "gate_lr_status": "not used because fixed gate logits are frozen",
                "threshold_ratio": 0.05, "per_channel_scale": True,
                "gate_lambda": 0.001, "budget_lambda": 0.01,
                "preprocessing": "existing FirmwareMNIST / mnist_to_64",
                "data_augmentation": False,
            },
        }
        (folder / "mask.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return masks


def train_one(mask_seed, training_seed, output_dir, data_root="data"):
    seed_all(training_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    mask = make_mask(mask_seed)
    train = FirmwareMNIST(data_root, True)
    test = FirmwareMNIST(data_root, False)

    model = OffsetGatedHardwareAwareCNN(
        True, 0.05, True, target_retained_offsets=K, gate_mode="hard")
    state = torch.load("artifacts/checkpoints/mnist_scamp_qat.pt",
                       map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=False)
    with torch.no_grad():
        model.offset_logits.fill_(-20.0)
        for i, j in mask:
            model.offset_logits[i, j] = 20.0
    model.offset_logits.requires_grad_(False)
    model = model.to(device)

    # Same Adam update rule on all QAT weights as the learned-mask experiment.
    # The learned experiment's separate gate group is absent because this mask
    # is fixed by design; gate_lr is recorded but has no trainable parameters.
    optimizer = torch.optim.Adam(
        [p for name, p in model.named_parameters() if name != "offset_logits"],
        lr=1e-3)
    train_loader = DataLoader(train, batch_size=128, shuffle=True)
    test_loader = DataLoader(test, batch_size=128, shuffle=False)
    history = []
    for epoch in range(1, 5):
        model.train(); loss_sum = 0.0; seen = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            task_loss = F.cross_entropy(logits, labels)
            l1, budget = model.regularization()
            loss = task_loss + 0.001 * l1 + 0.01 * budget
            loss.backward(); optimizer.step()
            loss_sum += float(loss.detach()) * len(labels); seen += len(labels)
        with torch.inference_mode():
            test_metrics = run_epoch(model, test_loader, None, device)
        row = {"epoch": epoch, "train_loss": loss_sum / seen,
               "test": test_metrics, "mask_seed": mask_seed,
               "training_seed": training_seed}
        history.append(row); print(json.dumps(row), flush=True)

    train_checkpoint = output_dir / "checkpoint.pt"
    deploy_checkpoint = output_dir / "deploy.pt"
    torch.save(model.cpu().state_dict(), train_checkpoint)
    torch.save(export_deploy_state(model), deploy_checkpoint)
    report = {
        "method": "retrained_fixed_random_13_offset",
        "mask_seed": mask_seed, "training_seed": training_seed,
        "selected_offsets": mask, "k": K,
        "training_config": {
            "init_checkpoint": "artifacts/checkpoints/mnist_scamp_qat.pt",
            "epochs": 4, "batch_size": 128, "optimizer": "Adam",
            "base_lr": 1e-3, "gate_lr": 2e-3,
            "gate_lr_status": "not used; fixed gate logits frozen",
            "threshold_ratio": 0.05, "per_channel_scale": True,
            "gate_lambda": 0.001, "budget_lambda": 0.01,
            "preprocessing": "existing FirmwareMNIST / mnist_to_64",
            "data_augmentation": False,
        },
        "device": str(device), "history": history,
        "checkpoint": str(train_checkpoint), "deploy_checkpoint": str(deploy_checkpoint),
        "test_samples_per_epoch": len(test),
        "note": "software experiment; no SCAMP-5 hardware measurement",
    }
    (output_dir / "experiment.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mask-seed", type=int)
    p.add_argument("--training-seed", type=int, default=7)
    p.add_argument("--output-dir")
    p.add_argument("--data-root", default="data")
    p.add_argument("--prepare-all", action="store_true")
    p.add_argument("--root", default="results/retrained_random13")
    a = p.parse_args()
    if a.prepare_all:
        masks = prepare_all(a.root)
        print(json.dumps({"mask_seeds": MASK_SEEDS,
                          "masks": dict(zip(MASK_SEEDS, masks))}, indent=2))
        return
    if a.mask_seed is None or not a.output_dir:
        p.error("--mask-seed and --output-dir are required unless --prepare-all is used")
    train_one(a.mask_seed, a.training_seed, a.output_dir, a.data_root)


if __name__ == "__main__": main()
