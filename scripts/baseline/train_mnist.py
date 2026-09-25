"""Train the SCAMP-shaped MNIST network as a PC software baseline.

This trains on the deterministic CPU approximation of the firmware input path
and saves an ordinary PyTorch checkpoint. It does not emulate SCAMP timing,
analog noise, or power consumption.
"""
import argparse
import json
import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from baseline.torch_model import build_model
from baseline.preprocessing import mnist_to_64


class FirmwareMNIST:
    def __init__(self, root, train):
        self.base = datasets.MNIST(root, train=train, download=False)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image, label = self.base[index]
        # Keep the same crop/centre/45px approximation used by the evaluator.
        image = image.convert("L")
        raw = list(image.tobytes())
        pixels28 = [raw[row * image.width:(row + 1) * image.width]
                    for row in range(image.height)]
        pixels = torch.tensor(mnist_to_64(pixels28), dtype=torch.float32).unsqueeze(0)
        return pixels, int(label)


def run_epoch(model, loader, optimizer, device):
    training = optimizer is not None
    model.train(training)
    loss_fn = torch.nn.CrossEntropyLoss()
    total_loss = correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        if training:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.item()) * len(y)
        correct += int((logits.argmax(1) == y).sum().item())
        total += len(y)
    return {"loss": total_loss / total, "accuracy": correct / total}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data")
    p.add_argument("--output", default="artifacts/checkpoints/mnist_scamp_software.pt")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-train", type=int)
    p.add_argument("--max-test", type=int)
    p.add_argument("--num-workers", type=int, default=0)
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = FirmwareMNIST(a.data_root, train=True)
    test_ds = FirmwareMNIST(a.data_root, train=False)
    if a.max_train:
        train_ds.base.data = train_ds.base.data[:a.max_train]
        train_ds.base.targets = train_ds.base.targets[:a.max_train]
    if a.max_test:
        test_ds.base.data = test_ds.base.data[:a.max_test]
        test_ds.base.targets = test_ds.base.targets[:a.max_test]
    train_loader = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True,
                               num_workers=a.num_workers)
    test_loader = DataLoader(test_ds, batch_size=a.batch_size, shuffle=False,
                              num_workers=a.num_workers)

    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=a.lr)
    history = []
    for epoch in range(1, a.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device)
        with torch.no_grad():
            test_metrics = run_epoch(model, test_loader, None, device)
        row = {"epoch": epoch, "train": train_metrics, "test": test_metrics}
        history.append(row)
        print(json.dumps(row))

    os.makedirs(os.path.dirname(a.output) or ".", exist_ok=True)
    torch.save(model.cpu().state_dict(), a.output)
    report = {
        "dataset": "MNIST",
        "input_preprocessing": "threshold, bbox centre, aspect-preserving resize to 45px, 64x64 canvas",
        "architecture": "pad -> Conv2d(1,16,4) -> ReLU -> MaxPool2d(4) -> Linear(4096,10)",
        "device": str(device),
        "epochs": a.epochs,
        "train_samples": len(train_ds),
        "test_samples": len(test_ds),
        "checkpoint": a.output,
        "history": history,
        "note": "PC software baseline; not SCAMP-5 hardware reproduction",
    }
    report_path = os.path.splitext(a.output)[0] + ".json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({"checkpoint": a.output, "report": report_path,
                      "final_test": history[-1]["test"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
