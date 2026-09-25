"""Test the trained SCAMP-shaped MNIST software checkpoint.

Examples:
    python -m baseline.test_mnist
    python -m baseline.test_mnist --max-samples 1000
    python -m baseline.test_mnist --image path/to/digit.png
"""
import argparse
import json

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets

from baseline.preprocessing import mnist_to_64
from baseline.torch_model import build_model
from baseline.hardware_aware import HardwareAwareCNN


def image_to_tensor(image):
    """Apply the same firmware-style preprocessing used during training."""
    image = image.convert("L")
    raw = list(image.tobytes())
    pixels = [raw[y * image.width:(y + 1) * image.width]
              for y in range(image.height)]
    processed = mnist_to_64(pixels)
    return torch.tensor(processed, dtype=torch.float32).unsqueeze(0)


class TestMNIST:
    def __init__(self, root, max_samples=None):
        self.base = datasets.MNIST(root, train=False, download=False)
        self.length = min(len(self.base), max_samples or len(self.base))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image, label = self.base[index]
        return image_to_tensor(image), int(label)


def test_dataset(model, root, batch_size, max_samples, device):
    dataset = TestMNIST(root, max_samples=max_samples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    confusion = torch.zeros((10, 10), dtype=torch.int64)
    correct = total = 0

    model.eval()
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images).argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            total += labels.numel()
            for expected, predicted in zip(labels.cpu(), predictions.cpu()):
                confusion[int(expected), int(predicted)] += 1

    return {
        "samples": total,
        "correct": correct,
        "accuracy": correct / total,
        "confusion_matrix": confusion.tolist(),
    }


def test_image(model, path, device):
    image = Image.open(path)
    tensor = image_to_tensor(image).unsqueeze(0).to(device)
    model.eval()
    with torch.inference_mode():
        probabilities = model(tensor).softmax(dim=1)[0].cpu()
    prediction = int(probabilities.argmax().item())
    return {
        "image": path,
        "prediction": prediction,
        "confidence": float(probabilities[prediction].item()),
        "probabilities": [float(value) for value in probabilities],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",
                        default="artifacts/checkpoints/mnist_scamp_software.pt")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--image", help="Test one handwritten-digit image")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU inference")
    args = parser.parse_args()

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if "conv_weight" in state:
        model = HardwareAwareCNN(quantize=True)
    else:
        model = build_model()
    model.load_state_dict(state)
    model.to(device)

    if args.image:
        result = test_image(model, args.image, device)
    else:
        result = test_dataset(
            model, args.data_root, args.batch_size, args.max_samples, device
        )
    result["checkpoint"] = args.checkpoint
    result["device"] = str(device)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
