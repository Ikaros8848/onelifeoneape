"""Evaluate the *firmware* ternary weights on MNIST with matching preprocessing.

This is intentionally separate from ``evaluate_mnist.py``: that module tests a
new floating-point PyTorch graph, while this script runs the checked-in
SCAMP-5 header weights and the crop/centre/45px input path used by
``MAIN_MNIST_SINGLE_LAYER_16.cpp``.
"""
import argparse, json
import numpy as np
from baseline.model import ScampCNN
from baseline.preprocessing import mnist_to_64
from baseline.weights import load_original_header

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', default='data')
    p.add_argument('--download', action='store_true')
    p.add_argument('--max-samples', type=int, default=1000)
    p.add_argument('--weights-header', default='original_repo/WEIGHTS_MNIST_4x4_16CONVOLS_64x64INPUT_95ACC.hpp')
    a = p.parse_args()
    from torchvision import datasets
    ds = datasets.MNIST(a.data_root, train=False, download=a.download)
    conv, fc = load_original_header(a.weights_header)
    model = ScampCNN(np.asarray(conv, dtype=np.int8).reshape(16, 1, 4, 4),
                     np.asarray(fc, dtype=np.int8).reshape(10, 16, 16, 16))
    n = min(len(ds), a.max_samples) if a.max_samples > 0 else len(ds)
    correct = 0
    for i in range(n):
        image, label = ds[i]
        # PIL -> nested greyscale list; do not use torchvision Resize here.
        x = mnist_to_64(np.asarray(image, dtype=np.float32) / 255.0)
        pred = model.forward(x)['pred']
        correct += int(pred == int(label))
    result = {
        'dataset': 'MNIST test split (torchvision)',
        'samples': n,
        'accuracy': correct / n if n else 0.0,
        'preprocessing': 'threshold foreground, bbox centre, aspect-preserving resize to 45px, 64x64 canvas',
        'weights': a.weights_header,
        'note': 'CPU functional reproduction; not SCAMP-5 hardware. Exact score may differ because firmware uses analogue raster/scaler semantics.'
    }
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
