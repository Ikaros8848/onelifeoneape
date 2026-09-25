"""MNIST input conversion for the SCAMP-5 example.

The firmware does *not* resize the complete 28x28 MNIST canvas to 64x64.  It
first thresholds/flood-fills the digit, centres its bounding box, scales the
foreground to roughly 45 pixels, and only then places that digit on the 64x64
working plane.  Stretching the whole canvas (the old implementation) makes
the digit much too small and is the main source of the misleading baseline
results.

The SCAMP digital scaler is a binary morphology primitive, so this is a
deterministic CPU approximation: nearest-neighbour crop/resize with preserved
aspect ratio and centred placement.  It keeps the data convention explicit
without pretending to reproduce analogue hardware interpolation exactly.
"""
from typing import List, Sequence

def resize_nearest(image: Sequence[Sequence[float]], out_h: int = 64, out_w: int = 64) -> List[List[float]]:
    h, w = len(image), len(image[0])
    return [[float(image[min(h-1, int(y*h/out_h))][min(w-1, int(x*w/out_w))]) for x in range(out_w)] for y in range(out_h)]

def _normalise(image: Sequence[Sequence[float]]) -> List[List[float]]:
    out = [[float(v) for v in row] for row in image]
    mx = max((v for row in out for v in row), default=1.0)
    if mx > 1.0:
        out = [[v / 255.0 for v in row] for row in out]
    return out

def mnist_to_64(image: Sequence[Sequence[float]], threshold=0.10,
                digit_size: int = 45, canvas_size: int = 64) -> List[List[float]]:
    """Convert one MNIST image using the firmware's crop/centre convention.

    ``threshold`` is applied only to find the foreground bounding box; the
    resized crop retains greyscale values for the analogue convolution.
    Empty images return an all-zero canvas.
    """
    src = _normalise(image)
    h, w = len(src), len(src[0]) if src else 0
    if not h or not w:
        return [[0.0] * canvas_size for _ in range(canvas_size)]
    ys = [y for y in range(h) for x in range(w) if src[y][x] >= threshold]
    xs = [x for y in range(h) for x in range(w) if src[y][x] >= threshold]
    if not xs:
        return [[0.0] * canvas_size for _ in range(canvas_size)]
    x0, x1, y0, y1 = min(xs), max(xs) + 1, min(ys), max(ys) + 1
    crop = [row[x0:x1] for row in src[y0:y1]]
    ch, cw = len(crop), len(crop[0])
    scale = min(digit_size / max(ch, 1), digit_size / max(cw, 1))
    oh, ow = max(1, round(ch * scale)), max(1, round(cw * scale))
    scaled = resize_nearest(crop, oh, ow)
    out = [[0.0] * canvas_size for _ in range(canvas_size)]
    top = (canvas_size - oh) // 2
    left = (canvas_size - ow) // 2
    for yy, row in enumerate(scaled):
        for xx, value in enumerate(row):
            if 0 <= top + yy < canvas_size and 0 <= left + xx < canvas_size:
                out[top + yy][left + xx] = value
    return out

def save_pgm(image, path):
    h, w = len(image), len(image[0]); vals = [max(0,min(255,int(round(v*255)))) for r in image for v in r]
    with open(path, 'w', encoding='ascii') as f:
        f.write(f'P2\n{w} {h}\n255\n'); f.write(' '.join(map(str, vals)))
