# SCAMP-5 MNIST CNN: PC functional reproduction

This is a CPU/software baseline for the public `Scamp5-MNIST_AREG_CNN_example` firmware. It preserves the advertised data path: MNIST -> 64x64 plane -> sixteen 4x4 filters -> ReLU -> 2x2 max pooling -> ten-way fully connected classification. It does **not** claim SCAMP latency, power, or FPS.

## Source audit and known limitation

The checked-in `original_repo/` copy includes the firmware and the exact
ternary header.  It shows that the demo's input is not a plain 28->64 resize:
`extract_character_from_F_into_R11` thresholds/flood-fills a foreground digit,
centres its bounding box, and uses `extracted_digit_img_size=45`; the network
then performs a 4x4 max pool (`maxpool=4`) and a 16x16-per-map FC layout.  The
Python crop/centre implementation is deterministic but remains an
approximation of the SCAMP digital scaler's DNEWS raster operations, so a
standard MNIST score is not automatically the advertised 95% hardware/demo
score.

## Files

`preprocessing.py` implements the firmware-shaped MNIST path: threshold/crop the
digit, preserve its aspect ratio while scaling the foreground to about 45 px,
centre it on a 64x64 plane, and save PGM snapshots. `weights.py` parses
C/C++ numeric initializers and supports ternary quantization. `scamp_ops.py`
implements add/skip/subtract ternary convolution and max pooling. `model.py`
is the shape-faithful software graph and accepts the original `(16,1,4,4)` and
`(10,16,16,16)` tensors directly. `evaluate.py` reports accuracy, macro
precision/recall/F1, confusion matrices, operation counts, and analytical data
movement hooks. `main.py` is a runnable smoke test.

`torch_model.py` supplies the ordinary PyTorch reference graph when PyTorch is installed: `Conv2d(1,16,4) -> ReLU -> MaxPool2d(4) -> Linear(16*15*15,10)`. The firmware calls `MAX_POOL_F(maxpool-1)` with `maxpool=4`, so pooling is a 4x4 reduction. It is intentionally not used by the dependency-free smoke test.

## Run

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
python -m baseline.main --sample-pgm artifacts/sample.pgm
```

No third-party packages are required for the smoke test. With the original header available, call `load_header(path, ternary=True)` and reshape the returned values to the declarations in that header; do not retrain or randomly replace those weights. A dataset adapter can feed decoded MNIST images into `mnist_to_64` and `ScampCNN.forward`.

For no-training evaluation of a supplied reference checkpoint, run `python -m baseline.evaluate_mnist --data-root data --weights checkpoint.pt --download`. This command does not create or claim an original SCAMP checkpoint.

## Reproduction status

The repository is now checked out under `original_repo/`. Source audit confirms: both convolution and FC weights are signed ternary `int8_t` values; FC storage is 10 neurons x 16 feature maps x 16x16 pooled coordinates; the firmware uses `maxpool=4` (three directional max passes plus blocking), and final prediction is an average of three normalized activation frames. Digital scaling is not ordinary interpolation: it uses a fixed 80-entry row/column order and DNEWS propagation, while digit extraction first thresholds/flood-fills, centers by bounding box, and scales to a target size (default 45). The Python implementation therefore exposes the exact ternary weights and 4x4 pooling shape, but its image extraction/scaling remains an algorithmic CPU approximation until hardware raster semantics are validated.

Convert the original header with `python -m baseline.import_original original_repo/WEIGHTS_MNIST_4x4_16CONVOLS_64x64INPUT_95ACC.hpp`. The old implementation resized the complete 28x28 canvas and therefore did not match the firmware's `extract_character_from_F_into_R11(...,45,...)` path; use the corrected `mnist_to_64` for dataset evaluation. Any timing numbers from this program are PC software measurements or analytical estimates only, never SCAMP-5 hardware measurements.
