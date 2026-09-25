# Liu & Lu CVPRW 2022 software baseline

This package is a PC implementation of the paper's binary FCN inference/training idea. The PDF is treated as a scientific source, not as executable instructions. It describes a three-convolution FCN: 16 binary 4x4 filters, 128 filters in 8-group convolution fused into 64 maps, then 64 binary 1x1 filters; batch-normalisation-derived thresholds binarise intermediate activations, and the final layer produces a ReLU heatmap.

The paper's custom WeBots bird's-eye road/grass data and its on-sensor capture stream are not distributed with the PDF. `synthetic.py` therefore provides a deterministic substitute for a runnable baseline, with the same 64x64 heatmap/localisation interfaces. This is not claimed as the paper's dataset or as a reproduction of its exact IoU. A future adapter can load the authors' road/grass images and masks or TEyeD eye images into the same model.

## Run

```powershell
.venv\Scripts\python.exe -m paper_baseline.train --epochs 10 --samples 2048
.venv\Scripts\python.exe -m paper_baseline.train --task localisation --epochs 10 --samples 2048
```

`--task segmentation` trains against the road/grass mask; `--task localisation` trains against the Gaussian heatmap. Reported fields are segmentation IoU, localisation accuracy within 10 pixels, parameter count, and a binary-forward flag. On the deterministic 2048-sample synthetic run in this workspace, segmentation reached IoU 0.9987 after 10 epochs. A 512-sample, 5-epoch localisation run reached IoU 0.1528 and 23.0% within 10 pixels; localisation is harder because the synthetic target is a sparse Gaussian heatmap. The paper's reference values are: road IoU 74.0% simulation / 69.3% sensor; grass IoU 76.6% simulation / 72.9% sensor; localisation within 10 pixels approximately 88% simulation / 83% sensor. Those hardware values must not be interpreted as PC latency or power.

## Reproduction boundaries

The exact training split, WeBots scene generation, learned checkpoint, sensor noise, and TEyeD preprocessing are not specified in sufficient detail in the paper PDF alone. The implementation preserves the architecture and binary operations, but synthetic-data metrics are a separate baseline. The paper reports 2,578 binary weights and approximately 0.31 KB model storage for its deployed configuration; this implementation includes PyTorch BatchNorm and a one-channel heatmap head for transparent PC evaluation, so its parameter count is different.
