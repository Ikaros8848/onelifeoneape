# YSYX 工程说明

本仓库提供整理后的工程代码

本项目包含三条独立的软件实验线：SCAMP-5/MNIST 基线、二值 FCN 论文基线和 SegNet 对照基线；`original_repo/` 保留原始 SCAMP-5 C++ 参考工程。

## 目录

- `baseline/`：SCAMP 风格操作、模型、预处理、量化和指标核心代码。
- `paper_baseline/`：二值 FCN 模型及合成/真实数据适配器。
- `segnet_baseline/`：SegNet 模型。
- `scripts/`：训练、评估、推理入口。
- `tests/`：快速检查和测试入口。
- `experiments/`：消融等实验脚本。
- `original_repo/`：原始 C++ 工程和固件权重头文件。
- `data/`：本地数据集，不纳入源码版本。
- `references/`：论文和提取文本。
- `artifacts/reports/统一实验记录.md`：整理前实验结果汇总。

## 运行

### 从 GitHub 克隆并运行

```powershell
git clone https://github.com/Ikaros8848/onelifeoneape.git
cd YSYX
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m scripts.baseline.run_inference
.venv\Scripts\python.exe -m scripts.paper_baseline.train --epochs 3 --samples 512
.venv\Scripts\python.exe -m scripts.segnet_baseline.train --epochs 3 --samples 512
.venv\Scripts\python.exe -m tests.sanity_hardware
```

需要 PyTorch 的训练和评估脚本使用 `requirements.txt` 中的依赖。完整实验输出默认写入 `artifacts/checkpoints/` 和 `artifacts/reports/`；数据路径默认是 `data/`。

## 数据集放置指引

仓库不包含数据集。运行他人的代码前，请在项目根目录创建 `data/`，并按照下面的目录结构放置数据。

### MNIST

MNIST 使用 torchvision 的目录格式，推荐放置为：

```text
YSYX/
└── data/
    └── MNIST/
        └── raw/
            ├── train-images-idx3-ubyte
            ├── train-labels-idx1-ubyte
            ├── t10k-images-idx3-ubyte
            └── t10k-labels-idx1-ubyte
```

如果手头是 `.gz` 压缩包，可以直接放入 `data/MNIST/raw/`，torchvision 通常会在首次读取时完成解压；也可以先手动解压。训练脚本默认不联网下载，因此训练前应先准备好 MNIST：

```powershell
python -m scripts.baseline.train_mnist --data-root data
python -m scripts.baseline.train_ternary_qat --data-root data
```

评估脚本支持自动下载时，可显式添加 `--download`，例如：

```powershell
python -m scripts.baseline.evaluate_mnist --data-root data --download
```

### Oxford-IIIT Pet

论文基线的真实数据适配器使用 torchvision 的 Oxford-IIIT Pet，目录应为：

```text
YSYX/
└── data/
    └── oxford-iiit-pet/
        ├── images/
        └── annotations/
            └── trimaps/
```

首次运行可以让 torchvision 自动下载和整理数据：

```powershell
python -m scripts.paper_baseline.train `
  --dataset oxford_pet --download --device cpu `
  --epochs 3 --samples 100 --data-root data
```

如果数据放在其他位置，使用 `--data-root` 指定其父目录，例如 `--data-root D:/datasets`。代码会在该目录下查找 `MNIST/` 或 `oxford-iiit-pet/`。

### 合成数据集

`synthetic` 和 `webots_like` 是代码即时生成的确定性合成数据，不需要准备任何文件：

```powershell
python -m scripts.paper_baseline.train --dataset synthetic --epochs 3 --samples 512
python -m scripts.paper_baseline.train --dataset webots_like --epochs 3 --samples 512
python -m scripts.segnet_baseline.train --epochs 3 --samples 512
```

### 数据检查

如果出现 `Dataset not found` 或文件读取错误，请优先检查：

1. 命令是否在仓库根目录执行。
2. `--data-root` 是否指向包含 `MNIST/` 或 `oxford-iiit-pet/` 的父目录。
3. 数据集目录大小写和连字符是否正确。
4. 是否误把压缩包放在了 `data/` 根目录，而不是 torchvision 需要的子目录。

## 主要入口

```powershell
# SCAMP 风格软件冒烟推理
python -m scripts.baseline.run_inference

# 普通 MNIST 基线训练
python -m scripts.baseline.train_mnist --data-root data

# SCAMP 感知三值 QAT
python -m scripts.baseline.train_ternary_qat --data-root data

# 二值 FCN 合成数据实验
python -m scripts.paper_baseline.train --dataset synthetic --epochs 3 --samples 512

# SegNet 合成数据实验
python -m scripts.segnet_baseline.train --epochs 3 --samples 512
```

完整训练通常需要 GPU 和较长时间；首次运行建议先使用较小的 `--epochs`、`--samples` 或 `--max-train` 做冒烟验证。




## 验证

```powershell
python -m compileall -q baseline paper_baseline segnet_baseline scripts tests experiments
python -m tests.sanity_hardware
```

实验结果汇总见 [`artifacts/reports/统一实验记录.md`](artifacts/reports/统一实验记录.md)。
