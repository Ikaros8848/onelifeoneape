# YSYX 工程说明

本仓库是可直接在 GitHub Codespaces 中打开的工程代码仓库。数据集、虚拟环境、模型权重和临时样例均不上传到 GitHub。

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

### GitHub Codespaces

在 GitHub 仓库页面选择 **Code → Codespaces → Create codespace on master**。容器创建后会自动安装 `requirements.txt` 中的 Python 依赖。

本项目默认从项目根目录下的 `data/` 读取数据。由于数据集不随仓库上传，需要在 Codespace 中自行准备数据，或运行脚本时传入其他数据目录。

### 本地环境

在项目根目录执行：

```powershell
.venv\Scripts\python.exe -m scripts.baseline.run_inference
.venv\Scripts\python.exe -m scripts.paper_baseline.train --epochs 3 --samples 512
.venv\Scripts\python.exe -m scripts.segnet_baseline.train --epochs 3 --samples 512
.venv\Scripts\python.exe -m tests.sanity_hardware
```

需要 PyTorch 的训练和评估脚本使用 `requirements.txt` 中的依赖。完整实验输出默认写入 `artifacts/checkpoints/` 和 `artifacts/reports/`；数据路径默认是 `data/`。

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

完整训练通常需要 GPU 和较长时间；Codespaces 中建议先使用较小的 `--epochs`、`--samples` 或 `--max-train` 做冒烟验证。

## 数据与模型文件

以下内容明确排除在 Git 版本库之外：

- `data/`：MNIST、Oxford-IIIT Pet 等数据集
- `.venv/`：本地 Python 虚拟环境
- `artifacts/checkpoints/`：训练产生的模型权重
- `artifacts/samples/`：PGM 等样例输出
- `*.pt`、`*.pth`、`*.ckpt`、`*.npz`：模型和导出的权重文件

原始 SCAMP-5 C++ 参考代码位于 `original_repo/`，其中的固件权重头文件属于源代码参考的一部分，会随仓库保留。

## 验证

```powershell
python -m compileall -q baseline paper_baseline segnet_baseline scripts tests experiments
python -m tests.sanity_hardware
```

实验结果汇总见 [`artifacts/reports/统一实验记录.md`](artifacts/reports/统一实验记录.md)。
