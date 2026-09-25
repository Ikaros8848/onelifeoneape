# YSYX 工程说明

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

在项目根目录执行：

```powershell
.venv\Scripts\python.exe -m scripts.baseline.run_inference
.venv\Scripts\python.exe -m scripts.paper_baseline.train --epochs 3 --samples 512
.venv\Scripts\python.exe -m scripts.segnet_baseline.train --epochs 3 --samples 512
.venv\Scripts\python.exe -m tests.sanity_hardware
```

需要 PyTorch 的训练和评估脚本使用 `requirements.txt` 中的依赖。完整实验输出默认写入 `artifacts/checkpoints/` 和 `artifacts/reports/`；数据路径默认是 `data/`。
