# Reproduction

Use the project environment (`.venv`) and run commands from `E:\YSYX`.

```powershell
.venv\Scripts\python.exe benchmark.py `
  --model fp32:fp32:artifacts/checkpoints/mnist_scamp_software.pt `
  --model qat:qat:artifacts/checkpoints/mnist_scamp_qat.pt `
  --model proposed:qat:artifacts/checkpoints/mnist_scamp_offset_gated.deploy.pt `
  --reference qat --output results/comparison.csv --json-output results/benchmark.json
```

Train a new offset-gated model without modifying baseline checkpoints:

```powershell
.venv\Scripts\python.exe -m scripts.optimized.train_offset_gated `
  --epochs 4 --target-retained-offsets 13 --seed 7 `
  --output artifacts/checkpoints/mnist_scamp_offset13.pt
```

The script writes both a training checkpoint and a `.deploy.pt` checkpoint. The latter is the model to benchmark. All proxy definitions and assumptions are in `scamp_hardware_assumptions.md` and `hardware_model/`.
