# Results

All task metrics are software measurements on the 10,000-image torchvision MNIST test split using the repository preprocessing. Hardware columns are analytical normalized proxies, not SCAMP-5 measurements.

## Main comparison

| Method | Accuracy | Macro F1 | Params | Active terms | Local movement | Latency proxy | Energy proxy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Float CNN | 97.08% | 97.059% | 41,242 | 1,089,536 | 122,880 | 65.0 | 1,292,801 |
| Ternary QAT | 97.36% | 97.339% | 41,242 | 900,945 | 122,880 | 65.0 | 1,074,160 |
| Proposed offset-14 | **97.46%** | **97.444%** | 41,242 | 751,083 | 114,688 | 61.0 | 906,317 |
| Proposed offset-13 | 97.35% | 97.335% | 41,242 | 664,554 | 114,688 | 60.0 | 811,103 |

Relative to ternary QAT, offset-14 improves active terms by 16.63%, local movement by 6.67%, latency proxy by 6.15%, and energy proxy by 15.63%, with +0.10 percentage point accuracy. Offset-13 improves active terms by 26.20%, latency proxy by 7.69%, and energy proxy by 24.49%, with -0.01 percentage point accuracy.

The balanced composite proxy is 0.917 for offset-14 and 0.877 for offset-13 when QAT is normalized to 1.0. These are dimensionless relative scores.

## Pareto and sensitivity

`results/accuracy_cost.csv` contains the 13/14/15-tap Pareto sweep. `hardware_model/config.py` defines balanced, compute-heavy, movement-heavy, schedule-heavy, and storage-heavy profiles. The proposed models improve every profile because the reductions are present in both scalar activity and sequential schedule terms; the absolute composite values differ by profile.

## Reproducibility and limitations

Seed-7 and seed-42 13-tap runs measured 97.35% and 97.38%, respectively. This is evidence of short-run stability, not a claim of exhaustive statistical convergence. No real SCAMP-5 board, instruction trace, analogue register capture, clock counter, or power meter was used.
