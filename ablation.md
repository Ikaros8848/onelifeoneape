# Ablation

| Variant | Accuracy | Retained offsets | Interpretation |
|---|---:|---:|---|
| QAT baseline | 97.36% | 16 | No schedule sparsity |
| Offset-15 | 97.38% | 15 | One tap removed |
| Offset-14 | 97.46% | 14 | Selected accuracy-oriented point |
| Offset-13 | 97.35% | 13 | Selected cost-oriented point |
| Soft-gate post-training export | 87.56% | 14 | Failure: training did not see the hard schedule |
| Naive post-training ternary | 63.97% | n/a | Failure: confirms QAT is required |

The soft-gate failure is retained in `artifacts/checkpoints/mnist_scamp_offset_soft_failed.*`. It motivated straight-through top-k gating during training.
