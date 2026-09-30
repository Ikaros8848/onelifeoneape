# Stage 3A: Ablation, Tap Sweep, and Pareto Validation

Date: 2026-09-30. No model structure, optimization module, cost-model coefficient, dataset, or accuracy definition was changed in this stage. Existing baseline checkpoints remain intact.

## 1. Experimental questions

This stage tests (1) how accuracy/cost changes as the retained kernel-offset budget changes, (2) whether the learned fixed subset is important for the trained checkpoint, and (3) how much of the observed behavior depends on weight adaptation under the selected mask.

The complete per-model record is `results/tap_sweep.csv`; mask-only comparisons are also in `results/random13_ablation.csv`, `results/random13_qat_ablation.csv`, and `results/gate_contribution.csv`.

## 2. Fixed baselines and training configuration

- FP32 checkpoint: fixed historical baseline, 97.08%; historical training seed/config metadata is unavailable.
- Ternary QAT checkpoint: fixed baseline, 97.36%; historical training seed/config metadata is unavailable.
- New gated sweep controls k=16 through 10: seed 7, 4 epochs, batch 128, Adam, base learning rate `1e-3`, gate learning rate `2e-3`, threshold ratio `0.05`, per-channel scale, `gate_lambda=0.001`, `budget_lambda=0.01`, initialized from the QAT checkpoint, no data augmentation.
- k=13 seed 42 repeats the same configuration with seed 42.
- The k=16 gated control is separate from the original QAT baseline. At k=16, no offset is removed, but fine-tuning changes the ternary weight distribution; it is included to expose this training effect.
- Random-mask trials apply masks at inference to fixed weights and are not random-mask retraining experiments. The mask seed and exact selected offsets are recorded per row.

All benchmarked task metrics use the full 10,000-image MNIST test split and the existing firmware-style preprocessing.

## 3. Tap sweep

| Method | k | Accuracy | Macro Precision | Macro Recall | Macro F1 | Params | Logical Active Ops | Logical Skipped Ops | Local Movement | Latency Proxy | Energy Proxy B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FP32 | 16 | 97.08% | 97.088% | 97.068% | 97.059% | 41,242 | 1,089,536 | 0 | 122,880 | 65 | 1,172,656.8 |
| Ternary QAT baseline | 16 | 97.36% | 97.347% | 97.336% | 97.339% | 41,242 | 900,945 | 188,591 | 122,880 | 65 | 984,065.8 |
| Gated 16, seed 7 | 16 | 97.36% | 97.346% | 97.347% | 97.345% | 41,242 | 816,598 | 272,938 | 122,880 | 65 | 899,718.8 |
| Gated 15, seed 7 | 15 | 97.38% | 97.370% | 97.361% | 97.363% | 41,242 | 779,539 | 309,997 | 118,784 | 63 | 861,162.2 |
| Gated 14, seed 7 | 14 | 97.46% | 97.442% | 97.450% | 97.444% | 41,242 | 751,083 | 338,453 | 114,688 | 61 | 831,208.6 |
| Gated 13, seed 7 | 13 | 97.35% | 97.340% | 97.334% | 97.335% | 41,242 | 664,554 | 424,982 | 114,688 | 60 | 744,647.6 |
| Gated 13, seed 42 | 13 | 97.38% | 97.363% | 97.370% | 97.362% | 41,242 | 664,668 | 424,868 | 114,688 | 60 | 744,761.6 |
| Gated 12, seed 7 | 12 | 97.29% | 97.276% | 97.263% | 97.268% | 41,242 | 660,553 | 428,983 | 106,496 | 57 | 737,683.4 |
| Gated 11, seed 7 | 11 | 97.25% | 97.239% | 97.226% | 97.230% | 41,242 | 603,107 | 486,429 | 106,496 | 56 | 680,205.4 |
| Gated 10, seed 7 | 10 | 97.12% | 97.092% | 97.099% | 97.093% | 41,242 | 565,998 | 523,538 | 102,400 | 54 | 641,598.8 |

The main QAT-to-13-tap seed-7 logical active-op reduction remains 26.24%. Against the more controlled same-mechanism gated-16 run, the reduction is 18.62%. The difference matters: continued QAT under the gated training flow changes weight sparsity even when all 16 offsets are retained.

Accuracy is not strictly monotonic in k: k=14 is higher than k=15, while k=12, 11, and 10 trend downward. From the QAT baseline, the predefined 0.10 percentage-point tolerance corresponds to 97.26%; k=12 is 97.29%, k=11 is 97.25% (0.01 pp below that line), and k=10 is 97.12% (0.14 pp below). With only one run for most budgets, these are observed points, not confidence intervals.

Active operations generally decrease with k, but not in direct proportion to tap count. In particular, reducing k from 13 to 12 removes one tap but active operations fall by only 4,001; independently trained ternary weights and their zero distributions change with each budget. In the seed-7 gated sweep, Pearson correlation is approximately 0.989 between k and active terms. Energy Proxy B is almost a linear transformation of active terms for this sweep (Pearson correlation approximately 0.99995), so those two cost columns are not independent evidence.

## 4. Random and manual offset ablations

All masks below contain exactly 13 offsets. Two weight contexts are kept separate:

1. `random_fixed_mask`: random masks applied post hoc to the seed-7 learned-offset checkpoint's underlying ternary weights (the mask is changed, weights are held fixed).
2. `random_mask_on_qat`: the same five masks applied to the original QAT ternary weights without retraining.

| Mask experiment | Weight checkpoint | Accuracy | Macro F1 | Active Ops | Local Movement | Latency Proxy | Energy Proxy B |
|---|---|---:|---:|---:|---:|---:|---:|
| Learned 13 mask, fully trained | learned 13 seed 7 | 97.35% | 97.335% | 664,554 | 114,688 | 60 | 744,647.6 |
| Random 1 | learned 13 latent weights | 79.71% | 79.873% | 664,554 | 118,784 | 61 | 746,113.2 |
| Random 2 | learned 13 latent weights | 83.89% | 84.204% | 668,650 | 110,592 | 59 | 747,278.0 |
| Random 3 | learned 13 latent weights | 89.03% | 88.919% | 656,362 | 118,784 | 61 | 737,921.2 |
| Random 4 | learned 13 latent weights | 81.95% | 83.318% | 648,170 | 114,688 | 60 | 728,263.6 |
| Random 5 | learned 13 latent weights | 95.90% | 95.869% | 664,554 | 118,784 | 61 | 746,113.2 |

The post-hoc random-mask mean accuracy is 86.10% (sample standard deviation 6.47 pp); mean macro F1 is 86.44%. All five are below the fully trained learned-mask result, but this comparison uses weights adapted to the learned mask and is therefore expected to penalize a changed mask.

On original QAT weights, the five random masks have accuracy mean 82.25% (sample standard deviation 12.06 pp), range 69.38%–95.23%. The learned mask applied to QAT weights without gated fine-tuning scores 90.86% (macro F1 91.016%). One random mask is better than this learned mask, so these samples do not prove that the learned mask alone dominates arbitrary masks on frozen QAT weights.

A simple manual rule that drops the three corners `(0,0)`, `(0,3)`, `(3,0)` was also tested on original QAT weights: accuracy 92.89%, macro F1 92.912%. It is a transparent rule, not claimed as an optimized baseline.

## 5. Gate contribution ablation

| Condition | Accuracy | Interpretation |
|---|---:|---|
| A. Ternary QAT | 97.36% | Full 16-offset QAT baseline |
| B. QAT + fixed random 13-mask | mean 82.25%, range 69.38%–95.23% | Direct mask application, no weight adaptation |
| B2. QAT + manual corner-drop 13-mask | 92.89% | Fixed rule, no weight adaptation |
| C. QAT + learned 13-mask, no gated fine-tuning | 90.86% | Learned subset alone does not preserve accuracy on frozen QAT weights |
| D. Full learned 13-tap training | 97.35% (seed 7), 97.38% (seed 42) | Training under the hard retained-offset schedule recovers task performance |

This is not a fully crossed factorial experiment: random masks were not individually retrained with the same schedule. Therefore it supports the value of joint mask/weight adaptation relative to the tested post-hoc masks, but does not isolate a causal advantage of the learned subset over every possible retrained random subset.

## 6. Pareto analysis

Generated artifacts:

- `results/pareto_accuracy_active_ops.csv`
- `results/pareto_accuracy_energy.csv`
- `figures/accuracy_vs_active_ops.png`
- `figures/accuracy_vs_energy_proxy.png`

Both plots contain FP32, QAT, the same-mechanism gated k=16 control, k=15 through k=10 models, and both k=13 seeds. The Pareto x-axis metrics are logical active operations and Energy Proxy B, not measured hardware execution or energy.

Across k=16 to k=10, observed accuracy falls from 97.36% to 97.12%, while active operations fall from 816,598 to 565,998 and Energy Proxy B falls from 899,718.8 to 641,598.8. This is an observed accuracy-cost trade-off, with the largest tested budget reduction crossing the 0.10 pp accuracy tolerance. The original QAT baseline is separately shown for reference.

## 7. Offset stability and spatial pattern

Seed-7 and seed-42 k=13 runs selected exactly the same offsets:

```text
retained: (0,0) (0,1) (0,2) (0,3)
          (1,0)       (1,2) (1,3)
          (2,0)              (2,3)
          (3,0) (3,1) (3,2) (3,3)
removed:  (1,1), (2,1), (2,2)
```

This is exact support stability across two seeds, but two seeds are a small sample. Lower-k masks are not simply nested versions of this support; e.g. the 12-tap run retains `(2,2)` and removes other positions. The 13-tap pattern's absent offsets form a compact central cluster, but no causal geometric interpretation is established.

## 8. Failed/negative experiments retained

- Random fixed 13-offset masks: large and variable accuracy drops; none of the five masks on learned weights matches the trained learned-mask result.
- Learned mask directly applied to QAT weights without offset-gated fine-tuning: 90.86%, a substantial drop.
- Manual corner-drop mask on QAT weights: 92.89%, also below baseline.
- k=10: accuracy 97.12%, 0.24 pp below QAT and outside the previously used 0.10 pp tolerance.
- The Stage-2 soft-gate export failure and epoch-5 overfit remain recorded in the existing experiment log/checkpoints.

No failed point was removed from the raw result files.

## 9. Limitations and objective contribution

The evidence supports the narrower statement that a statically selected offset subset, when trained jointly with the ternary weights under the hard schedule, can retain approximately baseline accuracy at fewer logical active terms. The two k=13 seeds select identical supports and reproduce 97.35%/97.38%.

The evidence does not establish that the learned subset alone preserves accuracy on frozen QAT weights; it does not establish superiority to retrained random masks because that factorial comparison was not run; and it does not show CPU/GPU acceleration from sparse execution. Energy Proxy B is highly correlated with active operation count and remains a dimensionless analytical proxy. No SCAMP-5 hardware validation has been performed.

## 10. Evidence Classification

### PC Measurement

- Accuracy, macro precision, macro recall, macro F1 on the full MNIST test split.
- Confusion matrices in `results/seed_validation.csv` for Stage-2 fixed checkpoints.
- Parameter count.
- The Stage-2 sparse-reference numerical agreement and CPU runtime. The Stage-3 models use the same PC inference framework and preprocessing; no sparse kernel speedup is claimed here.

### Logical Simulation

- Selected offset sets and schedule length.
- Per-checkpoint logical active/skipped operation counts.
- The operation-count relation to retained offsets.

### Hardware Proxy

- Local movement, normalized latency proxy, and Energy Proxy B.
- These are not measured movement, SCAMP clock cycles, chip energy, power, or FPS.
