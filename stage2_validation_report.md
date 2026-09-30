# Stage 2 Validation Report

Date: 2026-09-30  
Scope: validate the existing static 13-tap/14-tap method. No new model structure or optimization module was added.

## 1. Seed-42 complete validation

`results/seed_validation.csv` now contains full 10,000-sample test-set results for QAT, seed-7 13-tap, seed-42 13-tap, and 14-tap. The seed-42 deployment checkpoint is fully validated:

| Model | Accuracy | Macro F1 | Logical active ops | Movement local | Latency proxy | Energy A | Energy B |
|---|---:|---:|---:|---:|---:|---:|---:|
| QAT | 97.36% | 97.3393% | 900,945 | 122,880 | 65 | 1,074,160.3 | 984,065.8 |
| 13-tap seed 7 | 97.35% | 97.3353% | 664,554 | 114,688 | 60 | 811,103.0 | 744,647.6 |
| 13-tap seed 42 | 97.38% | 97.3615% | 664,668 | 114,688 | 60 | 811,228.4 | 744,761.6 |
| 14-tap | 97.46% | 97.4440% | 751,083 | 114,688 | 61 | 906,316.9 | 831,208.6 |

Confusion matrices for every row are stored in the `confusion_matrix_json` column of `results/seed_validation.csv`.

## 2. Static schedule definition

The method is **static top-k scheduling**, not dynamic input-dependent gating. Training learns 16 offset logits and a preferred subset. Deployment freezes the selected subset and bakes it into the convolution weights.

The 13-tap schedule retains 13 offsets and removes `(1,1)`, `(2,1)`, `(2,2)`. The 14-tap schedule removes `(1,1)`, `(2,1)`. Complete records are in `results/stage2_schedule_trace.json`.

For each retained offset, the trace records positive, negative, zero ternary weights and expanded operation counts over all `64x64` output positions. Example 13-tap structure:

```text
baseline: (0,0), (0,1), ..., (3,3)             # 16 offset stages
13-tap:   (0,0), (0,1), (0,2), (0,3),
          (1,0),       (1,2), (1,3),
          (2,0),              (2,3),
          (3,0), (3,1), (3,2), (3,3)             # 13 stages
```

The simulator is `hardware_model/sparse_schedule_simulator.py`. It executes only selected offsets in `sparse_reference_conv` and does not call a full 4x4 convolution kernel.

## 3. Sparse reference correctness

Dense masked convolution and explicit sparse reference convolution were compared on all 10,000 test images for the seed-7 13-tap deployment weights:

| Check | Result |
|---|---:|
| Full-test max absolute feature error | 2.8610e-06 |
| Full-test mean absolute feature error | 1.2180e-08 |
| Classification prediction agreement | 100% |

The tiny floating-point difference comes from accumulation order: dense convolution uses a backend kernel, while sparse reference adds one offset at a time with `einsum`. It is numerically negligible and does not change predictions.

## 4. Logical operation reduction

The explicit schedule trace reproduces the logical count:

```text
QAT active = 900,945
13-tap seed 7 active = 664,554
(900,945 - 664,554) / 900,945 = 26.24%
```

This is now executable as a transparent offset schedule. It remains a logical schedule count, not a measured GPU kernel reduction. `results/seed_validation.csv` explicitly reports `software_inference_skip_realized=False` because the ordinary PyTorch benchmark still uses dense kernels.

## 5. CPU reference runtime

For a 16-image CPU batch, the measured reference runtimes were:

| Path | Seconds per call |
|---|---:|
| Dense masked `F.conv2d` | 0.000506 |
| Python/PyTorch sparse offset reference | 0.001369 |

The sparse reference was slower by about 2.7x. This is expected interpreter/kernel-launch overhead and is not evidence against a hardware schedule. It is not SCAMP latency and is not used as an efficiency claim.

## 6. Energy proxy correction

Proxy A preserves the historical formula:

```text
arithmetic + comparison + movement +
0.10 * (active_terms + comparisons + local_movement_elements) + control
```

Proxy B removes active arithmetic from register accesses:

```text
arithmetic + comparison + movement +
0.10 * (comparisons + local_movement_elements) + control
```

All coefficients remain dimensionless manually selected proxy weights. Neither proxy is measured energy.

`results/energy_ablation.csv` reports four views:

| Model | A: arithmetic only | B: operation + movement | C: + orthogonal register | D: full current proxy | B orthogonal total |
|---|---:|---:|---:|---:|---:|
| QAT | 900,945.0 | 963,873.8 | 982,305.8 | 1,074,160.3 | 984,065.8 |
| 13-tap seed 7 | 664,554.0 | 725,434.8 | 743,047.6 | 811,103.0 | 744,647.6 |
| 13-tap seed 42 | 664,668.0 | 725,548.8 | 743,161.6 | 811,228.4 | 744,761.6 |
| 14-tap | 751,083.0 | 811,963.8 | 829,576.6 | 906,316.9 | 831,208.6 |

Relative to QAT for seed-7 13-tap:

| Model | Reduction |
|---|---:|
| A arithmetic only | 26.24% |
| B operation + movement | 24.74% |
| C + orthogonal register | 24.36% |
| D historical full proxy | 24.49% |
| Orthogonal total B | 24.33% |

The reduction remains positive across these models, but the energy result is not independent of arithmetic activity. Proxy B is the more methodologically defensible version for future reporting because it avoids active-term double counting in register accesses.

## 7. Measurement boundary

### Measured on PC

- Full-test accuracy, macro precision, macro recall, macro F1
- Confusion matrices
- Parameter count
- CPU reference runtime for dense masked and sparse reference functions
- Dense-vs-sparse numerical error and prediction agreement

### Logical simulation

- Retained offset schedule
- Per-tap operation count
- Logical active/skipped operation count
- Schedule length

### Hardware proxy

- Local/external movement counts
- Register pressure
- Normalized latency proxy
- Energy Proxy A/B
- Composite normalized cost

No SCAMP-5 board, instruction trace, clock counter, analogue register capture, power meter, or FPGA implementation was used.

## 8. Direct answers

1. The 13-tap schedule is explicitly executable in the independent simulator.
2. Sparse reference and dense masked inference agree to numerical tolerance with 100% prediction agreement over all 10,000 images.
3. Logical active-operation reduction remains 26.24% for seed 7; seed 42 gives 664,668 active terms, a nearly identical reduction.
4. Removing active-term double counting gives 24.33% orthogonal total-energy reduction for seed 7, versus 24.49% for the historical proxy.
5. 13-tap accuracy is 97.35% (seed 7) and 97.38% (seed 42).
6. Seed 42 is now fully reproduced, including F1, confusion matrix, operations, movement, latency, and both energy proxies.
7. The schedule is static; no runtime gating is needed or charged.
8. Accuracy/F1/parameters/runtime/error/agreement are PC measurements.
9. Schedule and operation counts are logical simulation.
10. Movement, latency, and energy are hardware proxies.

## 9. Remaining limitation

The most important unresolved validation is an actual sparse low-level execution trace or SCAMP/FPGA implementation. The current Python sparse reference verifies mathematical schedule correctness, but its runtime is slower than dense `F.conv2d` and must not be used as a hardware efficiency result.
