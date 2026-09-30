# Strict Results Audit: 13-tap / 14-tap

Audit date: 2026-09-30. This document audits the existing checkpoints and formulas without changing the core method.

## 1. What a tap means

In this project a tap is one spatial kernel offset `(i,j)` in the 4x4 convolution stencil, shared across all 16 output filters. It is not a pixel location, output feature-map position, channel, parameter, or generic operation slot. The baseline has 4x4 = 16 possible taps.

For an input plane `X` and filter `W[o,c,i,j]`, the baseline output at one position is:

```text
z[o,y,x] = bias[o] + sum(c=0..0, i=0..3, j=0..3)
                         W[o,c,i,j] * X[c,y+i,x+j]
```

The offset gate is shared across filters:

```text
p = sigmoid(offset_logits / T)
g = straight_through_top_k(p, k)
z[o,y,x] = bias[o] + sum(i,j where g[i,j]=1) W[o,0,i,j] * X[0,y+i,x+j]
```

The 13-tap model removes the offset groups `(1,1)`, `(2,1)`, and `(2,2)` from the scheduled stencil; the 14-tap model removes `(1,1)` and `(2,1)`. Every removed group would otherwise contribute up to 16 filter terms at each of 64x64 output positions. It does not remove whole output pixels or channels.

The training graph applies the hard top-k mask before the convolution. The exported deployment state bakes the mask into zero convolution weights. However, the benchmarked PyTorch inference still calls dense `F.conv2d`; therefore the logical sparse schedule is reduced, while the actual CPU/GPU dense kernel candidate count is not. The cost model is a deployment-schedule estimate, not a measured PyTorch kernel reduction.

## 2. Active-operation arithmetic

The cost model uses ternary signs after thresholding:

- `+1`: one positive-add term;
- `-1`: one negative-subtract term;
- `0`: one skipped candidate term.

For convolution, each non-zero or zero weight is expanded over 64x64 output positions. FC terms are counted once. Thus:

```text
N_active = (N_conv_plus + N_conv_minus) * 64 * 64
           + N_fc_plus + N_fc_minus
N_dense  = (16*1*4*4) * 64 * 64 + 10*4096 = 1,089,536
N_skipped = N_dense - N_active
```

Measured benchmark values:

| Model | Active terms | Skipped terms | Dense candidate terms |
|---|---:|---:|---:|
| Ternary QAT | 900,945 | 188,591 | 1,089,536 |
| 14-tap | 751,083 | 338,453 | 1,089,536 |
| 13-tap | 664,554 | 424,982 | 1,089,536 |

For 13-tap:

```text
(900,945 - 664,554) / 900,945
= 236,391 / 900,945
= 26.243% (rounded benchmark values)
```

The earlier headline 26.20% used a slightly different rounded/intermediate report; the exact current CSV values give 26.24%. This is a logical ternary schedule reduction, not a measured GPU runtime reduction.

For 14-tap:

```text
(900,945 - 751,083) / 900,945 = 16.633%
```

Training and benchmark sign accounting are structurally consistent: training forward applies top-k; benchmark loads the exported masked state and recounts non-zero signs. The PyTorch execution itself remains dense, so `software_inference_skip_realized=False` is now explicit in benchmark output.

## 3. What is really executed

The answer is split:

- Intended SCAMP deployment schedule: only the retained top-k offsets are scheduled, so the logical active count above is appropriate for that schedule abstraction.
- Current software inference: `F.conv2d` receives a dense 4x4 tensor containing zeros. It may still scan all 16 offsets internally. The actual software candidate count is therefore 1,089,536 for all models; no GPU/CPU speedup is claimed.

The benchmark now reports both `logical_deployment_active_terms` and `actual_software_candidate_terms`, plus `software_inference_skip_realized=False`.

## 4. Energy proxy formula and provenance

The implemented formula is:

```text
E_proxy = 1.00 * positive_terms
        + 1.00 * negative_terms
        + 0.50 * comparisons
        + 0.25 * local_movement_elements
        + 0.01 * external_movement_bits
        + 0.10 * register_accesses
        + 32.0 * control_stages
```

where:

```text
register_accesses = active_terms + comparisons + local_movement_elements
control_stages = retained_offsets + shortest_route_depth
                 + pool_schedule_depth + fc_mask_passes
```

The coefficients are dimensionless proxy weights. They are not SCAMP energy-per-operation measurements. Register capacities and local NEWS connectivity are literature facts; the decomposition, accounting widths, live-plane counts, and all numeric coefficients are software abstractions or manually fixed proxy assumptions. No coefficient is an analogue voltage, clock energy, joule, watt, or chip calibration.

## 5. Is energy just active operations in disguise?

It is strongly correlated because active terms appear directly in the add/sub terms and again inside `register_accesses`. That is a real double contribution in the current proxy and should not be interpreted as independent physical evidence.

For QAT to 13-tap, component reductions are:

| View | Reduction |
|---|---:|
| Active terms only | 26.24% |
| External + local movement combined | 3.02% |
| Latency proxy | 7.69% |
| Full energy proxy | 24.49% |

With each energy coefficient changed independently by -50%, -20%, baseline, +20%, +50%, the 13-tap energy improvement remains positive. The measured ranges are approximately:

| Coefficient | -50% | -20% | baseline | +20% | +50% |
|---|---:|---:|---:|---:|---:|
| add | 23.56% | 24.18% | 24.49% | 24.74% | 25.05% |
| subtract | 24.45% | 24.48% | 24.49% | 24.50% | 24.52% |
| compare | 24.85% | 24.63% | 24.49% | 24.35% | 24.14% |
| local move | 24.75% | 24.59% | 24.49% | 24.39% | 24.24% |
| external bit | 24.51% | 24.50% | 24.49% | 24.48% | 24.47% |
| register access | 24.59% | 24.53% | 24.49% | 24.45% | 24.40% |
| control stage | 24.50% | 24.50% | 24.49% | 24.49% | 24.48% |

This is coefficient sensitivity, not hardware calibration. A more important follow-up is ablation of entire energy components and removal of the active-term contribution from register accesses.

## 6. Movement proxy

```text
local_movement = shortest_route_depth * 4096 + comparisons
comparisons = 16 * 16 * 16 * (16 - 1) = 61,440
```

Current values:

| Model | Local movement | External bits | Combined movement |
|---|---:|---:|---:|
| QAT | 122,880 | 148,880 | 271,760 |
| 14-tap | 114,688 | 148,880 | 263,568 |
| 13-tap | 114,688 | 148,880 | 263,568 |

Thus 13-tap and 14-tap local movement reductions are 6.67%, while combined movement reduction is 3.02%. The external movement is unchanged because the current model still accounts for the full stored tensor footprint and does not model compressed schedule metadata.

## 7. Latency proxy

```text
L_proxy = retained_offsets
        + shortest_route_depth
        + pool_schedule_depth
        + fc_mask_passes
        + output_classes
        + spill_penalty
```

With current assumptions, spill penalty is zero and the values are:

| Model | Retained offsets | Route depth | Latency proxy |
|---|---:|---:|---:|
| QAT | 16 | 15 | 65 |
| 14-tap | 14 | 13 | 61 |
| 13-tap | 13 | 13 | 60 |

Reductions relative to QAT are 6.15% and 7.69%. These are normalized schedule-depth estimates, not SCAMP clock cycles or wall-clock runtime.

## 8. Parameters and gating cost

The baseline and exported deployment models each contain 41,242 parameters. The training-only gate has 16 floating-point logits. It is not included in the deployment state: the selected mask is statically baked into zero weights and the deployment schedule can store a fixed list of retained offsets. This means:

- dynamic gate computation is not paid at inference;
- training gate logits are correctly excluded from deployed parameter count;
- static schedule metadata and any compiler/control cost are not currently charged;
- the current active/latency/energy proxies therefore omit gate-generation cost, because deployment is static, but also omit a small fixed schedule metadata/control cost.

This omission is acceptable only if the deployment is explicitly described as static top-k scheduling. It would be invalid for dynamic per-input gating.

## 9. Seed results

The checked-in seed-7 and seed-42 13-tap deployment checkpoints both use the same fixed schedule budget. The current full-test benchmark results are:

| Seed | Accuracy | Macro F1 | Active terms | Energy proxy |
|---:|---:|---:|---:|---:|
| 7 | 97.35% | 97.335% | 664,554 | 811,103 |
| 42 | 97.38% | 97.362% | 664,668 | 811,228.4 |

The active/energy values are determined by the exported mask and ternary signs; accuracy/F1 must be benchmarked from the seed-42 deploy checkpoint before reporting it as a completed statistical result.

## 10. Audited comparison

| Method | Accuracy | Macro F1 | Params | Logical active ops | Movement combined | Latency proxy | Energy proxy |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP32 | 97.08% | 97.059% | 41,242 | 1,089,536 | 271,760 | 65 | 1,281,610.4 |
| Ternary QAT | 97.36% | 97.339% | 41,242 | 900,945 | 271,760 | 65 | 1,074,160 |
| 14-tap | 97.46% | 97.444% | 41,242 | 751,083 | 263,568 | 61 | 906,317 |
| 13-tap | 97.35% | 97.335% | 41,242 | 664,554 | 263,568 | 60 | 811,103 |

Relative to QAT, 14-tap changes are: accuracy +0.10 pp, F1 +0.105 pp, active -16.63%, movement -3.02% combined, latency -6.15%, energy -15.63%. 13-tap changes are: accuracy -0.01 pp, F1 -0.004 pp, active -26.24%, movement -3.02%, latency -7.69%, energy -24.49%.

## Final audit conclusions

1. The 26.24% active-operation reduction is mathematically reproducible as a logical sparse ternary schedule count. It is not a measured CPU/GPU execution reduction.
2. The 24.49% energy reduction is a valid output of the declared proxy formula, but it is strongly correlated with active terms and partly double-counts them through register accesses. It is not measured energy.
3. The current model has repeated active-term influence: direct add/sub terms plus register-access term. This is a methodological limitation that should be fixed or ablated before strong energy claims.
4. The metric was not intentionally redefined after seeing the result, but several coefficients and widths are manually selected proxy assumptions. They must remain labelled as such.
5. Safe for a defense: accuracy, F1, parameter counts, ternary sign counts, logical active/skipped terms, and explicitly normalized proxy values with formulas and caveats.
6. Every latency, movement, register-pressure, energy, and composite number requires `proxy`, `normalized`, or `simulation estimate`; none is SCAMP-5 measured hardware data.
7. The single most important missing experiment is a component ablation/reparameterization of the energy proxy: compare the current formula against an orthogonal version where register accesses exclude active arithmetic, plus a sparse-kernel or schedule-trace implementation that verifies actual executed tap instructions.
