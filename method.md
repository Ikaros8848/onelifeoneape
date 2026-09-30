# Offset-Group SCAMP-Aware Optimization

## Motivation

The ternary QAT checkpoint already removes many scalar terms, but its sixteen 4x4 kernel offsets remain in the sequential convolution schedule. The original SCAMP-oriented firmware executes these offsets as tap stages, so unstructured zero weights do not imply a shorter schedule. The core method therefore learns one gate per spatial kernel offset, shared by all 16 convolution filters.

## Mathematical definition

Let `S[o,c,y,x] in {-1,0,+1}` be the ternary convolution sign and `g[y,x] in {0,1}` a shared offset gate. The deployed convolution is:

`z[o,y,x] = alpha[o] * sum_{c,i,j} g[i,j] S[o,c,i,j] X[c,y+i,x+j]`.

During training, `p = sigmoid(a / T)` and a straight-through top-k operator produces exactly `K` forward-active offsets. The objective is:

`L = CE(f_theta(X), y) + lambda_g sum p + lambda_b (sum p - K)^2`.

The reported runs use `K in {13,14,15}`, `T=1`, `lambda_g=0.001`, and `lambda_b=0.01`. The exported deploy checkpoint bakes the hard mask into the latent weights and contains only the original four tensors (`conv_weight`, `conv_bias`, `fc_weight`, `fc_bias`).

## Why this is SCAMP-aware

- A whole zero offset can remove one sequential stencil stage; isolated zero weights cannot.
- The cost model counts the retained offset count and shortest NEWS-style Manhattan route separately from scalar additions/subtractions.
- The method does not claim analogue voltage, instruction timing, power, or chip FPS.

## Accuracy constraint

The primary selection rule is `accuracy >= 0.9736 - delta`, with delta tested at 0.1 percentage point. The 13-tap model reaches 97.35% (seed 7) and 97.38% (seed 42), satisfying this threshold; the 14-tap model reaches 97.46%.

## Implementation

- Training model: `optimized/offset_gated.py`
- Training script: `scripts/optimized/train_offset_gated.py`
- Deployment export: `export_deploy_state`
- Unified benchmark: `benchmark.py`
- Cost model: `hardware_model/cost_model.py`

The 13-tap configuration is the main cost-oriented candidate. The 14-tap checkpoint is the accuracy-oriented candidate. Both are retained for Pareto analysis.
