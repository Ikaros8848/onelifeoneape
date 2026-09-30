# SCAMP-aware MNIST Baseline Audit

Audit date: 2026-09-30  
Git commit: 0ea21674bfcdecfc2131db55db58bc90a539d61a  
Scope: software measurement and analytical accounting only; no SCAMP-5 hardware measurement.

## Reproduction

Both checked-in checkpoints were evaluated on all 10,000 torchvision MNIST test samples with the project firmware-style preprocessing.

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| FP32 software CNN | 97.08% | not recorded | not recorded | not recorded |
| SCAMP-aware ternary QAT | 97.36% | 97.3470% | 97.3356% | 97.3393% |
| Naive post-training ternary | 63.97% | not recorded | not recorded | not recorded |

Checkpoint inference is reproducible. Training stability is not yet proven because the training scripts do not set deterministic seeds and no repeated full-training runs are retained.

## Preprocessing

The input path normalizes grayscale values, uses threshold 0.10 to find the foreground bounding box, preserves grayscale within the crop, resizes the longest side to about 45 pixels with nearest-neighbor interpolation, and centers the digit on a 64 by 64 canvas. This is a deterministic CPU approximation of the firmware morphology path, not an exact DNEWS or analogue-camera simulation.

## Model structure

| Stage | Operation | Output shape | Elements per sample |
|---|---|---:|---:|
| Input | firmware-style MNIST preprocessing | 1 x 64 x 64 | 4,096 |
| Padding | left 1, right 2, top 1, bottom 2 | 1 x 67 x 67 | 4,489 |
| Convolution | 16 filters, 4 x 4, stride 1 | 16 x 64 x 64 | 65,536 |
| ReLU | local activation | 16 x 64 x 64 | 65,536 |
| Max pool | 4 x 4, stride 4 | 16 x 16 x 16 | 4,096 |
| Flatten | reshape | 4,096 | 4,096 |
| Classifier | linear 4096 to 10 | 10 | 10 |

## Parameters

| Layer | Weights | Biases | Total |
|---|---:|---:|---:|
| Convolution | 256 | 16 | 272 |
| Fully connected | 40,960 | 10 | 40,970 |
| Total | 41,216 | 26 | 41,242 |

The FC layer contains 99.38% of weight parameters, while convolution dominates repeated arithmetic.

## Ternary weights

The QAT checkpoint uses threshold ratio 0.05 and one scale per output channel.

| Layer | Positive | Zero | Negative | Zero fraction | Mean scale |
|---|---:|---:|---:|---:|---:|
| Conv | 128 | 39 | 89 | 15.234% | 0.34547 |
| FC | 5,404 | 28,847 | 6,709 | 70.427% | 0.11971 |
| All weights | 5,532 | 28,886 | 6,798 | 70.085% | not comparable across layers |

The 70.085% global zero-weight fraction is not a 70% compute saving because convolution weights are reused at every spatial output location.

## Algorithmic operation counts

These are scalar analytical terms for the current graph, not SCAMP instruction or cycle counts.

| Layer | Positive adds | Negative subtracts | Zero skips | Active terms | Dense terms |
|---|---:|---:|---:|---:|---:|
| Conv | 524,288 | 364,544 | 159,744 | 888,832 | 1,048,576 |
| FC | 5,404 | 6,709 | 28,847 | 12,113 | 40,960 |
| Total | 529,692 | 371,253 | 188,591 | 900,945 | 1,089,536 |

The expanded skip fraction is 17.309%. Convolution contributes 98.656% of all active terms. The current PyTorch simulator still launches dense positive and negative kernels, so software runtime does not realize these analytical skips.

## Activation distribution

| Tensor | Min | Max | Mean | Std | Zero fraction |
|---|---:|---:|---:|---:|---:|
| Input | 0.0000 | 1.0000 | about 0.117 | about 0.292 | about 82.6% |
| Conv pre-ReLU | -5.9869 | 5.2691 | 0.00870 | 0.47931 | about 0% |
| Conv post-ReLU | 0.0000 | 5.2691 | 0.14374 | 0.30190 | 51.043% |
| Pooled features | 0.0000 | 5.2691 | 0.26164 | 0.50151 | 42.543% |
| Logits | -29.1702 | 16.8376 | about -4.13 | about 6.31 | about 0% |

The high ReLU sparsity is a candidate for active-region or feature-routing optimization, but SIMD instructions cannot be counted as skipped unless an entire scheduled region or instruction is removed.

## Current inference flow

1. Apply crop, scale, and center preprocessing.
2. Pad to 67 by 67.
3. Quantize latent FP32 weights to per-channel scaled ternary signs.
4. Accumulate positive convolution terms and negative convolution terms separately.
5. Apply optional accumulator clipping, bias, ReLU, and optional activation clipping.
6. Apply 4 by 4 max pooling.
7. Quantize FC weights and execute positive and negative linear paths.
8. Add bias and select the maximum logit.

For the reproduced 97.36% result, activation_limit and accumulator_limit are both None. Finite-range clipping exists only as an unused interface in this baseline.

## What the software currently represents

- Ternary signs minus one, zero, and plus one.
- Add, subtract, and analytical zero skip semantics.
- Per-output-channel scale.
- A 64 by 64 working plane.
- ReLU and 4 by 4 pooling.
- Optional activation and accumulator clipping interfaces.
- Firmware-inspired input crop, centering, and scaling.

## What it does not simulate

- The physical 256 by 256 pixel-processor array.
- Analogue voltage, charge, noise, mismatch, drift, or precision loss.
- Exact analogue and digital register allocation or conflicts.
- Instruction-level timing, scheduling, and kernel depth.
- NEWS and DNEWS direction-dependent data-movement cost.
- Global reduction and readout timing.
- Real chip latency, FPS, energy, or power.
- Camera exposure and acquisition behavior.

All later latency and energy outputs must be labeled proxy, normalized metric, or simulation estimate.

## Original firmware mapping facts

The original C++ project maps sixteen 64 by 64 filter regions onto a four by four partition of the 256 by 256 PPA. The 4 by 4 convolution therefore has sixteen sequential kernel-offset stages, while pixels and mapped filters operate in parallel. Pooling uses directional NEWS movement and analogue comparisons. The classifier uses positive and negative masks followed by sparse global sums across classes and quadrants.

Therefore the hardware cost model must keep two separate views:

1. Algorithmic scalar work: positive adds, negative subtracts, and zero skips.
2. PPA instruction-depth proxy: retained kernel offsets, mask setup, NEWS or DNEWS shifts, pooling passes, sparse global sums, readout, and register-pressure penalties.

## Confirmed issues in existing accounting

1. baseline/evaluate.py defaults to a 61 by 61 convolution output, but the actual padded graph produces 64 by 64. Its 952,576 convolution-term result is incorrect for this checkpoint.
2. The same estimator reports no negative subtraction terms.
3. The existing movement estimator uses inconsistent 61 and 30 spatial sizes, a single two-bit width for unrelated tensors, and omits FC weights, pooled features, reads and writes, masks, and reuse.
4. Existing training scripts do not record seeds, deterministic settings, optimizer state, RNG state, git commit, or repeated trials.
5. The consolidated historical report rounds old runs to 97.09% and 97.37%, while the current checkpoints reproduce 97.08% and 97.36%.
6. scripts/baseline/run_inference.py instantiates zero weights by default and is only a smoke test.

## Main bottlenecks

- Compute and sequential-depth bottleneck: convolution, with 98.656% of active scalar terms and sixteen sequential kernel-offset stages in the firmware mapping.
- Parameter and loading bottleneck: FC, with 99.38% of weights and 70.427% zeros.
- Feature-volume bottleneck: the 16 x 64 x 64 convolution and ReLU maps.
- Register-pressure uncertainty: the simulator has no valid physical register model, so only normalized scenario-based pressure may be reported.

## Baseline conclusion

The 97.36% ternary QAT checkpoint is a valid and reproducible software baseline and slightly exceeds the 97.08% FP32 checkpoint. It is not a SCAMP-5 hardware result. The next stage should replace the current inaccurate estimates with a dual-view cost model that separately measures scalar work and PPA instruction depth, then adds explicit movement and normalized storage-pressure proxies.
