# Final Evidence Summary

## Task performance

Clean full-test results from the Stage 4 run are: QAT 97.36% accuracy / 97.3393% macro F1; learned 13-tap seed 7 97.35% / 97.3353%; learned 13-tap seed 42 97.38% / 97.3615%. The detailed robustness records are in `results/robustness.csv`.

## Efficiency

The established logical/proxy results remain unchanged: QAT has 900,945 logical active operations; learned 13-tap has 664,554 (seed 7) and 664,668 (seed 42). Movement, latency, and Energy Proxy B are analytical hardware proxies, not physical measurements.

## Correctness

Stage 2 dense/sparse schedule agreement remains documented in `stage2_validation_report.md`. Stage 4 uses the same full 10,000-image test split and existing firmware-style preprocessing.

## Ablation and reproducibility

Tap sweep, frozen-mask, retrained-random-mask, and gate-contribution results are documented in the Stage 3 reports. Stage 3B records exact masks, seeds, checkpoints, and fairness checks. The Stage 4 script is `scripts/stage4_robustness.py`; noise seed is 1729.

## Robustness

Gaussian noise was added after `mnist_to_64` preprocessing to the normalized 64x64 model input and clipped to [0, 1]. Sigmas were fixed at 0.05, 0.10, and 0.20. Translation used zero-filled shifts of 1 and 2 pixels in all four directions. No retraining was performed. QAT and both learned checkpoints received the same generated corruption tensors.

At sigma 0.05, accuracies were QAT 84.78%, 13-tap seed 7 86.65%, and seed 42 89.87%; at sigma 0.10 they were 51.36%, 44.68%, and 52.73%; at sigma 0.20 they were 14.20%, 11.88%, and 12.78%. Translation degradation was generally comparable, with direction-dependent differences. The results do not establish a uniform robustness advantage: the 13-tap models are better at the lightest Gaussian level, while results vary at stronger noise and across translations.

## Evidence classification

### PC measurement

- Accuracy, macro F1, confusion matrices, and parameter counts.
- CPU/GPU software execution and the fixed-input robustness evaluation.

### Logical simulation

- Selected offsets, schedule trace, active/skipped operation counts.

### Hardware proxy

- Movement proxy, latency proxy, and Energy Proxy B.

## Hardware limitation

No physical SCAMP-5 measurement, FPGA implementation, measured power, measured hardware latency, or hardware FPS is available. Proxy values must retain the `proxy` or `simulation estimate` qualifier.
