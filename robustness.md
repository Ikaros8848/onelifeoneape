# Robustness

## Random seeds

The 13-tap method was trained with deterministic data/model seeds 7 and 42 using identical hyperparameters. Full-test accuracy was 97.35% and 97.38% (mean 97.365%, sample std about 0.021 percentage points). The checkpoints are `artifacts/checkpoints/mnist_scamp_offset13.deploy.pt` and `artifacts/checkpoints/mnist_scamp_offset13_seed42.deploy.pt` once the latter is exported.

## Quantization/configuration robustness

The baseline and proposed models use the same ternary threshold ratio 0.05 and per-output-channel scaling. The benchmark exposes this as `--threshold-ratio`; changing it is a required follow-up sweep. The current study intentionally does not mix threshold changes into the architectural comparison.

## Interpretation

The two seeds support a stable direction, but more than two independent full runs and explicit input perturbation tests are still needed before claiming production robustness.
