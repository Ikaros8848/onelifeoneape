# Stage 3B: Retrained Random 13-Offset Ablation

Date: 2026-09-30. This experiment adds only fixed-mask training runs; it does not change model architecture, the learned 13-tap method, the test set, or cost-model coefficients.

## 1. Question

Under the same 13-offset budget and QAT initialization/training recipe, how do preselected random static offset masks compare with the existing learned static mask?

## 2. Protocol and fairness checks

Five random masks were generated before training using mask seeds 101, 202, 303, 404, and 505. They were not selected or rejected based on accuracy. All five were trained with training seed 7. Masks 101 and 202 also received a predeclared repeat with training seed 42 to estimate some training-seed variation.

Every run:

- starts from `artifacts/checkpoints/mnist_scamp_qat.pt`;
- uses the same 60,000-image MNIST training set and 10,000-image test set;
- uses the existing `FirmwareMNIST` / `mnist_to_64` preprocessing;
- trains 4 epochs, batch size 128, Adam, base LR `1e-3`;
- uses ternary threshold ratio `0.05`, per-channel scaling, `gate_lambda=0.001`, and `budget_lambda=0.01`;
- has no data augmentation;
- exports a static mask and fixed deployment checkpoint.

In the random-mask runs, the 16 gate logits carry the predetermined binary mask, are frozen, and are excluded from the optimizer. `gate_lr=2e-3` is recorded for parity but is inapplicable because there are no trainable gate parameters. The gate regularizers are constants with respect to trainable weights, so the weight-gradient objective is the same classification objective under the fixed mask. The learned runs retain trainable gate logits; learning the subset is the factor under test.

Automated checks in `results/retrained_random13/fairness_checks.json` passed: all masks have k=13, all deployment models have 41,242 parameters, all full test evaluations use 10,000 samples, and the common training configuration fields match. Inference used the same preprocessing and evaluation definition. PyTorch still uses dense `F.conv2d`; operation counts are logical schedule counts, not measured PC sparse execution.

## 3. Results

Accuracy/F1 and cost metrics were re-evaluated from every deployment checkpoint on the full test set. Complete records are in `results/retrained_random13_ablation.csv`; confusion matrices are in `results/retrained_random13/confusion_matrices.json`.

| Method | Mask seed | Training seed | Accuracy | Macro Precision | Macro Recall | Macro F1 | Params | Active Ops | Skipped Ops | Local Movement | Latency Proxy | Energy Proxy B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Learned 13-tap | learned | 7 | 97.35% | 97.3396% | 97.3336% | 97.3353% | 41,242 | 664,554 | 424,982 | 114,688 | 60 | 744,647.6 |
| Learned 13-tap | learned | 42 | 97.38% | 97.3635% | 97.3697% | 97.3615% | 41,242 | 664,668 | 424,868 | 114,688 | 60 | 744,761.6 |
| Random 101 | 101 | 7 | 97.27% | 97.2553% | 97.2475% | 97.2494% | 41,242 | 652,390 | 437,146 | 110,592 | 59 | 731,018.0 |
| Random 101 | 101 | 42 | 97.29% | 97.2694% | 97.2717% | 97.2689% | 41,242 | 644,154 | 445,382 | 110,592 | 59 | 722,782.0 |
| Random 202 | 202 | 7 | 97.10% | 97.0796% | 97.0833% | 97.0764% | 41,242 | 668,845 | 420,691 | 114,688 | 60 | 748,938.6 |
| Random 202 | 202 | 42 | 97.02% | 97.0076% | 96.9964% | 96.9983% | 41,242 | 664,750 | 424,786 | 114,688 | 60 | 744,843.6 |
| Random 303 | 303 | 7 | 97.07% | 97.0561% | 97.0547% | 97.0519% | 41,242 | 656,942 | 432,594 | 122,880 | 62 | 739,966.8 |
| Random 404 | 404 | 7 | 97.21% | 97.1941% | 97.1871% | 97.1890% | 41,242 | 672,682 | 416,854 | 122,880 | 62 | 755,706.8 |
| Random 505 | 505 | 7 | 96.94% | 96.9413% | 96.9241% | 96.9283% | 41,242 | 681,211 | 408,325 | 114,688 | 60 | 761,304.6 |

Energy Proxy B, movement, and latency columns are unchanged analytical proxies; the coefficients were not modified in Stage 3B.

## 4. Mask and training-seed variation

The exact offset lists are stored in every `experiment.json` and in the `selected_offsets` CSV column. As compact spatial summaries, the three omitted offsets are:

| Configuration | Omitted offsets |
|---|---|
| Learned 13-tap, seed 7 | `(1,1), (2,1), (2,2)` |
| Learned 13-tap, seed 42 | `(1,1), (2,1), (2,2)` |
| Random mask 101 | `(1,0), (2,2), (3,0)` |
| Random mask 202 | `(0,1), (1,0), (1,1)` |
| Random mask 303 | `(0,2), (1,3), (2,0)` |
| Random mask 404 | `(1,3), (3,1), (3,3)` |
| Random mask 505 | `(0,1), (0,2), (1,2)` |

For the two learned runs, mean accuracy is 97.365% with sample standard deviation 0.021 percentage points; mean macro F1 is 97.3484% with sample standard deviation 0.0186 pp. These are two training seeds for one learned mask, not two mask samples.

Across the five distinct random mask configurations at training seed 7, accuracy mean is 97.118%, sample standard deviation 0.128 pp, and range 96.94%–97.27%. Macro F1 mean is 97.0990%, sample standard deviation 0.125 pp. These five observations estimate mask variation at one training seed; they must not be described as five independent training seeds.

The training-seed repeats show mask 101 at 97.27% (seed 7) and 97.29% (seed 42), and mask 202 at 97.10% and 97.02%, respectively. These two pairs are too few to characterize training-seed variance generally.

## 5. Cost and performance comparison

The learned-mask mean has higher accuracy than the mean of the five random masks trained with seed 7 by 0.247 pp. Both learned seed results exceed every random mask's seed-7 result in this tested set (random maximum 97.27%, learned minimum 97.35%). This is an observed comparison, not a significance test.

Costs overlap rather than uniformly favoring learned masks. Random mask 101 has lower active operations (652,390) and Energy Proxy B (731,018.0) than learned seed 7 (664,554 and 744,647.6), at 0.08 pp lower accuracy. Random 505 has both the largest active count and largest Energy Proxy B among random masks, and the lowest accuracy. Equal k therefore does not imply equal ternary active operations or equal movement/route proxy.

The learned mask was identical across seeds 7 and 42. Random masks vary spatially as expected from their independent predeclared samples. No causal claim is made from the visual shape of any mask.

## 6. Interpretation

Under the tested conditions, the learned static offset configuration achieved higher classification performance than the five randomly selected configurations, while retaining the same 13-offset schedule budget. The random masks still reached 96.94%–97.27%, so much of the task performance is compatible with the 13-offset budget, but the tested learned configuration performed better than those five seed-7 random-mask runs. The learned-mask SD is measured across two training seeds for one mask, whereas the random-mask SD is measured across five mask configurations at one training seed; these SDs do not establish that either method is more stable than the other.

This does not prove that learned selection is globally optimal or superior to all random subsets. The random-mask sample is small, and only two masks were repeated across a second training seed. The mask seed-7 mean versus learned two-seed mean also mixes mask-configuration variation with different numbers of training-seed observations. No significance test is justified by this design.

## 7. Evidence classification

### PC Measurement

- Accuracy, macro precision, macro recall, macro F1 on 10,000 test images.
- Parameter count and confusion matrices.
- PC inference used PyTorch dense `F.conv2d`; these runs do not demonstrate PC sparse-kernel speedup.

### Logical Simulation

- Exact fixed selected-offset lists and k=13 schedule length.
- Ternary logical active/skipped operation counts.

### Hardware Proxy

- Local movement, latency proxy, Energy Proxy B.
- These are not SCAMP-5 measurements, clock cycles, power, energy, or FPS.

## 8. Reproducibility artifacts and limitations

- Training command implementation: `scripts/optimized/train_fixed_mask.py`.
- Full-test aggregation and fairness checks: `scripts/aggregate_stage3b.py`.
- Each run folder under `results/retrained_random13/random{mask_seed}_seed{training_seed}/` contains `checkpoint.pt`, `deploy.pt`, and `experiment.json`; seed-7 folders also contain `mask.json`.
- Existing learned checkpoints were reused without retraining.
- The next stronger statistical check would be more independently generated masks, each trained with multiple training seeds. This is a scope/compute extension, not evidence currently available.
