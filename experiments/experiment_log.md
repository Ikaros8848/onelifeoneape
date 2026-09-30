# Experiment Log

| ID | Date | Method/config | Seed | Accuracy | Key proxy outcome | Status |
|---|---|---|---:|---:|---|---|
| E00 | 2026-09-30 | FP32 checkpoint reproduction | n/a | 97.08% | dense reference | retained |
| E01 | 2026-09-30 | Ternary QAT checkpoint reproduction | n/a | 97.36% | 900,945 active terms | retained baseline |
| E02 | 2026-09-30 | Naive post-training ternary | n/a | 63.97% | accuracy collapse | retained failure |
| E03 | 2026-09-30 | Soft gates then hard top-14 export, 5 epochs | 7 | 87.56% | proxy cost down, unusable accuracy | retained failure |
| E04 | 2026-09-30 | Hard STE top-14, 4 epochs | 7 | 97.46% | active -16.63%, energy proxy -15.63% | selected |
| E05 | 2026-09-30 | Hard STE top-15, 4 epochs | 7 | 97.38% | active -13.41%, energy proxy -12.58% | Pareto |
| E06 | 2026-09-30 | Hard STE top-13, 4 epochs | 7 | 97.35% | active -26.20%, energy proxy -24.49% | selected cost point |
| E07 | 2026-09-30 | Hard STE top-13, 4 epochs | 42 | 97.38% | seed confirmation | robustness |
| E08 | 2026-09-30 | Retrained fixed random 13-offset masks 101/202/303/404/505, 4 epochs from common QAT checkpoint; masks 101/202 repeated at training seed 42 | 7, 42 (partial repeats) | 96.94%-97.29% | random-mask variation; all k=13, 41,242 parameters; see `results/retrained_random13_ablation.csv` | retained comparison; no mask selected by accuracy |

Git baseline commit before new work: `0ea21674bfcdecfc2131db55db58bc90a539d61a`. Checkpoints and JSON reports preserve the exact command outputs.
