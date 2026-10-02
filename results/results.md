| split | model | n | rmse | mae | pearson | spearman | n_train | n_skipped |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random | random forest | 11035 | 0.648 | 0.474 | 0.88 | 0.876 | 44141 | 0 |
| random | gnn | 11035 | 0.856 | 0.653 | 0.798 | 0.793 | 44141 | 0 |
| random | per-target mean | 11035 | 1.18 | 0.941 | 0.49 | 0.467 | 44141 | 0 |
| scaffold | random forest | 11035 | 0.78 | 0.58 | 0.817 | 0.808 | 44141 | 0 |
| scaffold | gnn | 11035 | 0.86 | 0.654 | 0.777 | 0.77 | 44141 | 0 |
| scaffold | per-target mean | 11035 | 1.18 | 0.947 | 0.474 | 0.455 | 44141 | 0 |
| scaffold_shuffled | random forest | 11035 | 0.761 | 0.565 | 0.827 | 0.825 | 44141 | 0 |
| scaffold_shuffled | gnn | 11035 | 0.974 | 0.759 | 0.707 | 0.704 | 44141 | 0 |
| scaffold_shuffled | per-target mean | 11035 | 1.186 | 0.949 | 0.473 | 0.447 | 44141 | 0 |
