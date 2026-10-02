| split | model | target | n | rmse | mae | pearson | spearman |
| --- | --- | --- | --- | --- | --- | --- | --- |
| random | gnn | ACHE | 1123 | 1.011 | 0.755 | 0.67 | 0.665 |
| random | gnn | BACE1 | 1952 | 0.834 | 0.657 | 0.739 | 0.742 |
| random | gnn | DRD2 | 1709 | 0.901 | 0.691 | 0.632 | 0.627 |
| random | gnn | EGFR | 1980 | 0.932 | 0.717 | 0.726 | 0.732 |
| random | gnn | HERG | 1921 | 0.741 | 0.536 | 0.622 | 0.568 |
| random | gnn | JAK2 | 2350 | 0.776 | 0.615 | 0.817 | 0.81 |
| random | per-target mean | ACHE | 1123 | 1.325 | 1.062 | nan | nan |
| random | per-target mean | BACE1 | 1952 | 1.199 | 0.986 | nan | nan |
| random | per-target mean | DRD2 | 1709 | 0.997 | 0.792 | nan | nan |
| random | per-target mean | EGFR | 1980 | 1.31 | 1.082 | nan | nan |
| random | per-target mean | HERG | 1921 | 0.907 | 0.673 | nan | nan |
| random | per-target mean | JAK2 | 2350 | 1.289 | 1.052 | nan | nan |
| random | random forest | ACHE | 1123 | 0.714 | 0.534 | 0.851 | 0.813 |
| random | random forest | BACE1 | 1952 | 0.712 | 0.527 | 0.807 | 0.81 |
| random | random forest | DRD2 | 1709 | 0.613 | 0.45 | 0.797 | 0.799 |
| random | random forest | EGFR | 1980 | 0.709 | 0.528 | 0.845 | 0.85 |
| random | random forest | HERG | 1921 | 0.562 | 0.409 | 0.791 | 0.738 |
| random | random forest | JAK2 | 2350 | 0.596 | 0.429 | 0.888 | 0.886 |
| scaffold | gnn | ACHE | 1262 | 1.03 | 0.796 | 0.652 | 0.646 |
| scaffold | gnn | BACE1 | 1846 | 0.849 | 0.667 | 0.747 | 0.744 |
| scaffold | gnn | DRD2 | 1816 | 0.837 | 0.63 | 0.637 | 0.645 |
| scaffold | gnn | EGFR | 1764 | 1.002 | 0.766 | 0.658 | 0.658 |
| scaffold | gnn | HERG | 2257 | 0.693 | 0.511 | 0.622 | 0.535 |
| scaffold | gnn | JAK2 | 2090 | 0.809 | 0.636 | 0.805 | 0.804 |
| scaffold | per-target mean | ACHE | 1262 | 1.311 | 1.097 | nan | nan |
| scaffold | per-target mean | BACE1 | 1846 | 1.297 | 1.073 | nan | nan |
| scaffold | per-target mean | DRD2 | 1816 | 1.049 | 0.842 | nan | nan |
| scaffold | per-target mean | EGFR | 1764 | 1.269 | 1.032 | nan | nan |
| scaffold | per-target mean | HERG | 2257 | 0.862 | 0.659 | nan | nan |
| scaffold | per-target mean | JAK2 | 2090 | 1.309 | 1.076 | nan | nan |
| scaffold | random forest | ACHE | 1262 | 0.902 | 0.703 | 0.734 | 0.725 |
| scaffold | random forest | BACE1 | 1846 | 0.856 | 0.657 | 0.767 | 0.769 |
| scaffold | random forest | DRD2 | 1816 | 0.728 | 0.547 | 0.731 | 0.724 |
| scaffold | random forest | EGFR | 1764 | 0.867 | 0.654 | 0.733 | 0.737 |
| scaffold | random forest | HERG | 2257 | 0.646 | 0.479 | 0.669 | 0.608 |
| scaffold | random forest | JAK2 | 2090 | 0.724 | 0.513 | 0.841 | 0.841 |
| scaffold_shuffled | gnn | ACHE | 1204 | 1.177 | 0.899 | 0.537 | 0.52 |
| scaffold_shuffled | gnn | BACE1 | 2005 | 0.995 | 0.818 | 0.681 | 0.681 |
| scaffold_shuffled | gnn | DRD2 | 1529 | 0.878 | 0.677 | 0.531 | 0.532 |
| scaffold_shuffled | gnn | EGFR | 2024 | 0.976 | 0.769 | 0.61 | 0.608 |
| scaffold_shuffled | gnn | HERG | 1815 | 0.84 | 0.61 | 0.386 | 0.354 |
| scaffold_shuffled | gnn | JAK2 | 2458 | 0.996 | 0.796 | 0.675 | 0.65 |
| scaffold_shuffled | per-target mean | ACHE | 1204 | 1.382 | 1.132 | nan | nan |
| scaffold_shuffled | per-target mean | BACE1 | 2005 | 1.261 | 1.051 | nan | nan |
| scaffold_shuffled | per-target mean | DRD2 | 1529 | 0.98 | 0.779 | nan | nan |
| scaffold_shuffled | per-target mean | EGFR | 2024 | 1.238 | 1.014 | nan | nan |
| scaffold_shuffled | per-target mean | HERG | 1815 | 0.856 | 0.652 | nan | nan |
| scaffold_shuffled | per-target mean | JAK2 | 2458 | 1.297 | 1.048 | nan | nan |
| scaffold_shuffled | random forest | ACHE | 1204 | 0.96 | 0.704 | 0.732 | 0.741 |
| scaffold_shuffled | random forest | BACE1 | 2005 | 0.794 | 0.619 | 0.788 | 0.79 |
| scaffold_shuffled | random forest | DRD2 | 1529 | 0.673 | 0.502 | 0.737 | 0.733 |
| scaffold_shuffled | random forest | EGFR | 2024 | 0.816 | 0.637 | 0.758 | 0.755 |
| scaffold_shuffled | random forest | HERG | 1815 | 0.621 | 0.449 | 0.699 | 0.655 |
| scaffold_shuffled | random forest | JAK2 | 2458 | 0.721 | 0.519 | 0.831 | 0.825 |
