# dr.active

Predicts protein-ligand binding affinity (pChEMBL, i.e. -log10 of IC50/Ki/Kd in
molar units) from a **SMILES string** and a **target name**.

Two models, same data and same splits:

1. **Random forest** on RDKit physicochemical descriptors + Morgan fingerprint
   bits + a one-hot target indicator (scikit-learn).
2. **Experimental GNN**: a small edge-conditioned message-passing network with a
   GRU update and a learned target embedding, written in plain PyTorch.

Both are compared against a deliberately trivial baseline — *predict each
target's mean training pChEMBL* — on a random split **and** a Bemis-Murcko
scaffold split. The measured numbers are in [Results](#results); they are what
the code produced, including the part where the experimental GNN loses to the
much simpler random forest.

This is a prototype, not a tool for making decisions about real chemistry. See
[Limitations](#limitations).

## Install

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11 or 3.12.

```bash
git clone https://github.com/k8rthik/dr.active.git
cd dr.active
uv sync
uv run pytest            # 191 tests, no network, no downloads (94% statement coverage)
```

## Quick start

```bash
# 1. download raw activities from ChEMBL (85,472 records, 30 MB, ~18 min)
uv run python scripts/download_chembl.py

# 2. clean them into data/processed/affinity.csv
uv run python scripts/prepare_dataset.py

# 3. train and evaluate on all three splits (~75 min for rf+gnn)
uv run dr-active evaluate --models rf,gnn

# 4. train a model you want to keep, then predict with it
uv run dr-active train-rf
uv run dr-active predict "CC(=O)Oc1ccccc1C(=O)O" --target EGFR
```

`predict` output:

```
$ uv run dr-active predict "CC(=O)Oc1ccccc1C(=O)O" --target EGFR
target:            EGFR
model:             rf
predicted pChEMBL: 5.02
  approx.          9,471 nM
  tree spread      +/- 0.77 (disagreement, not a CI)
```

`--json` emits the same thing machine-readably. Invalid input is rejected at the
boundary with exit code 2:

```bash
$ uv run dr-active predict "not-a-molecule" --target EGFR
error: RDKit could not parse SMILES: 'not-a-molecule'

$ uv run dr-active predict "CCO" --target TP53
error: Unknown target 'TP53'. Known targets: EGFR, JAK2, BACE1, DRD2, HERG, ACHE.
```

## Commands

| command | what it does |
| --- | --- |
| `dr-active targets` | list the six modelled targets and their ChEMBL ids |
| `dr-active data-summary` | per-target row counts and pChEMBL distribution |
| `dr-active train-rf` | train the forest, save it, print top feature importances |
| `dr-active train-gnn` | train the GNN, save it |
| `dr-active evaluate --models rf,gnn` | all three splits, both models, plus baseline and a per-target breakdown; writes `results/` (narrow it with `--splits scaffold`) |
| `dr-active predict SMILES --target NAME` | one prediction (`--model-type gnn` for the GNN) |

Every training flag (`--seed`, `--n-estimators`, `--epochs`, `--hidden-dim`,
`--num-layers`, `--batch-size`, `--learning-rate`, `--device`, `--test-fraction`,
`--splits`)
defaults to the value in `src/dractive/config.py`.

## Data provenance

All data comes from the **ChEMBL web services** activity endpoint
(`https://www.ebi.ac.uk/chembl/api/data/activity`), downloaded by
`scripts/download_chembl.py`. ChEMBL data is released under CC BY-SA 3.0.

Targets were chosen for being well populated and mechanistically different from
each other, so that "target name" is a real input rather than decoration:

| name | ChEMBL id | description |
| --- | --- | --- |
| EGFR | CHEMBL203 | epidermal growth factor receptor (kinase) |
| JAK2 | CHEMBL2971 | tyrosine-protein kinase JAK2 |
| BACE1 | CHEMBL4822 | beta-secretase 1 (aspartyl protease) |
| DRD2 | CHEMBL217 | dopamine D2 receptor (GPCR) |
| HERG | CHEMBL240 | hERG potassium channel |
| ACHE | CHEMBL220 | acetylcholinesterase |

Download filters (applied server-side): `assay_type=B` (binding assays),
`standard_relation="="` (exact values only, no censored `>`/`<` data), and a
non-null `pchembl_value`.

Cleaning (`src/dractive/prepare.py`), applied in this order:

1. keep only `IC50`, `Ki`, `Kd` measurements;
2. drop any row ChEMBL itself flagged with a `data_validity_comment`;
3. drop targets outside the six above;
4. require `3.0 <= pChEMBL <= 12.0`;
5. parse each SMILES with RDKit, keep the **largest covalent fragment** (so a
   hydrochloride salt and its free base become the same molecule), drop anything
   unparseable or above 80 heavy atoms;
6. group by (canonical SMILES, target) and average the replicates — but **drop**
   any pair whose replicate measurements span more than 2.0 log units, since for
   those the "true" value is not knowable from this data.

Measured effect of that cleaning on the download of 2026-10-01: **85,472 raw
activity records in, 55,176 (molecule, target) pairs out (64.6% kept)**, over
54,187 distinct molecules. Per-target counts and label spread:

| target | pairs | mean pChEMBL | std | min | max |
| --- | --- | --- | --- | --- | --- |
| JAK2 | 11,580 | 7.49 | 1.29 | 3.84 | 10.97 |
| EGFR | 10,154 | 6.95 | 1.31 | 4.00 | 11.00 |
| BACE1 | 9,841 | 6.82 | 1.21 | 3.00 | 10.96 |
| HERG | 9,395 | 5.49 | 0.90 | 4.00 | 9.85 |
| DRD2 | 8,269 | 6.75 | 1.01 | 4.00 | 10.83 |
| ACHE | 5,937 | 6.14 | 1.29 | 4.00 | 10.96 |

ChEMBL is a moving target, so a later download will not reproduce these counts
exactly. `data/` is gitignored. `tests/fixtures/affinity_sample.csv` is a
600-row stratified sample of the prepared table, committed so the test suite
runs end-to-end on real molecules without a download.

## Evaluation protocol

Three splits, all 80/20, all scaffold-disjoint where the name says so:

- **`random`** — uniform over rows. Optimistic: close analogues of test
  molecules are usually in training.
- **`scaffold`** — rows grouped by Bemis-Murcko scaffold, whole groups assigned
  to one side, largest groups kept in training. **On this dataset that makes the
  test set entirely singleton scaffolds**: of 22,723 scaffolds, 15,257 occur
  once, which is more than the 11,035-row test quota, so the quota is filled
  from singletons alone. A genuinely hard test of one-off chemotypes, but not a
  representative sample of the data, so it is reported as the pessimistic bound
  rather than *the* answer.
- **`scaffold_shuffled`** — the same grouping, but whole groups are taken in a
  seeded random order, so the test set is scaffold-disjoint *and*
  frequency-weighted (it contains well-populated chemotypes too). This is the
  fairer "unseen scaffold" estimate.
- **Baseline** — per-target mean of the *training* labels only. Reported on
  every row of the results table. A model that cannot beat it has learned
  nothing about chemistry, only about which target is which.
- Each model is trained from scratch inside each split; the test rows are never
  featurized, fit, or used for early stopping. The GNN's early-stopping
  validation slice is carved out of the training rows.

<!-- RESULTS:START -->
## Results

Measured on 2026-10-01 from one `uv run dr-active evaluate --models rf,gnn` run:
**55,176 (molecule, target) pairs**, 80/20 split, seed 42, so 44,141 training and
11,035 test rows for each split. Units are pChEMBL log units. Regenerate with
the same command; raw output is in `results/`.

| split | model | RMSE | MAE | Pearson r | Spearman rho |
| --- | --- | --- | --- | --- | --- |
| random | **random forest** | **0.648** | 0.474 | **0.880** | 0.876 |
| random | GNN | 0.856 | 0.653 | 0.798 | 0.793 |
| random | per-target mean (baseline) | 1.180 | 0.941 | 0.490 | 0.467 |
| scaffold | **random forest** | **0.780** | 0.580 | **0.817** | 0.808 |
| scaffold | GNN | 0.860 | 0.654 | 0.777 | 0.770 |
| scaffold | per-target mean (baseline) | 1.180 | 0.947 | 0.474 | 0.455 |
| scaffold_shuffled | **random forest** | **0.761** | 0.565 | **0.827** | 0.825 |
| scaffold_shuffled | GNN | 0.974 | 0.759 | 0.707 | 0.704 |
| scaffold_shuffled | per-target mean (baseline) | 1.186 | 0.949 | 0.473 | 0.447 |

Reading these honestly:

- The random forest beats the baseline on all three splits: RMSE 0.65 vs 1.18
  random, 0.78 vs 1.18 scaffold, 0.76 vs 1.19 scaffold_shuffled. On the
  scaffold splits that is a 34-36% error reduction, and **0.76-0.78 log units is
  still a factor of ~6 in concentration** — useful for ranking, not for
  predicting a number.
- **The two scaffold splits agree**, which is the most reassuring number here.
  The all-singleton `scaffold` split (0.780) and the frequency-weighted
  `scaffold_shuffled` split (0.761) land 0.02 log units apart for the forest, so
  the pessimistic bound was not an artifact of testing only on one-off
  chemotypes — generalising to an unseen scaffold really does cost ~0.12 log
  units over a random split, however the unseen scaffolds are chosen.
- The baseline's non-zero correlation (r ≈ 0.48) comes entirely from the targets
  having different mean affinities. It is a reminder of how much of a naive
  "accuracy" figure is just target identity, which is why it is on every row.
- **The GNN does not beat the random forest.** It is worse on all three splits
  (RMSE 0.86 vs 0.65 random, 0.86 vs 0.78 scaffold, 0.97 vs 0.76
  scaffold_shuffled). It looked like the more split-robust of the two on the
  first two splits — 0.856 → 0.860 while the forest lost 0.13 — but
  `scaffold_shuffled` refutes that: the GNN degrades to 0.974, its worst number
  anywhere, while the forest holds at 0.761. The apparent robustness was a
  property of the all-singleton split, not of the model. No architecture or
  hyperparameter search was run; 40 epochs of one configuration is all this is.
- The forest's single most important feature is the HERG target indicator
  (importance 0.121, next highest 0.042): most of the easy signal is "which
  target is this", exactly what the baseline captures.

Per-target, scaffold split (the harder number), RMSE in log units:

| target | n test | baseline | random forest | GNN |
| --- | --- | --- | --- | --- |
| ACHE | 1,262 | 1.311 | **0.902** | 1.030 |
| BACE1 | 1,846 | 1.297 | **0.856** | 0.849 |
| DRD2 | 1,816 | 1.049 | **0.728** | 0.837 |
| EGFR | 1,764 | 1.269 | **0.867** | 1.002 |
| HERG | 2,257 | 0.862 | **0.646** | 0.693 |
| JAK2 | 2,090 | 1.309 | **0.724** | 0.809 |

Both models beat the baseline on every target. HERG is the easiest in absolute
RMSE, but it is also the target whose labels have the smallest spread
(std 0.90 vs ~1.3 for the kinases), so the baseline is already strong there; the
relative improvement is smallest for HERG and DRD2. BACE1 is the one target
where the GNN edges out the forest (0.849 vs 0.856).

Runtimes on an M3 Pro (18 GB, CPU): the full `evaluate --models rf,gnn` run over
all three splits — 3 × (featurize + forest + 30-40 GNN epochs) — took **66 min**. Training the forest alone on all 55,176 rows takes **9 min**, most
of it RDKit featurization rather than tree fitting. A GNN epoch on 44k molecules
is roughly 25-30 s on an otherwise idle machine, so one GNN training run is
10-20 min depending on when early stopping fires. Nothing here needs a GPU.

<!-- RESULTS:END -->

## Design decisions

- **No PyTorch Geometric.** Its macOS arm64 wheels are awkward, and the whole
  GNN needs is sparse neighbour aggregation. `src/dractive/gnn/graph.py` batches
  graphs by concatenating them and tracking a `graph_id` per atom;
  `segment_sum`/`segment_mean` do the aggregation with `index_add`. A test
  asserts batched predictions equal single-graph predictions.
- **GNN defaults to CPU.** These molecules average about 20 atoms, so kernel
  launch overhead dominates: one epoch over 5,000 molecules measured 2.5 s on
  CPU versus 7.0 s on Apple MPS (M3 Pro). `--device mps` is still available.
- **pChEMBL, not raw IC50.** ChEMBL's `pchembl_value` is already a -log10 molar
  value, which is the scale affinity errors are usually quoted on and keeps
  IC50/Ki/Kd roughly commensurate. Mixing those three assay types is itself a
  source of noise (see Limitations).
- **Replicate disagreement is dropped, not averaged.** Pairs whose measurements
  span more than 2 log units are discarded rather than smoothed into a mean no
  experiment supports.
- **GNN checkpoints hold only plain types** (the config is stored as a dict, not
  a pickled dataclass) so they load under `torch.load(weights_only=True)`.
  The forest is persisted with joblib, which is pickle-based, so only load
  `rf.joblib` files you produced yourself.
- **The saved forest is big.** 300 trees grown to purity on 55k rows is a
  293 MB joblib file. That is the cost of `min_samples_leaf=1`; raise it (or
  lower `--n-estimators`) if the artefact size matters more than the last
  hundredth of RMSE.
- **Target enters as an identifier, not a sequence.** One-hot for the forest, a
  learned embedding for the GNN. Nothing about the protein's structure or
  sequence is used, so the models cannot generalise to an unseen target — and
  `predict` warns on stderr when asked for a target the loaded model never saw.

## Limitations

- **Mixed assay types.** IC50, Ki and Kd are pooled. They are not the same
  physical quantity; IC50 in particular depends on assay conditions. Published
  analyses of repeated public measurements (Kramer et al., *J. Med. Chem.* 2012,
  for Ki; Kalliokoski et al., *PLoS ONE* 2013, for mixed IC50) put the
  experimental spread at roughly half a log unit or more, so an RMSE below about
  0.5-0.7 on data like this would say more about leakage than about modelling.
- **Six targets only.** There is no protein representation, so the model cannot
  say anything about a target it was not trained on.
- **Scaffold split is the number that matters.** The random-split numbers are
  inflated by analogue leakage and are reported only for contrast.
- **No applicability domain.** The forest's tree spread is reported, but it is
  not calibrated; a confident-looking prediction for a molecule unlike anything
  in ChEMBL means nothing.
- **Activity cliffs.** Single-atom changes can move affinity by orders of
  magnitude; a descriptor/fingerprint model with pooled graph readout cannot see
  that, and per-molecule errors are much larger than the aggregate RMSE suggests.
- **Publication bias.** ChEMBL over-represents molecules that worked. The
  pChEMBL distribution is not the distribution of chemical space.

## Layout

```
src/dractive/
  config.py      constants: targets, filters, thresholds, hyperparameters
  chem.py        validated RDKit wrappers (SMILES parsing, Murcko scaffolds)
  targets.py     target name/alias/ChEMBL-id resolution, fixed ordering
  chembl.py      paginated ChEMBL activity client (injectable HTTP getter)
  prepare.py     raw activities -> clean modelling table
  dataset.py     prepared-CSV loading and validation
  features.py    descriptors + Morgan bits + target one-hot
  splits.py      random, scaffold and frequency-weighted scaffold splits
  metrics.py     RMSE/MAE/Pearson/Spearman + per-target-mean baseline
  model_api.py   the AffinityModel protocol both models satisfy
  rf_model.py    random forest: train, predict, persist
  gnn/graph.py   molecule -> graph tensors, manual batching
  gnn/model.py   message-passing network
  gnn/train.py   GNN training loop, persistence, inference
  evaluate.py    split x model x baseline harness, results writers
  reporting.py   per-target metrics and prediction dumps
  commands.py    CLI command implementations
  cli.py         argument parsing and exit-code mapping
scripts/
  download_chembl.py   resumable per-target download
  prepare_dataset.py   cleaning + fixture generation
tests/                 pytest suite (unit + integration on the fixture)
```

## License

MIT (see `LICENSE`). ChEMBL data is CC BY-SA 3.0 and is not redistributed here.
