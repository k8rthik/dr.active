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
scaffold split. The measured numbers are in [Results](#results); they are the
numbers the code produced, including where a model fails to beat the baseline.

This is a prototype, not a tool for making decisions about real chemistry. See
[Limitations](#limitations).

## Install

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11 or 3.12.

```bash
git clone https://github.com/k8rthik/dr.active.git
cd dr.active
uv sync
uv run pytest            # ~160 tests, no network, no downloads
```

## Quick start

```bash
# 1. download raw activities from ChEMBL (~90k records, ~35 MB, a few minutes)
uv run python scripts/download_chembl.py

# 2. clean them into data/processed/affinity.csv
uv run python scripts/prepare_dataset.py

# 3. train and evaluate on both splits
uv run dr-active evaluate --models rf,gnn

# 4. train a model you want to keep, then predict with it
uv run dr-active train-rf
uv run dr-active predict "CC(=O)Oc1ccccc1C(=O)O" --target EGFR
```

`predict` output:

```
target:            EGFR
model:             rf
predicted pChEMBL: 5.42
  approx.          3,802 nM
  tree spread      +/- 0.51 (disagreement, not a CI)
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
| `dr-active evaluate --models rf,gnn` | both splits, both models, plus baseline; writes `results/` |
| `dr-active predict SMILES --target NAME` | one prediction (`--model-type gnn` for the GNN) |

Every training flag (`--seed`, `--n-estimators`, `--epochs`, `--hidden-dim`,
`--num-layers`, `--batch-size`, `--learning-rate`, `--device`, `--test-fraction`)
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

`data/` is gitignored. `tests/fixtures/affinity_sample.csv` is a small
stratified sample of the prepared table, committed so the test suite runs
end-to-end on real molecules without a download.

## Evaluation protocol

- **Random split** — uniform over rows. Optimistic: close analogues of test
  molecules are usually in training.
- **Scaffold split** — rows are grouped by Bemis-Murcko scaffold and whole
  groups are assigned to one side, so no scaffold appears in both. Large groups
  stay in training and the test set is filled from the smaller ones, which is
  the harder and more honest number for "will this work on a new chemotype".
- **Baseline** — per-target mean of the *training* labels only. Reported on
  every row of the results table. A model that cannot beat it has learned
  nothing about chemistry, only about which target is which.
- Each model is trained from scratch inside each split; the test rows are never
  featurized, fit, or used for early stopping. The GNN's early-stopping
  validation slice is carved out of the training rows.

<!-- RESULTS:START -->
## Results

Run `uv run dr-active evaluate --models rf,gnn` to regenerate; numbers land in
`results/results.json`.

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
- **Target enters as an identifier, not a sequence.** One-hot for the forest, a
  learned embedding for the GNN. Nothing about the protein's structure or
  sequence is used, so the models cannot generalise to an unseen target — and
  `predict` warns on stderr when asked for a target the loaded model never saw.

## Limitations

- **Mixed assay types.** IC50, Ki and Kd are pooled. They are not the same
  physical quantity; IC50 in particular depends on assay conditions. Public
  reproducibility studies put the noise floor for heterogeneous ChEMBL IC50 data
  at roughly 0.5-0.8 log units, which is the realistic floor for RMSE here.
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
  splits.py      random and scaffold splits
  metrics.py     RMSE/MAE/Pearson/Spearman + per-target-mean baseline
  rf_model.py    random forest: train, predict, persist
  gnn/graph.py   molecule -> graph tensors, manual batching
  gnn/model.py   message-passing network
  gnn/train.py   GNN training loop, persistence, inference
  evaluate.py    split x model x baseline harness, results writers
  commands.py    CLI command implementations
  cli.py         argument parsing and exit-code mapping
scripts/
  download_chembl.py   resumable per-target download
  prepare_dataset.py   cleaning + fixture generation
tests/                 pytest suite (unit + integration on the fixture)
```

## License

MIT (see `LICENSE`). ChEMBL data is CC BY-SA 3.0 and is not redistributed here.
