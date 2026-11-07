# Exact Edge-Edit Certificates for One-Layer Mean-Aggregation Graph Models

Code, raw per-split results and scripts that regenerate every number, table and figure of the paper.

## Overview

For a one-layer graph model whose score uses the graph only through a mean of neighbour scores,

```
logit_i,c = f_c(x_i) + mean_{u in N(i)} g_c(x_u)
```

the worst set of edge insertions and deletions within a budget *b* can be found exactly by sorting: only edges at
node *i* matter, and for *p* deletions and *q* insertions the worst case removes the *p* largest and adds the *q*
smallest neighbour scores. For additive models (CertGAM) the same argument certifies feature attributions, and
certified training is convex. The repository also contains an exact single-edit certificate for two-layer
GraphSAGE, a degree-shrinkage variant (ShrinkGAM) and the baselines (randomized smoothing, π-PPNP).

The main empirical result is that graph-free models, which edge edits cannot affect, usually have higher certified
accuracy than certified graph models: at one edit, 72 of 77 model–dataset cells on the non-trivial datasets are
below the validation-selected graph-free reference.

## Requirements

* Python 3.12, PyTorch and PyTorch Geometric (versions used for the paper are pinned in `requirements.txt`)
* levels 1–2 and the quick start run on CPU; GPU checks and reruns need CUDA (the paper used one 8 GB GPU)

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # choose the torch build matching your CUDA
```

Datasets are downloaded by PyTorch Geometric into `data/` on first use (about 270 MB).
Set `PY=/path/to/python` to use another interpreter.

## Quick start

```bash
python scripts/quickstart.py Texas 3     # CPU, a few seconds
```

Trains CertGAM on one split with its frozen configuration, certifies every test node exactly against
1, 2, 3, 5 and 10 edge edits, and prints the graph-free reference next to it.

## Verifying the paper

All scripts work on a copy in `work/`; the shipped results are never overwritten.

| Level | Command | Hardware | Time | Checks |
|---|---|---|---|---|
| 1 | `bash scripts/verify_paper.sh` | CPU | ~1 min | verdicts and all generated tables/numbers match `expected/gen/` |
| 2 | `bash scripts/run_checks.sh` | CPU (+GPU for 5 checks) | ~10 min | certificates are exact (brute force on small graphs), theorems hold |
| 3 | `bash scripts/rerun.sh RESULT CMD...` | GPU | per run | a shipped result is reproduced |

Level 3 examples:

```bash
bash scripts/rerun.sh s03/Texas__certrob.json      exp.py s03 Texas certrob   # CertGAM-R
bash scripts/rerun.sh s05/Texas__sage2.json        exp6.py Texas              # two-layer SAGE
bash scripts/rerun.sh s07/Texas__shrink.json       exp8.py Texas              # ShrinkGAM-R
bash scripts/rerun.sh s09/Texas__pippnp_exact.json exp10e.py Texas            # pi-PPNP
```

`rerun.sh` deletes the target file in `work/`, reruns it with the configuration frozen on validation data, and
compares the mean accuracy and certified accuracy at *b* = 1 with the shipped result (tolerance two points).
Convex models reproduce exactly; GPU-trained neural models vary within about one point.

Expected output of level 1:

```
SAME    verdict.json  ...  SAME    verdict8.json
PASS: 20 generated files identical; figures in work/paper/figs/
```

## Results format

Each `results/sXX/DATASET__MODEL.json` holds the selected `cfg`, a `status` (`ok`, `OOM`, `error`) and `rows`,
one entry per test split (3–9) with clean accuracy `acc`, certified accuracy `cert` (per budget) or `cert1`,
and, where recorded, per-node degrees, radii and the number of real edits tried against the certificates.
Configurations were selected on the validation set of split 0 only; failed runs are kept, not replaced.

## Layout

```
code/          models, certificates, runners, hypothesis tests, correctness checks, make_paper.py, make_figs.py
  cert.py        CertGAM: exact certificates and convex certified training
  shrink.py      ShrinkGAM: degree shrinkage
  cert2.py       two-layer GraphSAGE: exact single-edit certificate
  ppr.py         pi-PPNP certificates
  smooth.py      randomized-smoothing baselines
  s02.sh..s09b.sh  full experiment sequence (about 45 GPU hours)
results/       per-split results of every run (s02 selection, s03 main, s04-s09 further studies), verdicts
expected/gen/  generated LaTeX used by the paper (reference for level 1)
prereg/        analysis plans and hashes fixed before the confirmatory runs
scripts/       verification entry points
```

## Licence

MIT. Datasets keep their original licences.
