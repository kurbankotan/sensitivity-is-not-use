# Sensitivity Is Not Use

Code and analysis for the study *Sensitivity Is Not Use: Testing Whether Structure-Based
Affinity Models Learn from Geometry*.

## What this is

A diagnostic protocol, not a new scoring function. It asks one question of a
structure-based protein-ligand affinity model: does the three-dimensional geometry it is
given actually contribute to the prediction?

The protocol separates two things that are routinely conflated. A model can be
**sensitive** to geometry (perturbing coordinates changes its output) without **using**
it (removing geometry and retraining costs nothing). Only the second is evidence that
geometry carries signal the model has learned to read. Perturbation probes measure the
first and are regularly read as though they measured the second.

Three architectures were tested, each against a retrained control arm rather than against
its own perturbation response.

| Architecture | Parameters | Inference probe says | Retraining from scratch says |
|---|---:|---:|---:|
| Pair representation (axial attention) | 1.962 M | +0.004 pK | -0.013 pK |
| Voxel 3D-CNN | 0.761 M | +1.385 pK | +0.0004 pK |
| E(n)-invariant graph network | 0.431 M | +0.052 pK | -0.0017 pK |
| Contact counts, random forest | - | +0.511 pK | +0.0238 pK |

The contact-count forest is the positive control: a model that provably does use
geometry. It is the only arm where removing geometry and retraining costs anything
two-level significant (+0.0236 pK, Welch p = 4e-6). Even there the inference probe
overstates the cost by a factor of 21. For the voxel network the probe overstates it by
a factor of about 3500.

The proposed explanation, offered as a hypothesis with a testable prediction rather than
a conclusion: the whole geometric signal is about 0.024 pK, while seed-to-seed standard
deviation is 0.0170 pK for the pair model, 0.0183 for the equivariant network and 0.0249
for the voxel network, against 0.0026 for the forest. The signal sits at roughly one
seed-standard-deviation for every neural architecture and about nine for the forest.

## Reproducing the results

Everything runs from a single Colab notebook, top to bottom, with `Run all`:

```
notebooks/EnsAff_Colab_v17.ipynb
```

The notebook installs the package, fetches structures, trains every arm, runs the probes
and the controls, and prints a report block at the end. It needs a GPU runtime; High-RAM
is not required (the chain cache in `ensaff/ccdyn.py` is capped at 1500 parsed chains for
exactly this reason).

To run the package directly instead:

```bash
pip install -r requirements.txt
pip install -e .
python -m ensaff.train --help
```

## Data

All data are public and need no registration. Affinity labels and splits come from the
Leak Proof PDBBind release (CL1 non-covalent: 7393 train, 1370 validation, 4250 test, of
which 4222 test complexes are scored in the paired comparisons). Complex structures are
downloaded by identifier from the RCSB Protein Data Bank; the retrieval and validation
code is in `ensaff/fetch_structures.py` and `tools/validate_extraction.py`. Structure
files are not redistributed here.

## Layout

```
ensaff/      the pipeline
  data.py              splits, labels, leak-proof split handling
  features.py          featurisation
  embed.py             protein representations
  model.py             pair-representation model (axial attention)
  voxel.py             voxel 3D-CNN, second architecture
  equiv.py             E(n)-invariant graph network, third architecture
  contactcount.py      contact-count features
  ccdyn.py             shell ladder, sibling conformations, RMSF, paired bootstrap
  conformers.py        sibling-crystal ensembles by Kabsch superposition
  geom_probe.py        inference-time perturbation probes
  ablate_states.py     retrained control arms
  baseline.py          ligand-fingerprint baseline
  train.py             training, evaluation, pose discrimination
  ensemble.py          multi-seed ensembling
  compare.py           paired bootstrap and per-seed Welch tests
  evaluate.py          metrics
  applicability.py     applicability domain
  fetch_structures.py  RCSB retrieval
  report.py            the report block the notebook prints

tools/       tests and validation
  test_voxel.py             grid framing, mass conservation
  test_ccdyn.py             shell ladder, zero-shell collapse to the outer product
  test_equiv.py             translation and rotation invariance to ~1e-8
  validate_voxel_probe.py   probe sanity on the voxel grid
  validate_extraction.py    structure retrieval correctness
  ...

paper/       figure and manuscript checks
  figstyle.py   shared style and the DATA block; every number carries a comment
                naming the table it came from, and nothing is recomputed here
  figures.py    the 13 figures and the TOC graphic
  check.py      structural checks on the manuscript
  numcheck.py   fails if an edit introduces a number that was not there before
  refs.py       the reference list
```

## Statistical protocol

Every comparison is paired on the same test complexes and reported two ways: a paired
bootstrap over complexes (2000 resamples, 95% percentile interval) and a per-seed Welch
t-test over independently seeded runs. An effect is called real only when both levels
agree. Where they disagree, the paper says so and does not claim the effect.

Pose-discrimination chance level is computed from the states actually used at evaluation
as `1 / C(n_real + n_decoy, n_real)`, in `train.py:_chance_level`. An earlier version
printed a chance level derived from the wrong decoy count; that defect and seven others
found by audit are recorded in section 6.11 of the paper, including three retractions of
claims the experiments did not support.

## Citation

See `CITATION.cff`.

## Licence

MIT, see `LICENSE`. The licence covers the code in this repository. It does not cover the
manuscript text or figures, and it does not grant rights under any patent application.

## Generative AI declaration

Claude (Anthropic) was used to write this pipeline, to propose and specify the
perturbation probes and the retrained control arms, to perform the statistical analysis
of the logged outputs, and to draft the manuscript. The same assistant conducted an
adversarial audit of its own code, which is how several of the defects in section 6.11
were found. The author takes full responsibility for the content.
