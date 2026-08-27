# BAMPS

**Bacterial AMR Modelling and Prediction Suite**

BAMPS is a small, reproducible toolkit for building, validating and applying antimicrobial-resistance phenotype prediction models from bacterial genome-derived feature matrices.

The package supports quantitative MIC regression and categorical resistance prediction. It is deliberately feature-agnostic: models can use curated AMR determinants, GWAS-derived features, pangenome features, or combinations of these.

## Scope

BAMPS contains the reusable modelling framework only. Species- and manuscript-specific data processing, GWAS interpretation and figure-generation scripts belong in companion analysis repositories. The worked *Acinetobacter baumannii* analysis is maintained separately in [Acinetobacter-resistance-architectures](https://github.com/Benizao1980/Acinetobacter-resistance-architectures).

## Core workflow

1. Build an AMR determinant matrix from assemblies with `run_amrfinder.py` (optional).
2. Train per-antibiotic regression or classification models with `train_model.py`.
3. Tune supported models with `tune_model.py`.
4. Apply saved models with `predict.py` or `predict_all.py`.
5. Evaluate quantitative MIC predictions with `evaluate_mic_predictions.py`.

See `docs/quickstart.md`, `docs/input_formats.md` and `docs/supported_interface.md`.

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For AMRFinderPlus feature extraction, install NCBI AMRFinderPlus separately and ensure `amrfinder` is available on `$PATH`.

## Minimal example

```bash
python scripts/train_model.py \
  --feature-table examples/toy_example/features.tsv \
  --mic-file examples/toy_example/phenotypes.tsv \
  --task regression \
  --classifier ridge \
  --model-dir outputs/example/models \
  --plot-dir outputs/example/plots \
  --self-test
```

The toy data are synthetic and exist only to test file formats and command execution.

## Reproducibility

For publication-grade analyses, retain the exact input feature matrix, phenotype table, command line, random seed, model metadata and software environment. Feature selection must be performed using training data only when evaluating held-out or external samples.

## Citation

If you use BAMPS, cite the archived release DOI once available. See `CITATION.cff`.

## Licence

GPL-3.0. See `LICENSE`.

## Status

This is a pre-1.0 research software release. Interfaces may change while the API and test coverage are consolidated.
