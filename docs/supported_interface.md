# Supported command-line interface

For the v0.1.x series, the supported public scripts are:

- `run_amrfinder.py` — construct AMR determinant features from assemblies/AMRFinderPlus output.
- `train_model.py` — train per-antibiotic regression/classification models.
- `tune_model.py` — tune supported estimators on training data.
- `predict.py` — apply one saved model.
- `predict_all.py` — apply a set of saved per-antibiotic models.
- `evaluate_mic_predictions.py` — evaluate quantitative MIC predictions.
- `plot_predicted_mic_panel.py` — lightweight predicted/observed MIC visualisation.

Species-specific GWAS interpretation, lineage analysis, genomic-context logic and manuscript figure construction are deliberately outside the BAMPS API and belong in companion analysis repositories.
