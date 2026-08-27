# Quick start

BAMPS expects a feature table with one row per isolate and a phenotype table containing the same isolate identifiers and one or more antibiotic columns.

## 1. Train

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

## 2. Tune

```bash
python scripts/tune_model.py \
  --feature-table examples/toy_example/features.tsv \
  --mic-file examples/toy_example/phenotypes.tsv \
  --task regression \
  --classifier ridge \
  --antibiotics drug_a drug_b \
  --log2 --cv 3 --n-iter 5 \
  --outdir outputs/example/tuning
```

## 3. Predict and evaluate

Use `predict.py` for a single saved model or `predict_all.py` for a directory following the BAMPS model naming convention. `evaluate_mic_predictions.py` calculates MAE, RMSE, R² and dilution accuracy from quantitative MIC predictions.
