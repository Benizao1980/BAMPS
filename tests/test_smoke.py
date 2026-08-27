from pathlib import Path
import subprocess
import sys

SCRIPTS = [
    "train_model.py", "tune_model.py", "predict.py", "predict_all.py",
    "evaluate_mic_predictions.py", "run_amrfinder.py", "plot_predicted_mic_panel.py",
]

def test_cli_help():
    root = Path(__file__).resolve().parents[1]
    for script in SCRIPTS:
        result = subprocess.run([sys.executable, str(root / "scripts" / script), "--help"], capture_output=True, text=True)
        assert result.returncode == 0, f"{script}: {result.stderr}"

def test_toy_data_present_and_joinable():
    import pandas as pd
    root = Path(__file__).resolve().parents[1]
    features = pd.read_csv(root / "examples" / "toy_example" / "features.tsv", sep="\t")
    phenotypes = pd.read_csv(root / "examples" / "toy_example" / "phenotypes.tsv", sep="\t")
    assert len(features) > 0
    assert len(phenotypes) > 0
    common = set(features.iloc[:, 0].astype(str)) & set(phenotypes.iloc[:, 0].astype(str))
    assert common, "Toy feature and phenotype tables have no common sample IDs"
