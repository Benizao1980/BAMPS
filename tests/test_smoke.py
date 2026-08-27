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
