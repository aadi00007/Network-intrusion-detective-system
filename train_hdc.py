#!/usr/bin/env python
"""Quick script to train HDC model and show progress"""
import sys
import subprocess

cmd = [
    sys.executable,
    "nsl_kdd_analysis.py",
    "train",
    "--train_path", "KDDTrain+_20Percent.txt",  # Use smaller subset for testing
    "--test_path", "KDDTest+.txt",
    "--model_type", "hdc",
    "--model_out", "models/nsl_kdd_hdc.joblib",
    "--label_map_out", "models/label_map.joblib",
    "--report_out", "reports/metrics_hdc.json"
]

print("Starting HDC training...")
print("Command:", " ".join(cmd))
print("-" * 60)

try:
    result = subprocess.run(cmd, check=False, capture_output=False)
    print("\n" + "=" * 60)
    print(f"Training completed with exit code: {result.returncode}")
except KeyboardInterrupt:
    print("\nTraining interrupted by user")
except Exception as e:
    print(f"\nError: {e}")

