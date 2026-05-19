#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python - <<'PY'
import importlib.util
import subprocess
import sys

requirements = [
    ("cv2", "opencv-python"),
    ("numpy", "numpy"),
    ("faiss", "faiss-cpu"),
    ("matplotlib", "matplotlib"),
    ("tqdm", "tqdm"),
]

missing = [package for module, package in requirements if importlib.util.find_spec(module) is None]

if missing:
    print("Installing missing packages:", ", ".join(missing))
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)
else:
    print("All required packages are already available.")
PY

python -B kaggle_run.py --mode auto
