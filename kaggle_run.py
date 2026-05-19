import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_command(command, title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    subprocess.run(command, check=True)


def path_exists(path):
    return path is not None and os.path.exists(path)


def print_environment(CFG):
    print("\nDetected paths")
    print(f"  DATA_DIR        : {CFG.DATA_DIR}")
    print(f"  IMAGE_DIR       : {CFG.IMAGE_DIR}")
    print(f"  JSON_PATH       : {CFG.JSON_PATH}")
    print(f"  SPLIT_PATH      : {CFG.SPLIT_PATH}")
    print(f"  FEATURES_DIR    : {CFG.FEATURES_DIR}")
    print(f"  WEIGHTS_DIR     : {CFG.WEIGHTS_DIR}")
    print(f"  MODEL_SAVE_PATH : {CFG.BEST_MODEL_SAVE_PATH}")
    print(f"  MODEL_LOAD_PATH : {CFG.resolve_best_model_path()}")


def validate_dataset_paths(CFG):
    missing = []
    for label, path in [
        ("IMAGE_DIR", CFG.IMAGE_DIR),
        ("JSON_PATH", CFG.JSON_PATH),
        ("SPLIT_PATH", CFG.SPLIT_PATH),
    ]:
        if not path_exists(path):
            missing.append(f"{label}: {path}")

    if missing:
        print("\nCannot find required FSC147 files.")
        for item in missing:
            print(f"  - {item}")
        print("\nOn Kaggle, add the FSC147 dataset under /kaggle/input.")
        print("Expected files: annotation_FSC147_384.json, Train_Test_Val_FSC_147.json, images_384_VarV2/")
        raise SystemExit(1)


def model_file_exists(CFG):
    return path_exists(CFG.resolve_best_model_path())


def calibration_file_exists(CFG):
    return path_exists(CFG.resolve_count_calibration_path())


def main():
    parser = argparse.ArgumentParser(description="Kaggle runner for PCA/Whitening + FAISS evaluation.")
    parser.add_argument(
        "--mode",
        choices=["auto", "evaluate", "all"],
        default="auto",
        help=(
            "auto: evaluate if PCA/count calibration exist, otherwise prepare the missing parts. "
            "evaluate: only run evaluate.py. "
            "all: always extract features + train + count calibration + evaluate."
        ),
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    os.chdir(repo_root)
    sys.path.insert(0, str(repo_root))

    from src.config import CFG
    from src.utils import configure_utf8_output

    configure_utf8_output()
    print_environment(CFG)
    validate_dataset_paths(CFG)

    if args.mode == "evaluate" and not model_file_exists(CFG):
        print("\nCannot run evaluate because pca_whitening_best.npz was not found.")
        print("Either upload weights/pca_whitening_best.npz with the code, or run:")
        print("  python -B kaggle_run.py --mode all")
        raise SystemExit(1)

    should_train = args.mode == "all" or (args.mode == "auto" and not model_file_exists(CFG))

    if should_train:
        run_command([sys.executable, "-B", "src/feature_extractor.py"], "Step 1/3 - Extract HOG+HSV+LBP features")
        run_command([sys.executable, "-B", "src/train.py"], "Step 2/3 - Fit PCA/Whitening and save best checkpoint")
    else:
        print("\nFound PCA/Whitening model, skipping feature extraction and training.")

    should_calibrate = args.mode == "all" or (args.mode == "auto" and not calibration_file_exists(CFG))
    if should_calibrate:
        run_command([sys.executable, "-B", "calibrate.py"], "Calibration - Tune threshold/NMS/count scale on validation")
    else:
        print("\nFound count calibration, skipping calibration.")

    run_command([sys.executable, "-B", "evaluate.py"], "Step 3/3 - Evaluate MAE/RMSE")


if __name__ == "__main__":
    main()
