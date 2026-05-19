import json
import os

import cv2
import numpy as np
from tqdm import tqdm

from src.config import CFG
from src.counting import (
    annotation_boxes_to_rectangles,
    build_detection_candidates,
    count_from_candidates,
    fit_linear_count_calibration,
    save_count_calibration,
)
from src.model import PCAWhiteningModel
from src.utils import configure_utf8_output


def select_calibration_images(image_names):
    max_images = CFG.COUNT_CALIBRATION_MAX_IMAGES
    if max_images <= 0 or len(image_names) <= max_images:
        return image_names

    rng = np.random.default_rng(CFG.RANDOM_SEED)
    indices = np.sort(rng.choice(len(image_names), size=max_images, replace=False))
    return [image_names[idx] for idx in indices]


def compute_mae_rmse(pred_counts, gt_counts):
    pred_counts = np.asarray(pred_counts, dtype=np.float32)
    gt_counts = np.asarray(gt_counts, dtype=np.float32)
    errors = pred_counts - gt_counts
    return float(np.mean(np.abs(errors))), float(np.sqrt(np.mean(errors ** 2)))


def main():
    configure_utf8_output()
    os.makedirs(CFG.WEIGHTS_DIR, exist_ok=True)

    model_path = CFG.resolve_best_model_path()
    model = PCAWhiteningModel.load(model_path)
    print(f"Loaded PCA/Whitening model: {model_path}")

    with open(CFG.JSON_PATH, "r", encoding="utf-8") as f:
        annotations = json.load(f)
    with open(CFG.SPLIT_PATH, "r", encoding="utf-8") as f:
        splits = json.load(f)

    image_names = [name for name in splits.get("val", []) if name in annotations]
    image_names = select_calibration_images(image_names)
    print(f"Calibrating count on {len(image_names)} validation images")

    records = []
    for img_name in tqdm(image_names, desc="Collecting validation candidates"):
        img_path = os.path.join(CFG.IMAGE_DIR, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        annotation = annotations[img_name]
        exemplar_rectangles = annotation_boxes_to_rectangles(
            annotation.get("box_examples_coordinates", []),
            img.shape,
        )
        if not exemplar_rectangles:
            continue

        scores, boxes = build_detection_candidates(img, model, exemplar_rectangles)
        if len(scores) == 0:
            continue

        records.append({
            "scores": scores,
            "boxes": boxes,
            "gt_count": len(annotation.get("points", [])),
        })

    if not records:
        raise RuntimeError("Không thu được validation record nào để calibrate count.")

    best = None
    raw_baseline = None

    for similarity_threshold in CFG.COUNT_CALIBRATION_THRESHOLDS:
        for nms_threshold in CFG.COUNT_CALIBRATION_NMS_THRESHOLDS:
            raw_counts = [
                count_from_candidates(
                    record["scores"],
                    record["boxes"],
                    similarity_threshold=similarity_threshold,
                    nms_threshold=nms_threshold,
                )
                for record in records
            ]
            gt_counts = [record["gt_count"] for record in records]
            scale, bias = fit_linear_count_calibration(
                raw_counts,
                gt_counts,
                ridge=CFG.COUNT_CALIBRATION_RIDGE,
            )
            calibrated_counts = [
                max(0, round(scale * float(raw_count) + bias))
                for raw_count in raw_counts
            ]

            raw_mae, raw_rmse = compute_mae_rmse(raw_counts, gt_counts)
            val_mae, val_rmse = compute_mae_rmse(calibrated_counts, gt_counts)

            if similarity_threshold == CFG.SIMILARITY_THRESHOLD and nms_threshold == CFG.NMS_THRESHOLD:
                raw_baseline = {
                    "raw_val_mae": raw_mae,
                    "raw_val_rmse": raw_rmse,
                }

            candidate = {
                "similarity_threshold": float(similarity_threshold),
                "nms_threshold": float(nms_threshold),
                "scale": float(scale),
                "bias": float(bias),
                "val_mae": float(val_mae),
                "val_rmse": float(val_rmse),
                "raw_val_mae": float(raw_mae),
                "raw_val_rmse": float(raw_rmse),
                "val_image_count": len(records),
            }

            if best is None or candidate["val_mae"] < best["val_mae"]:
                best = candidate

    if raw_baseline is not None:
        best["raw_val_mae"] = raw_baseline["raw_val_mae"]
        best["raw_val_rmse"] = raw_baseline["raw_val_rmse"]

    save_count_calibration(CFG.COUNT_CALIBRATION_SAVE_PATH, best)

    print("\nSaved count calibration:", CFG.COUNT_CALIBRATION_SAVE_PATH)
    print(f"Best threshold       : {best['similarity_threshold']:.2f}")
    print(f"Best NMS threshold   : {best['nms_threshold']:.2f}")
    print(f"Linear scale/bias    : scale={best['scale']:.4f}, bias={best['bias']:.4f}")
    print(f"Raw val MAE/RMSE     : {best['raw_val_mae']:.2f} / {best['raw_val_rmse']:.2f}")
    print(f"Calibrated MAE/RMSE  : {best['val_mae']:.2f} / {best['val_rmse']:.2f}")


if __name__ == "__main__":
    main()
