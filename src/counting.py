import os

import numpy as np

from src.config import CFG
from src.feature_extractor import compute_features
from src.model import PCAWhiteningModel
from src.utils import non_max_suppression, sliding_window
from src.vector_search import FAISSSearchEngine


def annotation_boxes_to_rectangles(boxes, image_shape):
    rectangles = []
    img_h, img_w = image_shape[:2]

    for box in boxes:
        x_coords = [pt[0] for pt in box]
        y_coords = [pt[1] for pt in box]
        xmin, xmax = max(0, int(min(x_coords))), int(max(x_coords))
        ymin, ymax = max(0, int(min(y_coords))), int(max(y_coords))
        xmax, ymax = min(img_w, xmax), min(img_h, ymax)

        if xmax > xmin and ymax > ymin:
            rectangles.append([xmin, ymin, xmax, ymax])

    return rectangles


def build_detection_candidates(img, model, exemplar_rectangles):
    query_feats = []
    win_widths = []
    win_heights = []

    for xmin, ymin, xmax, ymax in exemplar_rectangles:
        patch = img[ymin:ymax, xmin:xmax]
        query_feats.append(compute_features(patch))
        win_widths.append(xmax - xmin)
        win_heights.append(ymax - ymin)

    if not query_feats:
        return np.empty((0,), dtype=np.float32), []

    query_vectors = model.transform(query_feats)
    query_vector = np.mean(query_vectors, axis=0, keepdims=True)
    query_vector = PCAWhiteningModel._l2_normalize(query_vector)

    base_w = int(np.mean(win_widths))
    base_h = int(np.mean(win_heights))

    all_patches = []
    all_boxes = []

    for scale in CFG.WINDOW_SCALES:
        win_w = int(base_w * scale)
        win_h = int(base_h * scale)

        if win_w < 10 or win_h < 10 or win_w > img.shape[1] or win_h > img.shape[0]:
            continue

        step_size = max(CFG.MIN_SLIDING_STEP, min(win_w, win_h) // CFG.SLIDING_STEP_DIVISOR)
        patches, boxes = sliding_window(img, (win_w, win_h), step_size)
        all_patches.extend(patches)
        all_boxes.extend(boxes)

    if not all_patches:
        return np.empty((0,), dtype=np.float32), []

    db_feats = [compute_features(patch) for patch in all_patches]
    db_vectors = model.transform(db_feats)

    searcher = FAISSSearchEngine(dimension=model.embedding_dim, metric='cosine')
    searcher.build_database(db_vectors)
    retrieval_top_k = min(CFG.FAISS_TOP_K, len(db_vectors))
    scores, indices = searcher.search_exemplar(query_vector, top_k=retrieval_top_k)

    valid_pairs = [(float(score), int(idx)) for score, idx in zip(scores, indices) if idx >= 0]
    candidate_scores = np.asarray([score for score, _ in valid_pairs], dtype=np.float32)
    candidate_boxes = [all_boxes[idx] for _, idx in valid_pairs]
    return candidate_scores, candidate_boxes


def count_from_candidates(scores, boxes, similarity_threshold=None, nms_threshold=None):
    if similarity_threshold is None:
        similarity_threshold = CFG.SIMILARITY_THRESHOLD
    if nms_threshold is None:
        nms_threshold = CFG.NMS_THRESHOLD

    if len(scores) == 0 or len(boxes) == 0:
        return 0

    candidate_pairs = [
        (float(score), idx)
        for idx, score in enumerate(scores)
        if float(score) > similarity_threshold
    ]

    if not candidate_pairs:
        fallback_k = min(50, len(scores))
        candidate_pairs = [(float(scores[idx]), idx) for idx in range(fallback_k)]

    candidate_scores = [score for score, _ in candidate_pairs]
    candidate_boxes = [boxes[idx] for _, idx in candidate_pairs]
    keep = non_max_suppression(candidate_boxes, candidate_scores, iou_threshold=nms_threshold)
    return len(keep)


def fit_linear_count_calibration(raw_counts, gt_counts, ridge=1e-3):
    raw_counts = np.asarray(raw_counts, dtype=np.float32)
    gt_counts = np.asarray(gt_counts, dtype=np.float32)

    if raw_counts.size == 0:
        return 1.0, 0.0

    x = np.column_stack([raw_counts, np.ones_like(raw_counts)])
    regularizer = np.eye(2, dtype=np.float32) * ridge
    regularizer[1, 1] = 0.0

    scale, bias = np.linalg.solve(x.T @ x + regularizer, x.T @ gt_counts)
    return float(scale), float(bias)


def apply_count_calibration(raw_count, calibration):
    if calibration is None:
        return int(raw_count)

    calibrated = calibration["scale"] * float(raw_count) + calibration["bias"]
    return int(max(0, round(calibrated)))


def save_count_calibration(path, calibration):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        similarity_threshold=np.array([calibration["similarity_threshold"]], dtype=np.float32),
        nms_threshold=np.array([calibration["nms_threshold"]], dtype=np.float32),
        scale=np.array([calibration["scale"]], dtype=np.float32),
        bias=np.array([calibration["bias"]], dtype=np.float32),
        val_mae=np.array([calibration["val_mae"]], dtype=np.float32),
        val_rmse=np.array([calibration["val_rmse"]], dtype=np.float32),
        raw_val_mae=np.array([calibration["raw_val_mae"]], dtype=np.float32),
        raw_val_rmse=np.array([calibration["raw_val_rmse"]], dtype=np.float32),
        val_image_count=np.array([calibration["val_image_count"]], dtype=np.int32),
    )


def load_count_calibration(path):
    if not os.path.exists(path):
        return None

    with np.load(path) as data:
        return {
            "similarity_threshold": float(data["similarity_threshold"][0]),
            "nms_threshold": float(data["nms_threshold"][0]),
            "scale": float(data["scale"][0]),
            "bias": float(data["bias"][0]),
            "val_mae": float(data["val_mae"][0]),
            "val_rmse": float(data["val_rmse"][0]),
            "raw_val_mae": float(data["raw_val_mae"][0]),
            "raw_val_rmse": float(data["raw_val_rmse"][0]),
            "val_image_count": int(data["val_image_count"][0]),
        }


def predict_count_from_annotation(img, model, annotation, calibration=None):
    exemplar_rectangles = annotation_boxes_to_rectangles(
        annotation.get("box_examples_coordinates", []),
        img.shape,
    )
    scores, boxes = build_detection_candidates(img, model, exemplar_rectangles)

    if calibration is None:
        raw_count = count_from_candidates(scores, boxes)
        return raw_count, raw_count

    raw_count = count_from_candidates(
        scores,
        boxes,
        similarity_threshold=calibration["similarity_threshold"],
        nms_threshold=calibration["nms_threshold"],
    )
    return apply_count_calibration(raw_count, calibration), raw_count
