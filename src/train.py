import os
import sys
import json
import numpy as np
from tqdm import tqdm

if __package__ is None or __package__ == "":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import CFG
from src.model import PCAWhiteningModel
from src.utils import configure_utf8_output

def load_split_feature_groups(split_name):
    with open(CFG.SPLIT_PATH, 'r', encoding='utf-8') as f:
        splits = json.load(f)

    image_names = splits.get(split_name, [])
    groups = []
    missing = 0

    for img_name in tqdm(image_names, desc=f"Loading {split_name} features"):
        feat_path = os.path.join(CFG.FEATURES_DIR, img_name.replace('.jpg', '.npy'))
        if not os.path.exists(feat_path):
            missing += 1
            continue

        features = np.load(feat_path).astype(np.float32)
        if features.ndim == 1:
            features = features.reshape(1, -1)

        if features.shape[1] != CFG.FEATURE_DIM:
            raise ValueError(
                f"Feature dim sai ở {feat_path}: {features.shape[1]} != {CFG.FEATURE_DIM}. "
                "Hãy chạy lại src/feature_extractor.py."
            )

        groups.append(features)

    if not groups:
        raise ValueError(
            f"Không tìm thấy feature nào cho split '{split_name}'. "
            f"Hãy chạy python src/feature_extractor.py trước. Missing files: {missing}"
        )

    return groups, missing

def flatten_feature_groups(groups):
    if not groups:
        return np.empty((0, 0), dtype=np.float32)
    return np.vstack(groups)

def load_split_features(split_name):
    groups, missing = load_split_feature_groups(split_name)
    features = flatten_feature_groups(groups)
    return features, missing

def maybe_subsample(features):
    max_samples = CFG.PCA_MAX_TRAIN_SAMPLES
    if max_samples is None or len(features) <= max_samples:
        return features

    rng = np.random.default_rng(CFG.RANDOM_SEED)
    indices = rng.choice(len(features), size=max_samples, replace=False)
    return features[indices]

def validate_matching(model, val_groups):
    embeddings = []
    labels = []

    for group_id, features in enumerate(val_groups):
        if len(features) < 2:
            continue
        group_embeddings = model.transform(features)
        embeddings.append(group_embeddings)
        labels.extend([group_id] * len(group_embeddings))

    if not embeddings:
        return {
            "loss": float("inf"),
            "accuracy": 0.0,
            "pos_similarity": 0.0,
            "hard_neg_similarity": 0.0,
            "num_queries": 0,
        }

    embeddings = np.vstack(embeddings).astype(np.float32)
    labels = np.asarray(labels)
    similarity = embeddings @ embeddings.T

    losses = []
    correct = 0
    pos_scores = []
    hard_neg_scores = []

    for i in range(len(embeddings)):
        positive_mask = labels == labels[i]
        positive_mask[i] = False
        negative_mask = labels != labels[i]

        if not np.any(positive_mask) or not np.any(negative_mask):
            continue

        pos_score = float(np.max(similarity[i, positive_mask]))
        hard_neg_score = float(np.max(similarity[i, negative_mask]))
        loss = max(0.0, CFG.PCA_VALIDATION_MARGIN - pos_score + hard_neg_score)

        losses.append(loss)
        pos_scores.append(pos_score)
        hard_neg_scores.append(hard_neg_score)
        if pos_score > hard_neg_score:
            correct += 1

    if not losses:
        return {
            "loss": float("inf"),
            "accuracy": 0.0,
            "pos_similarity": 0.0,
            "hard_neg_similarity": 0.0,
            "num_queries": 0,
        }

    return {
        "loss": float(np.mean(losses)),
        "accuracy": float(correct / len(losses)),
        "pos_similarity": float(np.mean(pos_scores)),
        "hard_neg_similarity": float(np.mean(hard_neg_scores)),
        "num_queries": int(len(losses)),
    }

if __name__ == '__main__':
    configure_utf8_output()
    os.makedirs(CFG.WEIGHTS_DIR, exist_ok=True)

    train_groups, train_missing = load_split_feature_groups('train')
    val_groups, val_missing = load_split_feature_groups('val')
    train_features = flatten_feature_groups(train_groups)
    fit_features = maybe_subsample(train_features)

    print(f"Loaded train features: {train_features.shape}")
    print(f"Missing train feature files: {train_missing}")
    print(f"Missing val feature files: {val_missing}")
    print(f"Fitting PCA/Whitening on: {fit_features.shape}")

    candidate_dims = sorted(set(int(dim) for dim in CFG.PCA_DIM_CANDIDATES))
    max_dim = max(candidate_dims)

    full_model = PCAWhiteningModel(
        n_components=max_dim,
        whitening_epsilon=CFG.PCA_WHITENING_EPS
    )
    full_model.fit(fit_features)

    best_model = None
    best_metrics = None
    all_results = []

    for dim in candidate_dims:
        candidate_model = full_model.truncated(dim)
        metrics = validate_matching(candidate_model, val_groups)
        metrics["pca_dim"] = candidate_model.embedding_dim
        metrics["explained_variance_ratio_sum"] = float(candidate_model.explained_variance_ratio.sum())
        all_results.append(metrics)

        print(
            f"PCA dim {candidate_model.embedding_dim}: "
            f"val_loss={metrics['loss']:.4f}, "
            f"acc={metrics['accuracy']:.4f}, "
            f"pos={metrics['pos_similarity']:.4f}, "
            f"hard_neg={metrics['hard_neg_similarity']:.4f}, "
            f"var={metrics['explained_variance_ratio_sum']:.4f}"
        )

        if best_metrics is None or metrics["loss"] < best_metrics["loss"]:
            best_model = candidate_model
            best_metrics = metrics

    if best_model is None or best_metrics is None:
        raise RuntimeError("Không chọn được best PCA/Whitening model.")

    best_model.save(CFG.BEST_MODEL_SAVE_PATH)

    np.savez_compressed(
        CFG.BEST_CHECKPOINT_PATH,
        model_path=np.array([CFG.BEST_MODEL_SAVE_PATH]),
        feature_dim=np.array([CFG.FEATURE_DIM], dtype=np.int32),
        embedding_dim=np.array([best_model.embedding_dim], dtype=np.int32),
        pca_dim=np.array([best_model.embedding_dim], dtype=np.int32),
        candidate_dims=np.array(candidate_dims, dtype=np.int32),
        train_feature_count=np.array([len(train_features)], dtype=np.int32),
        fit_feature_count=np.array([len(fit_features)], dtype=np.int32),
        val_feature_count=np.array([sum(len(group) for group in val_groups)], dtype=np.int32),
        best_val_loss=np.array([best_metrics["loss"]], dtype=np.float32),
        best_val_accuracy=np.array([best_metrics["accuracy"]], dtype=np.float32),
        best_pos_similarity=np.array([best_metrics["pos_similarity"]], dtype=np.float32),
        best_hard_neg_similarity=np.array([best_metrics["hard_neg_similarity"]], dtype=np.float32),
        best_num_queries=np.array([best_metrics["num_queries"]], dtype=np.int32),
        results=np.array([json.dumps(all_results, ensure_ascii=False)]),
        explained_variance_ratio=best_model.explained_variance_ratio,
    )

    print(f"Saved best PCA/Whitening model: {CFG.BEST_MODEL_SAVE_PATH}")
    print(f"Saved best checkpoint metadata: {CFG.BEST_CHECKPOINT_PATH}")
    print(f"Best PCA dim: {best_model.embedding_dim}")
    print(f"Best val loss: {best_metrics['loss']:.4f}")
    print(f"Best val accuracy: {best_metrics['accuracy']:.4f}")
    print(f"Explained variance ratio sum: {best_model.explained_variance_ratio.sum():.4f}")
