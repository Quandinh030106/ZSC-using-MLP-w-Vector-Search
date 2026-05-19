import numpy as np

class PCAWhiteningModel:
    def __init__(self, n_components=256, whitening_epsilon=1e-5):
        self.n_components = n_components
        self.whitening_epsilon = whitening_epsilon
        self.mean = None
        self.components = None
        self.whitening_scale = None
        self.explained_variance = None
        self.explained_variance_ratio = None

    @property
    def embedding_dim(self):
        if self.components is None:
            return self.n_components
        return self.components.shape[0]

    def fit(self, features):
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2:
            raise ValueError("features phải có shape (num_samples, feature_dim).")
        if features.shape[0] < 2:
            raise ValueError("Cần ít nhất 2 feature vectors để fit PCA/Whitening.")

        self.mean = features.mean(axis=0, keepdims=True).astype(np.float32)
        centered = features - self.mean

        _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
        max_components = min(self.n_components, vt.shape[0], features.shape[1])

        self.components = vt[:max_components].astype(np.float32)
        self.explained_variance = ((singular_values[:max_components] ** 2) / (features.shape[0] - 1)).astype(np.float32)

        total_variance = float(np.sum(singular_values ** 2) / (features.shape[0] - 1))
        if total_variance > 0:
            self.explained_variance_ratio = (self.explained_variance / total_variance).astype(np.float32)
        else:
            self.explained_variance_ratio = np.zeros_like(self.explained_variance, dtype=np.float32)

        self.whitening_scale = (1.0 / np.sqrt(self.explained_variance + self.whitening_epsilon)).astype(np.float32)
        return self

    def transform(self, features):
        if self.mean is None or self.components is None or self.whitening_scale is None:
            raise RuntimeError("PCAWhiteningModel chưa được fit hoặc load.")

        features = np.asarray(features, dtype=np.float32)
        if features.ndim == 1:
            features = features.reshape(1, -1)

        projected = (features - self.mean) @ self.components.T
        whitened = projected * self.whitening_scale
        return self._l2_normalize(whitened.astype(np.float32))

    def save(self, path):
        if (
            self.mean is None
            or self.components is None
            or self.whitening_scale is None
            or self.explained_variance is None
            or self.explained_variance_ratio is None
        ):
            raise RuntimeError("Không thể save model chưa được fit.")

        np.savez_compressed(
            path,
            n_components=np.array([self.n_components], dtype=np.int32),
            whitening_epsilon=np.array([self.whitening_epsilon], dtype=np.float32),
            mean=self.mean.astype(np.float32),
            components=self.components.astype(np.float32),
            whitening_scale=self.whitening_scale.astype(np.float32),
            explained_variance=self.explained_variance.astype(np.float32),
            explained_variance_ratio=self.explained_variance_ratio.astype(np.float32),
        )

    def truncated(self, n_components):
        if (
            self.mean is None
            or self.components is None
            or self.whitening_scale is None
            or self.explained_variance is None
            or self.explained_variance_ratio is None
        ):
            raise RuntimeError("Không thể truncate model chưa được fit.")

        n_components = min(n_components, self.components.shape[0])
        model = PCAWhiteningModel(
            n_components=n_components,
            whitening_epsilon=self.whitening_epsilon
        )
        model.mean = self.mean.copy()
        model.components = self.components[:n_components].copy()
        model.whitening_scale = self.whitening_scale[:n_components].copy()
        model.explained_variance = self.explained_variance[:n_components].copy()
        model.explained_variance_ratio = self.explained_variance_ratio[:n_components].copy()
        return model

    @classmethod
    def load(cls, path):
        with np.load(path) as data:
            model = cls(
                n_components=int(data["n_components"][0]),
                whitening_epsilon=float(data["whitening_epsilon"][0])
            )
            model.mean = data["mean"].astype(np.float32)
            model.components = data["components"].astype(np.float32)
            model.whitening_scale = data["whitening_scale"].astype(np.float32)
            model.explained_variance = data["explained_variance"].astype(np.float32)
            model.explained_variance_ratio = data["explained_variance_ratio"].astype(np.float32)
        return model

    @staticmethod
    def _l2_normalize(vectors):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / (norms + 1e-8)

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    dummy_features = rng.normal(size=(32, 2068)).astype(np.float32)
    model = PCAWhiteningModel(n_components=16).fit(dummy_features)
    embeddings = model.transform(dummy_features[:4])
    print("Embedding Shape:", embeddings.shape)
