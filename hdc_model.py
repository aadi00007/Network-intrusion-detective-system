import numpy as np
from numpy.linalg import norm
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted


class HDClassifier(BaseEstimator, ClassifierMixin):
    """Hyperdimensional Computing classifier that mimics scikit-learn's API.

    This implementation follows the NSL-KDD HDC workflow:
    1. Assign a random base hypervector per feature.
    2. Generate value-specific hypervectors for binned feature ranges.
    3. Encode each sample by XOR-ing base/value hypervectors and aggregating bits.
    4. Build class representative hypervectors and classify with cosine similarity.
    """

    def __init__(self, dim: int = 10000, n_bins: int = 10, n_iter: int = 50, lr: float = 0.1, random_state=None):
        self.dim = dim
        self.n_bins = n_bins
        self.n_iter = n_iter
        self.lr = lr
        self.random_state = random_state

    def _init_rng(self):
        self.rng_ = np.random.default_rng(self.random_state)

    @staticmethod
    def _ensure_dense(X):
        if sparse.issparse(X):
            return X.toarray()
        return np.asarray(X)

    def fit(self, X, y):
        X = self._ensure_dense(X)
        self._init_rng()

        self.label_encoder_ = LabelEncoder()
        y_enc = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_

        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        self.feature_mins_ = X.min(axis=0)
        self.feature_maxs_ = X.max(axis=0)

        self.base_hv_ = self.rng_.integers(0, 2, size=(n_features, self.dim), dtype=np.int8)
        self.value_hv_ = np.zeros((n_features, self.n_bins, self.dim), dtype=np.int8)

        for j in range(n_features):
            first = self.rng_.integers(0, 2, size=self.dim, dtype=np.int8)
            self.value_hv_[j, 0] = first
            for b in range(1, self.n_bins):
                hv = first.copy()
                n_flip = max(1, self.dim // (2 * self.n_bins))
                idx = self.rng_.choice(self.dim, size=n_flip, replace=False)
                hv[idx] = 1 - hv[idx]
                self.value_hv_[j, b] = hv

        H = self._encode_matrix(X)

        n_classes = len(self.classes_)
        reps = np.zeros((n_classes, self.dim), dtype=np.float32)
        for c in range(n_classes):
            class_mask = y_enc == c
            if not np.any(class_mask):
                continue
            reps[c] = H[class_mask].mean(axis=0)
        reps = (reps >= 0.5).astype(np.float32)
        self.reps_ = reps

        for _ in range(self.n_iter):
            preds = self._predict_from_hv(H)
            mis_idx = np.where(preds != y_enc)[0]
            if mis_idx.size == 0:
                break
            for i in mis_idx:
                hi = H[i].astype(np.float32)
                ci = y_enc[i]
                pi = preds[i]
                self.reps_[pi] -= self.lr * hi
                self.reps_[ci] += self.lr * hi

        self.reps_ = (self.reps_ >= 0).astype(np.float32)
        return self

    def _bin_index(self, value, feature_idx: int) -> int:
        """Convert a feature value to a bin index."""
        lo = self.feature_mins_[feature_idx]
        hi = self.feature_maxs_[feature_idx]
        if hi == lo:
            return 0
        # Ensure value is scalar
        if hasattr(value, '__len__') and not isinstance(value, str):
            value = float(value[0]) if len(value) > 0 else 0.0
        else:
            value = float(value)
        t = (value - lo) / (hi - lo)
        t = np.clip(t, 0.0, 1.0)
        b = int(t * self.n_bins)
        return max(0, min(self.n_bins - 1, b))

    def _encode_matrix(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        n_samples, n_features = X.shape
        H = np.zeros((n_samples, self.dim), dtype=np.int16)
        threshold = n_features / 2.0
        
        # Vectorized encoding for better performance
        for i in range(n_samples):
            hv_accum = np.zeros(self.dim, dtype=np.int16)
            # Pre-compute all bin indices for this sample
            bin_indices = np.array([self._bin_index(X[i, j], j) for j in range(n_features)])
            # Vectorized XOR and sum
            for j in range(n_features):
                hv_feat = np.bitwise_xor(self.base_hv_[j], self.value_hv_[j, bin_indices[j]]).astype(np.int16)
                hv_accum += hv_feat
            H[i] = (hv_accum > threshold).astype(np.int16)
            
            # Progress indicator for large datasets
            if (i + 1) % 10000 == 0:
                print(f"Encoded {i + 1}/{n_samples} samples...", flush=True)
        return H

    def _predict_from_hv(self, H: np.ndarray) -> np.ndarray:
        Hf = H.astype(np.float32)
        reps = self.reps_
        h_norm = np.maximum(norm(Hf, axis=1, keepdims=True), 1e-9)
        r_norm = np.maximum(norm(reps, axis=1), 1e-9)
        sims = (Hf @ reps.T) / (h_norm * r_norm)
        return sims.argmax(axis=1)

    def predict(self, X):
        check_is_fitted(self, ["reps_", "label_encoder_"])
        X = self._ensure_dense(X)
        H = self._encode_matrix(X)
        pred_idx = self._predict_from_hv(H)
        return self.label_encoder_.inverse_transform(pred_idx)

    def predict_proba(self, X):
        check_is_fitted(self, ["reps_", "label_encoder_"])
        X = self._ensure_dense(X)
        H = self._encode_matrix(X)
        Hf = H.astype(np.float32)
        reps = self.reps_
        h_norm = np.maximum(norm(Hf, axis=1, keepdims=True), 1e-9)
        r_norm = np.maximum(norm(reps, axis=1), 1e-9)
        sims = (Hf @ reps.T) / (h_norm * r_norm)
        exps = np.exp(sims - sims.max(axis=1, keepdims=True))
        probs = exps / np.maximum(exps.sum(axis=1, keepdims=True), 1e-9)
        return probs

