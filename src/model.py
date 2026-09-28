"""SVM Model Module for Named Entity Recognition.

Provides feature vectorization via DictVectorizer, safe sparse matrix casting
for 32-bit LibLinear compatibility, LinearSVC training, hyperparameter tuning,
and feature importance inspection.
"""

import time
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC


def ensure_int32_indices(matrix: sp.spmatrix) -> sp.spmatrix:
    """Ensure sparse matrix indices and indptr are int32 for LibLinear compatibility."""
    if hasattr(matrix, 'indices') and matrix.indices.dtype != np.int32:
        matrix.indices = matrix.indices.astype(np.int32)
    if hasattr(matrix, 'indptr') and matrix.indptr.dtype != np.int32:
        matrix.indptr = matrix.indptr.astype(np.int32)
    return matrix


class SVMNERModel:
    """End-to-End SVM Model for Token-Level Named Entity Recognition."""

    def __init__(
        self,
        C: float = 0.5,
        max_iter: int = 2000,
        random_state: int = 42,
        dual: bool = False
    ):
        """Initialize the SVM NER model.
        
        Args:
            C: Regularization parameter. Smaller values specify stronger regularization.
            max_iter: Maximum number of iterations for the solver.
            random_state: Seed for reproducibility.
            dual: Formulate the dual or primal problem. dual=False preferred when n_samples > n_features.
        """
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.dual = dual
        
        self.vectorizer = DictVectorizer(sparse=True)
        self.classifier = LinearSVC(
            C=self.C,
            max_iter=self.max_iter,
            random_state=self.random_state,
            dual=self.dual
        )
        self.is_fitted = False
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X_dicts: List[Dict[str, Any]], y: List[str]) -> "SVMNERModel":
        """Fit the vectorizer and LinearSVC model on training data.
        
        Args:
            X_dicts: List of token feature dictionaries.
            y: List of true token string labels.
            
        Returns:
            self
        """
        print(f"[SVMNERModel] Fitting DictVectorizer on {len(X_dicts)} tokens...")
        t0 = time.time()
        X_sparse = self.vectorizer.fit_transform(X_dicts)
        X_sparse = ensure_int32_indices(X_sparse)
        vec_time = time.time() - t0
        print(f"[SVMNERModel] Vectorization complete in {vec_time:.2f}s. Vocabulary size: {X_sparse.shape[1]:,}")

        print(f"[SVMNERModel] Training LinearSVC (C={self.C}, max_iter={self.max_iter})...")
        t0 = time.time()
        self.classifier.fit(X_sparse, y)
        train_time = time.time() - t0
        self.is_fitted = True
        self.classes_ = self.classifier.classes_
        print(f"[SVMNERModel] Training completed in {train_time:.2f}s across {len(self.classes_)} classes.")
        return self

    def predict(self, X_dicts: List[Dict[str, Any]]) -> np.ndarray:
        """Predict entity labels for token feature dictionaries.
        
        Args:
            X_dicts: List of token feature dictionaries.
            
        Returns:
            1D array of predicted string labels.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict.")
        X_sparse = self.vectorizer.transform(X_dicts)
        X_sparse = ensure_int32_indices(X_sparse)
        return self.classifier.predict(X_sparse)

    def tune_hyperparameters(
        self,
        X_train_dicts: List[Dict[str, Any]],
        y_train: List[str],
        X_val_dicts: List[Dict[str, Any]],
        y_val: List[str],
        c_candidates: List[float] = [0.05, 0.1, 0.5, 1.0]
    ) -> Dict[str, Any]:
        """Tune regularization parameter C using validation set Macro and Weighted F1.
        
        Args:
            X_train_dicts: Training feature dictionaries.
            y_train: Training labels.
            X_val_dicts: Validation feature dictionaries.
            y_val: Validation labels.
            c_candidates: List of C values to evaluate.
            
        Returns:
            Dictionary with tuning history and selected best C.
        """
        print("[SVMNERModel] Starting hyperparameter search for C...")
        X_train_sp = self.vectorizer.fit_transform(X_train_dicts)
        X_train_sp = ensure_int32_indices(X_train_sp)
        X_val_sp = self.vectorizer.transform(X_val_dicts)
        X_val_sp = ensure_int32_indices(X_val_sp)

        best_c = self.C
        best_macro_f1 = -1.0
        tuning_results = []

        for c_val in c_candidates:
            t0 = time.time()
            clf = LinearSVC(
                C=c_val,
                max_iter=self.max_iter,
                random_state=self.random_state,
                dual=self.dual
            )
            clf.fit(X_train_sp, y_train)
            train_duration = time.time() - t0

            val_preds = clf.predict(X_val_sp)
            macro_f1 = f1_score(y_val, val_preds, average='macro')
            weighted_f1 = f1_score(y_val, val_preds, average='weighted')

            tuning_results.append({
                'C': c_val,
                'macro_f1': macro_f1,
                'weighted_f1': weighted_f1,
                'training_time_s': train_duration
            })
            print(f"  Candidate C={c_val:<5} -> Val Macro F1: {macro_f1:.4f} | Weighted F1: {weighted_f1:.4f} | Time: {train_duration:.2f}s")

            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                best_c = c_val

        print(f"[SVMNERModel] Optimal hyperparameter selected: C={best_c} (Macro F1={best_macro_f1:.4f})")
        self.C = best_c
        self.classifier = LinearSVC(
            C=self.C,
            max_iter=self.max_iter,
            random_state=self.random_state,
            dual=self.dual
        )
        self.classifier.fit(X_train_sp, y_train)
        self.is_fitted = True
        self.classes_ = self.classifier.classes_

        return {
            'best_C': best_c,
            'best_macro_f1': best_macro_f1,
            'tuning_history': tuning_results
        }

    def get_top_features_per_class(self, top_n: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """Inspect the most influential features per entity class."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted to extract top features.")
            
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        top_features = {}
        for class_idx, class_name in enumerate(self.classifier.classes_):
            coefs = self.classifier.coef_[class_idx]
            top_positive_indices = np.argsort(coefs)[-top_n:][::-1]
            top_features[class_name] = [
                (feature_names[i], float(coefs[i])) for i in top_positive_indices
            ]
        return top_features

    def save(self, filepath: str) -> None:
        """Serialize model and vectorizer to disk."""
        joblib.dump({
            'vectorizer': self.vectorizer,
            'classifier': self.classifier,
            'C': self.C,
            'classes_': self.classes_
        }, filepath)
        print(f"[SVMNERModel] Model saved successfully to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "SVMNERModel":
        """Load serialized model from disk."""
        data = joblib.load(filepath)
        instance = cls(C=data['C'])
        instance.vectorizer = data['vectorizer']
        instance.classifier = data['classifier']
        instance.classes_ = data['classes_']
        instance.is_fitted = True
        print(f"[SVMNERModel] Model loaded successfully from {filepath}")
        return instance
