"""
AI Requirements Intelligence Platform (ARIP) - Model Pipeline Module
Defines canonical deployable pipeline structures for Joblib serialization.
"""

import numpy as np
import scipy.sparse as sp


class WordCharTfidfPipeline:
    """
    Deployable pipeline object combining Word (1,2) and Character (3,5) TF-IDF vectorizers
    with a classification model (e.g., Logistic Regression or Calibrated LinearSVM).
    """

    def __init__(self, vec_word, vec_char, clf, threshold=0.50):
        self.vec_word = vec_word
        self.vec_char = vec_char
        self.clf = clf
        self.threshold = threshold
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X_raw):
        if isinstance(X_raw, str):
            X_raw = [X_raw]
        X_w = self.vec_word.transform(X_raw)
        X_c = self.vec_char.transform(X_raw)
        X_comb = sp.hstack([X_w, X_c], format="csr")
        if hasattr(self.clf, "predict_proba"):
            return self.clf.predict_proba(X_comb)
        elif hasattr(self.clf, "decision_function"):
            df = self.clf.decision_function(X_comb)
            probs_pos = 1 / (1 + np.exp(-df))
            return np.vstack([1 - probs_pos, probs_pos]).T
        else:
            preds = self.clf.predict(X_comb)
            return np.vstack([1 - preds, preds]).T

    def predict(self, X_raw):
        probs = self.predict_proba(X_raw)[:, 1]
        return (probs >= self.threshold).astype(int)
