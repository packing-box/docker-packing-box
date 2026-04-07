# -*- coding: UTF-8 -*-
import numpy as np
from sklearn.base import BaseEstimator

from pboxllm import LLMBackend, FeatureFormatter, PromptStrategy


__all__ = ["LLMClassifier"]


class LLMClassifier(BaseEstimator):
    """LLM-based packing classifier using llama-cpp-python (zero-shot).

    This classifier mimics the sklearn fit/predict interface expected by pbox.
    It uses a local large language model (GGUF format, via llama-cpp-python)
    to determine whether a PE executable is packed, based on a human-readable
    text representation of selected binary features.

    No gradient-based training occurs. ``fit`` loads the model into memory;
    ``predict`` runs zero-shot inference for each sample.
    """

    classes_ = np.array([0, 1])
    _required_params = (
        "model_file",
        "model_repo",
        "prompt_file",
        "feature_names",
        "n_ctx",
        "n_threads",
        "max_tokens",
    )

    def __init__(
        self,
        model_file=None,
        model_repo=None,
        prompt_file=None,
        feature_names=None,
        n_ctx=None,
        n_threads=None,
        max_tokens=None,
    ):
        self.model_file = model_file
        self.model_repo = model_repo
        self.prompt_file = prompt_file
        self.feature_names = feature_names
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.max_tokens = max_tokens

    def __sklearn_is_fitted__(self):
        return hasattr(self, "backend_")

    def fit(self, X, y=None):
        missing = [p for p in self._required_params if getattr(self, p) in [None, ""]]
        if missing:
            raise ValueError(
                "[LLMClassifier] Missing required parameter(s): "
                + ", ".join(missing)
                + ". Configure them in algorithms.yml (LLM category)."
            )
        self.backend_ = LLMBackend(self.model_file, self.model_repo, self.n_ctx, self.n_threads)
        self.formatter_ = FeatureFormatter(self.feature_names)
        self.strategy_ = PromptStrategy(self.prompt_file)
        self.backend_.load()
        return self

    def predict(self, X):
        if not hasattr(self, "backend_"):
            raise RuntimeError("LLMClassifier must be fitted before calling predict.")
        results = []
        for i in range(len(X)):
            row = X.iloc[i] if hasattr(X, "iloc") else X[i]
            text = self.formatter_.format(row)
            prompt = self.strategy_.build_prompt(text)
            raw = self.backend_.generate(prompt, max_tokens=self.max_tokens)
            results.append(self.strategy_.parse(raw))
        return np.array(results, dtype=int)

    def predict_proba(self, X):
        preds = self.predict(X)
        return np.vstack([1 - preds, preds]).T.astype(float)
