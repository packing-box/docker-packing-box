# -*- coding: UTF-8 -*-
import numpy as np

from pboxllm import LLMBackend, FeatureFormatter, PromptStrategy


__all__ = ["LLMClassifier"]


class LLMClassifier:
    """LLM-based packing classifier using llama-cpp-python (zero-shot).

    This classifier mimics the sklearn fit/predict interface expected by pbox.
    It uses a local large language model (GGUF format, via llama-cpp-python)
    to determine whether a PE executable is packed, based on a human-readable
    text representation of selected binary features.

    No gradient-based training occurs. ``fit`` loads the model into memory;
    ``predict`` runs zero-shot inference for each sample.

    Attributes
    ----------
    classes_ : np.ndarray of shape (2,)
        ``[0, 1]`` — 0 is not-packed, 1 is packed.

    backend_ : LLMBackend
        Loaded LLM backend instance (available after ``fit``).

    Parameters
    ----------
    model_file : str
        Filename of the GGUF model stored in ``~/.cache/pboxllm/models/``
        (e.g. ``mistral-7b-instruct-v0.2.Q4_K_M.gguf``).

    model_repo : str, default="TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
        HuggingFace repository id used to download the model on first use.

    prompt_file : str, default="zero_shot_binary.txt"
        Filename of the prompt template in ``~/.packing-box/data/prompt/``.

    feature_names : list of str, default=["entropy", "entropy_code_section", "entropy_section_with_ep"]
        Features from the dataset DataFrame passed to the LLM. Must match column
        names produced by the pbox feature pipeline.

    n_ctx : int, default=2048
        LLM context window size in tokens.

    n_threads : int, default=4
        Number of CPU threads for inference.

    max_tokens : int, default=64
        Maximum tokens generated per prediction.

    References
    ----------
    llama-cpp-python: https://github.com/abetlen/llama-cpp-python

    Examples
    --------
    >>> from pbox.core.model.algorithm.algoLLM import LLMClassifier
    >>> clf = LLMClassifier(model_file="mistral-7b-instruct-v0.2.Q4_K_M.gguf")
    >>> clf.fit(X_train, y_train)
    LLMClassifier(...)
    >>> clf.predict(X_test[:3])
    array([0, 1, 0])
    """

    classes_ = np.array([0, 1])

    def __init__(
        self,
        model_file,
        model_repo="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        prompt_file="zero_shot_binary.txt",
        feature_names=None,
        n_ctx=2048,
        n_threads=4,
        max_tokens=64,
    ):
        self.model_file = model_file
        self.model_repo = model_repo
        self.prompt_file = prompt_file
        self.feature_names = feature_names or ["entropy", "entropy_code_section", "entropy_section_with_ep"]
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.max_tokens = max_tokens

    # ------------------------------------------------------------------
    # pbox / sklearn-compatible interface
    # ------------------------------------------------------------------

    def fit(self, X, y=None):
        """Load the LLM model into memory.

        No statistical training is performed. This method downloads the GGUF
        model if absent from the cache and loads it via llama-cpp-python so
        that ``predict`` can run without delay.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix (not used for model fitting).

        y : array-like of shape (n_samples,), optional
            Target labels (not used for model fitting).

        Returns
        -------
        self : LLMClassifier
            Fitted estimator.
        """
        self.backend_   = LLMBackend(self.model_file, self.model_repo, self.n_ctx, self.n_threads)
        self.formatter_ = FeatureFormatter(self.feature_names)
        self.strategy_  = PromptStrategy(self.prompt_file)
        self.backend_.load()
        return self

    def predict(self, X):
        """Classify each sample using zero-shot LLM inference.

        For each row in ``X``, the selected features are formatted as a
        human-readable text block, inserted into the prompt template, and sent
        to the local LLM. The response is parsed into a label following pbox
        conventions: 0 = not-packed, 1 = packed, -1 = uncertain (NOT_LABELLED,
        excluded from metrics).

        Parameters
        ----------
        X : array-like or pandas.DataFrame of shape (n_samples, n_features)
            Feature matrix. Column names must include all entries of
            ``feature_names``.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted labels (0, 1, or -1).
        """
        if not hasattr(self, "backend_"):
            raise RuntimeError("LLMClassifier must be fitted before calling predict.")
        results = []
        for i in range(len(X)):
            row    = X.iloc[i] if hasattr(X, "iloc") else X[i]
            text   = self.formatter_.format(row)
            prompt = self.strategy_.build_prompt(text)
            raw    = self.backend_.generate(prompt, max_tokens=self.max_tokens)
            results.append(self.strategy_.parse(raw))
        return np.array(results, dtype=int)

    def predict_proba(self, X):
        """Return class probability estimates.

        Probabilities are derived directly from ``predict``: 1.0 for the
        predicted class, 0.0 for the other. Uncertain samples (-1) are not
        handled specially here; callers should filter them out beforehand.

        Parameters
        ----------
        X : array-like or pandas.DataFrame of shape (n_samples, n_features)

        Returns
        -------
        proba : np.ndarray of shape (n_samples, 2)
            ``[:, 0]`` = probability of not-packed, ``[:, 1]`` = probability
            of packed.
        """
        
        # TODO
        preds = self.predict(X)
        return np.vstack([1 - preds, preds]).T.astype(float)
