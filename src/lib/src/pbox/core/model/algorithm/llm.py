# -*- coding: UTF-8 -*-
import json
import os
import warnings
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator

from pboxllm import LLMBackend, FeatureFormatter, PromptStrategy, proba_from_top_logprobs


__all__ = ["LLMClassifier"]


class LLMClassifier(BaseEstimator):
    """LLM-based packing classifier using llama-cpp-python (in-context / zero-shot).

    This classifier mimics the sklearn fit/predict interface expected by pbox.
    It uses a local large language model (GGUF format, via llama-cpp-python)
    to determine whether a PE executable is packed, based on a human-readable
    text representation of selected binary features.

    No gradient-based training occurs. ``fit`` loads the model into memory and,
    when ``few_shot_count`` is greater than zero, samples a fixed-size set of
    labelled demos either from the feature matrix passed to ``fit`` or from an
    external CSV / pbox dataset supplied via ``few_shot_examples_path``.

    Few-shot scope and leakage (pbox ``Model.train``):
        The final ``pipeline.fit`` call receives only the training split
        (``_train.data`` / ``_train.target``). Few-shot demos are therefore built
        exclusively from training rows. Metrics on the held-out test split do not
        inject test labels or test feature rows into the prompt, which matches the
        usual requirement that demonstration labels stay out of the evaluation
        fold (standard in-context learning evaluation; see e.g. discussions of
        train/test separation for prompt-based classifiers).

        Caveats: (1) If you evaluate on training rows, a demo can coincide with
        the target row by chance (sampling without replacement still leaves other
        overlap risks); prefer reporting generalisation on ``_test``. (2) When
        hyperparameters are chosen via ``GridSearchCV``, pbox fits the search on
        the combined pre-split matrix; inner CV folds may mix partitions
        differently from the outer train/test split — this is a broader tuning
        protocol issue, not specific to the LLM.

    Verbosity: set ``verbose`` > 0 to print the first-sample prompt and raw
    generation (similar to a debug mode); default is silent.
    """

    classes_ = np.array([0, 1])
    _few_shot_seed = 42
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
        temperature=0.0,
        top_p=1.0,
        packed_label="packed",
        not_packed_label="not-packed",
        representation_style="flat",
        few_shot_count=0,
        few_shot_examples_path=None,
        trace_outputs=False,
        trace_dir=None,
        parse_policy="p_packed_fallback",
        on_backend_error="unknown",
        verbose=0,
        backend_factory=None,
        formatter_factory=None,
        strategy_factory=None,
    ):
        self.model_file = model_file
        self.model_repo = model_repo
        self.prompt_file = prompt_file
        self.feature_names = feature_names
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.packed_label = packed_label
        self.not_packed_label = not_packed_label
        self.representation_style = representation_style
        self.few_shot_count = few_shot_count
        self.few_shot_examples_path = few_shot_examples_path
        self.trace_outputs = trace_outputs
        self.trace_dir = trace_dir
        self.parse_policy = parse_policy
        self.on_backend_error = on_backend_error
        self.verbose = int(verbose or 0)
        self.backend_factory = backend_factory
        self.formatter_factory = formatter_factory
        self.strategy_factory = strategy_factory

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
        _backend_cls = self.backend_factory or LLMBackend
        _formatter_cls = self.formatter_factory or FeatureFormatter
        _strategy_cls = self.strategy_factory or PromptStrategy
        self.backend_ = _backend_cls(self.model_file, self.model_repo, self.n_ctx, self.n_threads)
        self.formatter_ = _formatter_cls(self.feature_names, representation_style=self.representation_style)
        self.strategy_ = _strategy_cls(
            self.prompt_file,
            packed_label=self.packed_label,
            not_packed_label=self.not_packed_label,
            parse_policy=self.parse_policy,
        )
        self.strategy_.validate_template(self.few_shot_count)
        self.few_shot_examples_ = self._build_few_shot_examples(X, y)
        self._trace_run_id = None
        self._trace_path = None
        self.backend_.load()
        return self

    def predict(self, X):
        if not hasattr(self, "backend_"):
            raise RuntimeError("LLMClassifier must be fitted before calling predict.")
        results = []
        with self._open_trace_file() as trace_file:
            for i in range(len(X)):
                row = X.iloc[i] if hasattr(X, "iloc") else X[i]
                text = self.formatter_.format(row)
                prompt = self.strategy_.build_prompt(text, few_shot_examples=self.few_shot_examples_)
                prompt, trunc = self._truncate_prompt_to_context(prompt)
                if self.verbose and i == 0:
                    print("\n===== DEBUG PROMPT (first sample) =====\n")
                    print(prompt)
                    print("\n===== END DEBUG PROMPT =====\n")
                backend_error = None
                try:
                    raw = self.backend_.generate(
                        prompt,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                    )
                except Exception as exc:
                    raw = ""
                    backend_error = {"type": type(exc).__name__, "message": str(exc)}
                if self.verbose and i == 0:
                    print("\n===== DEBUG RAW LLM RESPONSE (first sample) =====\n")
                    print(raw)
                    print("\n===== DEBUG RAW LLM RESPONSE REPR (first sample) =====\n")
                    print(repr(raw))
                    print("\n===== END DEBUG RAW LLM RESPONSE =====\n")
                parsed_meta = self.strategy_.parse_with_meta(raw)
                if trunc:
                    parsed_meta = {**parsed_meta, "prompt_truncated": True}
                parsed = int(parsed_meta["label"])
                if backend_error is not None and str(self.on_backend_error).lower() == "unknown":
                    parsed = -1
                    parsed_meta["reason"] = "backend_error"
                self._trace_sample(
                    method="predict",
                    sample_index=i,
                    prompt=prompt,
                    raw_response=raw,
                    parsed_label=int(parsed),
                    parse_meta=parsed_meta,
                    error=backend_error,
                    trace_file=trace_file,
                )
                results.append(parsed)
        return np.array(results, dtype=int)

    def predict_proba(self, X):
        if not hasattr(self, "backend_"):
            raise RuntimeError("LLMClassifier must be fitted before calling predict_proba.")
        probas = []
        for i in range(len(X)):
            out = None
            raw = ""
            backend_error = None
            row = X.iloc[i] if hasattr(X, "iloc") else X[i]
            text = self.formatter_.format(row)
            prompt = self.strategy_.build_prompt(text, few_shot_examples=self.few_shot_examples_)
            prompt, _trunc = self._truncate_prompt_to_context(prompt)
            try:
                out = self.backend_.generate_with_logprobs(
                    prompt,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
                if self.verbose and i == 0:
                    print("\n===== DEBUG RAW LLM RESPONSE PROBA PATH (first sample) =====\n")
                    print(out.get("text", ""))
                    print("\n===== DEBUG RAW LLM RESPONSE PROBA REPR (first sample) =====\n")
                    print(repr(out.get("text", "")))
                    print("\n===== END DEBUG RAW LLM RESPONSE PROBA PATH =====\n")
                proba = proba_from_top_logprobs(out.get("top_logprobs"))
                if proba is None:
                    pred = self.strategy_.parse(out.get("text", ""))
                    if pred == 1:
                        proba = np.array([0.0, 1.0], dtype=float)
                    elif pred == 0:
                        proba = np.array([1.0, 0.0], dtype=float)
                    else:
                        proba = np.array([0.5, 0.5], dtype=float)
            except ValueError:
                # Some GGUF builds don't expose logprobs (logits_all=False).
                # Fallback: generate plain text and map it to packedness.
                try:
                    raw = self.backend_.generate(
                        prompt,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                    )
                except Exception as exc:
                    raw = ""
                    backend_error = {"type": type(exc).__name__, "message": str(exc)}
                parsed_meta = self.strategy_.parse_with_meta(raw)
                pred = parsed_meta["label"]
                if backend_error is not None and str(self.on_backend_error).lower() == "unknown":
                    pred = -1
                if pred == 1:
                    proba = np.array([0.0, 1.0], dtype=float)
                elif pred == 0:
                    proba = np.array([1.0, 0.0], dtype=float)
                else:
                    proba = np.array([0.5, 0.5], dtype=float)
            except Exception as exc:
                backend_error = {"type": type(exc).__name__, "message": str(exc)}
                proba = np.array([0.5, 0.5], dtype=float)
            probas.append(proba)
        return np.vstack(probas)

    def _build_few_shot_examples(self, X, y):
        """Pick up to ``few_shot_count`` labelled rows for in-context demos.

        When ``few_shot_examples_path`` is set, examples are drawn from that
        external source (CSV file path or pbox dataset name) instead of the
        training data. In both cases, balanced class sampling and shuffle apply.
        """
        if not self.few_shot_count:
            return []
        if self.few_shot_examples_path is not None:
            X_pool, y_pool = self._load_examples_from_path(self.few_shot_examples_path)
        else:
            if y is None:
                return []
            X_pool = X
            y_pool = np.asarray(y)
        labels = np.asarray(y_pool)
        if labels.size == 0:
            return []
        total = int(self.few_shot_count or 0)
        if total <= 0:
            return []
        rng = np.random.RandomState(self._few_shot_seed)
        packed_idx = np.where(labels == 1)[0]
        not_packed_idx = np.where(labels == 0)[0]
        selected_idx = []

        if not_packed_idx.size == 0 and packed_idx.size == 0:
            return []

        if not_packed_idx.size == 0:
            take = min(total, packed_idx.size)
            selected_idx = rng.choice(packed_idx, size=take, replace=False).tolist()
        elif packed_idx.size == 0:
            take = min(total, not_packed_idx.size)
            selected_idx = rng.choice(not_packed_idx, size=take, replace=False).tolist()
        else:
            n0_target = total // 2
            n1_target = total - n0_target
            n0 = min(n0_target, not_packed_idx.size)
            n1 = min(n1_target, packed_idx.size)
            if n0 + n1 < total:
                need = total - n0 - n1
                spare0 = not_packed_idx.size - n0
                spare1 = packed_idx.size - n1
                add0 = min(need, spare0)
                n0 += add0
                need -= add0
                if need > 0:
                    n1 += min(need, spare1)
            pick0 = rng.choice(not_packed_idx, size=n0, replace=False) if n0 else np.array([], dtype=int)
            pick1 = rng.choice(packed_idx, size=n1, replace=False) if n1 else np.array([], dtype=int)
            selected_idx = np.concatenate([pick0, pick1]).tolist()
            rng.shuffle(selected_idx)

        examples = []
        for idx in selected_idx:
            row = X_pool.iloc[idx] if hasattr(X_pool, "iloc") else X_pool[idx]
            examples.append({
                "features_text": self.formatter_.format(row),
                "label": int(labels[idx]),
            })
        return examples

    def _load_examples_from_path(self, path_or_name):
        """Return (X_pool, y_pool) from a CSV file path or pbox dataset name.

        Resolution order:
        1. Direct file path (absolute or relative).
        2. ~/.packing-box/datasets/<name>/data.csv  (pbox dataset name).

        The CSV must have a ``label`` column (0/1) and all columns listed in
        ``self.feature_names``. pbox dataset CSVs use a semicolon separator.
        """
        import pandas as pd
        df = self._resolve_examples_dataframe(path_or_name)
        label_col = "label"
        if label_col not in df.columns:
            raise ValueError(
                f"[LLMClassifier] few_shot_examples_path CSV must contain a "
                f"'{label_col}' column. Found: {list(df.columns)}"
            )
        missing_features = [f for f in self.feature_names if f not in df.columns]
        if missing_features:
            raise ValueError(
                f"[LLMClassifier] few_shot_examples_path CSV is missing feature "
                f"columns: {missing_features}"
            )
        y_pool = df[label_col].to_numpy()
        X_pool = df[list(self.feature_names)]
        return X_pool, y_pool

    def _resolve_examples_dataframe(self, path_or_name):
        import pandas as pd
        p = Path(path_or_name)
        if p.exists():
            sep = ";" if p.suffix == ".csv" and self._sniff_sep(p) == ";" else ","
            return pd.read_csv(p, sep=sep)
        pbox_candidate = (
            Path(os.path.expanduser("~/.packing-box/datasets"))
            / path_or_name
            / "data.csv"
        )
        if pbox_candidate.exists():
            return pd.read_csv(pbox_candidate, sep=";")
        raise FileNotFoundError(
            f"[LLMClassifier] few_shot_examples_path '{path_or_name}' not found "
            f"as a file path or pbox dataset name "
            f"(looked in ~/.packing-box/datasets/{path_or_name}/data.csv)."
        )

    @staticmethod
    def _sniff_sep(path):
        with open(path, encoding="utf-8", errors="replace") as f:
            first_line = f.readline()
        return ";" if first_line.count(";") > first_line.count(",") else ","

    def _truncate_prompt_to_context(self, prompt):
        """Trim prompt token-wise so prompt + generation fit in ``n_ctx``.

        Reserves ``max_tokens`` for the completion plus a small margin. Keeps
        the **suffix** of the prompt (the target sample usually appears at the
        end). Falls back to a character heuristic if tokenization is unavailable.
        """
        self.backend_.load()
        llm = self.backend_._llm
        if llm is None or not prompt:
            return prompt, False
        n_ctx = int(getattr(self.backend_, "n_ctx", 2048) or 2048)
        max_gen = int(self.max_tokens or 64)
        reserve = 8
        budget = max(32, n_ctx - max_gen - reserve)
        enc = prompt.encode("utf-8")
        tokens = None
        for call in (
            lambda: llm.tokenize(enc, add_bos=False),
            lambda: llm.tokenize(enc),
            lambda: llm.tokenize(enc, add_bos=True),
        ):
            try:
                tokens = call()
                break
            except TypeError:
                continue
            except Exception:
                continue
        if tokens is None:
            approx_chars = max(256, int(budget * 3.2))
            if len(prompt) > approx_chars:
                warnings.warn(
                    f"[LLMClassifier] Prompt length ~{len(prompt)} chars may exceed "
                    f"context (n_ctx={n_ctx}, max_tokens={max_gen}); truncated with "
                    "a character fallback (tokenization unavailable).",
                    UserWarning,
                    stacklevel=3,
                )
                return prompt[-approx_chars:], True
            return prompt, False
        if len(tokens) <= budget:
            return prompt, False
        kept = tokens[-budget:]
        try:
            new_prompt = llm.detokenize(kept).decode("utf-8", errors="replace")
        except Exception:
            approx_chars = max(256, int(budget * 3.2))
            new_prompt = prompt[-approx_chars:]
        warnings.warn(
            f"[LLMClassifier] Prompt truncated to fit context window: "
            f"{len(tokens)} tokens reduced to {len(kept)} (n_ctx={n_ctx}, "
            f"max_new_tokens={max_gen}).",
            UserWarning,
            stacklevel=3,
        )
        return new_prompt, True

    @contextmanager
    def _open_trace_file(self):
        """Open the trace file once for the duration of a predict call.

        Yields an open file handle when tracing is enabled, or None otherwise.
        Writing is unbuffered at the OS level — each sample is written immediately
        so a mid-run failure preserves all traces up to that point, without the
        per-sample fsync overhead of the previous design.
        """
        if not self.trace_outputs:
            yield None
            return
        path = self._resolve_trace_path()
        with open(path, "a", encoding="utf-8") as f:
            yield f

    def _resolve_trace_path(self):
        if self._trace_path is not None:
            return self._trace_path
        base_dir = self.trace_dir or os.path.expanduser("~/.packing-box/cache/llm-traces")
        Path(base_dir).mkdir(parents=True, exist_ok=True)
        if self._trace_run_id is None:
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            self._trace_run_id = f"{ts}-pid{os.getpid()}"
        model_stub = str(self.model_file or "model").replace("/", "_")
        algo_stub = str(self.representation_style or "flat").replace("/", "_")
        prompt_stub = str(self.prompt_file or "prompt").replace("/", "_")
        self._trace_path = str(Path(base_dir) / f"{self._trace_run_id}-{algo_stub}-{prompt_stub}-{model_stub}.jsonl")
        return self._trace_path

    def _trace_sample(
        self,
        method,
        sample_index,
        prompt,
        raw_response,
        parsed_label,
        proba=None,
        parse_meta=None,
        error=None,
        trace_file=None,
    ):
        if not self.trace_outputs or trace_file is None:
            return
        event = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "run_id": self._trace_run_id or "pending",
            "method": method,
            "sample_index": int(sample_index),
            "model_file": self.model_file,
            "model_repo": self.model_repo,
            "prompt_file": self.prompt_file,
            "representation_style": self.representation_style,
            "n_ctx": self.n_ctx,
            "n_threads": self.n_threads,
            "max_tokens": self.max_tokens,
            "temperature": getattr(self, "temperature", None),
            "top_p": getattr(self, "top_p", None),
            "parsed_label": int(parsed_label),
            "parse_meta": parse_meta or {},
            "raw_response": str(raw_response),
            "proba": proba,
            "error": error,
            "prompt": prompt,
            "trace_path": trace_file.name,
        }
        trace_file.write(json.dumps(event, ensure_ascii=False) + "\n")
