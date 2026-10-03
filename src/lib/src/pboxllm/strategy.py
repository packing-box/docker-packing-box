# -*- coding: UTF-8 -*-
import os
import re
import shutil
from pathlib import Path


__all__ = ["PromptStrategy"]

_PROMPT_CACHE_DIR = Path(os.path.expanduser("~/.packing-box/cache/prompt"))
_DEFAULT_PROMPT_DIR = Path(__file__).resolve().parent / "prompts"

_YES = re.compile(r"\b(packed|yes|true|1)\b", re.IGNORECASE)
_NO  = re.compile(r"\b(not.packed|unpacked|no|false|0)\b", re.IGNORECASE)


class PromptStrategy:
    """Load a prompt template and parse the LLM response into a binary label.

    The prompt template file must contain the placeholder ``{features}`` which is
    replaced at inference time with the formatted feature text block produced by
    :class:`~pboxllm.formatter.FeatureFormatter`.

    Prompt templates are copied to and loaded from ``~/.packing-box/cache/prompt/``.
    Defaults are bundled in ``pboxllm/prompts/`` and bootstrapped on first use.

    Parameters
    ----------
    prompt_file : str
        Filename of the prompt template (e.g. ``zero_shot_binary.txt``).

    prompt_dir : Path, optional
        Directory containing prompt templates. Defaults to
        ``~/.packing-box/cache/prompt/``. Override in tests to avoid
        filesystem dependencies.

    Attributes
    ----------
    _template : str or None
        Contents of the loaded prompt file (populated on first use).
    """

    def __init__(
        self,
        prompt_file,
        prompt_dir=_PROMPT_CACHE_DIR,
        default_prompt_dir=_DEFAULT_PROMPT_DIR,
        packed_label="packed",
        not_packed_label="not-packed",
        parse_policy="p_packed_fallback",
    ):
        self.prompt_file = prompt_file
        self.prompt_dir = prompt_dir
        self.default_prompt_dir = default_prompt_dir
        self.packed_label = str(packed_label).strip().lower()
        self.not_packed_label = str(not_packed_label).strip().lower()
        self.parse_policy = str(parse_policy or "p_packed_fallback").strip().lower()
        self._template = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def build_prompt(self, features_text, few_shot_examples=None):
        """Fill the prompt template with the formatted feature block.

        Parameters
        ----------
        features_text : str
            Human-readable feature block produced by :class:`~pboxllm.formatter.FeatureFormatter`.

        Returns
        -------
        str
            The full prompt ready to be sent to the LLM backend.
        """
        return self._render_template(
            features_text=features_text,
            few_shot_examples=few_shot_examples or [],
        )

    def parse(self, response):
        return self.parse_with_meta(response)["label"]

    def parse_with_meta(self, response):
        """Parse the LLM raw response into a binary label.

        Looks for keywords indicating packed (1) or not-packed (0). 
        Returns -1 when neither pattern matches so that
        the metrics pipeline can exclude the sample rather than misclassify it.

        Parameters
        ----------
        response : str
            Raw text returned by the LLM backend.

        Returns
        -------
        int
            ``1`` if packed, ``0`` if not-packed, ``-1`` if uncertain/unrecognised.
        """
        response = str(response or "")
        resp_lower = response.lower().strip()
        meta = {
            "label": -1,
            "reason": "unparsed",
            "decision": None,
            "decision_valid": False,
            "p_packed": None,
            "used_fallback": False,
        }
        if not resp_lower:
            meta["reason"] = "empty_response"
            return meta

        d = re.search(r"\bdecision\s*:\s*([A-Za-z]+)\b", response, re.IGNORECASE)
        if d:
            decision = d.group(1).strip().upper()
            meta["decision"] = decision
            if decision in ("YES", "NO"):
                meta["decision_valid"] = True

        # 1) Preferred path: parse explicit "P_PACKED: <p>" outputs.
        # This is required for prompts coming from the experiments pipeline
        # (e.g., zero_shot/verbalized_probability.txt).
        m = re.search(r"\bP_PACKED\s*:\s*([0-9]*\.?[0-9]+)\b", resp_lower, re.IGNORECASE)
        if m:
            try:
                p = float(m.group(1))
                meta["p_packed"] = p
                if self.parse_policy == "strict_unknown" and meta["decision"] is not None and not meta["decision_valid"]:
                    meta["reason"] = "invalid_decision_token"
                    return meta
                if p >= 0.5:
                    meta["label"] = 1
                    meta["reason"] = "p_packed_threshold"
                    if meta["decision"] == "NO":
                        meta["used_fallback"] = True
                    return meta
                if p < 0.5:
                    meta["label"] = 0
                    meta["reason"] = "p_packed_threshold"
                    if meta["decision"] == "YES":
                        meta["used_fallback"] = True
                    return meta
            except ValueError:
                pass

        if self.parse_policy == "strict_unknown" and meta["decision"] is not None and not meta["decision_valid"]:
            meta["reason"] = "invalid_decision_token"
            return meta

        # 2) Strict token match first (best for single-token prompts).
        token = resp_lower.split()
        token0 = token[0].strip().lower() if token else ""
        if token0:
            if token0 == self.not_packed_label:
                meta["label"] = 0
                meta["reason"] = "exact_token_not_packed"
                return meta
            if token0 == self.packed_label:
                meta["label"] = 1
                meta["reason"] = "exact_token_packed"
                return meta

        # 3) Fallback keyword search.
        # Note: keep _NO before _YES so "no"/"false"/"0" map to not-packed,
        # but ambiguous outputs are still handled by the exact-token path above.
        if _NO.search(response):
            meta["label"] = 0
            meta["reason"] = "keyword_no"
            return meta
        if _YES.search(response):
            meta["label"] = 1
            meta["reason"] = "keyword_yes"
            return meta
        return meta

    def validate_template(self, few_shot_count):
        """Raise ValueError if the template is incompatible with the few-shot config.

        Call this from ``LLMClassifier.fit`` so template/mode mismatches surface
        before any inference runs.
        """
        template = self._load_template()
        has_ph = "{few_shot_examples}" in template
        if few_shot_count and not has_ph:
            raise ValueError(
                f"[pboxllm] few_shot_count={few_shot_count} but template "
                f"'{self.prompt_file}' has no {{few_shot_examples}} placeholder."
            )

    # ------------------------------------------------------------------
    # Template loading
    # ------------------------------------------------------------------

    def _load_template(self):
        if self._template is not None:
            return self._template
        self._bootstrap_default_prompt()
        path = self.prompt_dir / self.prompt_file
        if not path.exists():
            raise FileNotFoundError(
                f"[pboxllm] Prompt template not found: {path}\n"
                f"Expected location: {self.prompt_dir}"
            )
        self._template = path.read_text(encoding="utf-8")
        return self._template

    def _render_template(self, features_text, few_shot_examples):
        template = self._load_template()
        mapping = {"features": features_text}
        has_ph = "{few_shot_examples}" in template
        if has_ph:
            mapping["few_shot_examples"] = self._format_few_shot_examples(few_shot_examples)
            return template.format(**mapping)
        if few_shot_examples:
            raise ValueError(
                f"[pboxllm] few_shot_count > 0 but template '{self.prompt_file}' "
                "has no {few_shot_examples} placeholder. "
                "Add the placeholder to the template where you want demo examples to appear, "
                "or call validate_template() during fit to catch this early."
            )
        return template.format(**mapping)

    def _few_shot_body(self, few_shot_examples):
        """Return the numbered example block only (no surrounding English header)."""
        if not few_shot_examples:
            return ""
        lines = []
        for i, example in enumerate(few_shot_examples, start=1):
            feat_txt = example.get("features_text", "")
            if not feat_txt:
                raise ValueError(
                    f"[pboxllm] FewShotExample at index {i - 1} has an empty "
                    "'features_text'. This would inject a blank example into the prompt."
                )
            label = self._normalize_label(example.get("label", "unknown"))
            lines.append(
                "Example {i}:\nFeatures:\n{features}\nLabel: {label}".format(
                    i=i,
                    features=feat_txt,
                    label=label,
                )
            )
        return "\n\n".join(lines)

    def _format_few_shot_examples(self, few_shot_examples):
        body = self._few_shot_body(few_shot_examples)
        if not body:
            return ""
        return "\nHere are some examples to guide your reasoning:\n\n{}\n".format(body)

    def _normalize_label(self, label):
        """Map training / sklearn-style labels to the prompt wording configured on this instance.

        Uses ``self.packed_label`` / ``self.not_packed_label`` so that demo
        examples in few-shot prompts use the same label vocabulary as the
        response instruction (critical for neutral-label configs such as A/B).
        """
        if label in [1, "1", True]:
            return self.packed_label
        if label in [0, "0", False]:
            return self.not_packed_label
        text = str(label).strip().lower()
        if text == self.packed_label or text == self.not_packed_label:
            return text
        return "unknown"

    def _bootstrap_default_prompt(self):
        self.prompt_dir.mkdir(parents=True, exist_ok=True)
        target = self.prompt_dir / self.prompt_file
        if target.exists():
            return
        source = self.default_prompt_dir / self.prompt_file
        if source.exists():
            shutil.copyfile(source, target)
