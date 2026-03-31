# -*- coding: UTF-8 -*-
import os
import re
from pathlib import Path


__all__ = ["PromptStrategy"]

_PROMPT_DIR = Path(os.path.expanduser("~/.packing-box/data/prompt"))

_YES = re.compile(r"\b(packed|yes|true|1)\b", re.IGNORECASE)
_NO  = re.compile(r"\b(not.packed|unpacked|no|false|0)\b", re.IGNORECASE)


class PromptStrategy:
    """Load a prompt template and parse the LLM response into a binary label.

    The prompt template file must contain the placeholder ``{features}`` which is
    replaced at inference time with the formatted feature text block produced by
    :class:`~pboxllm.formatter.FeatureFormatter`.

    Prompt templates live in ``~/.packing-box/data/prompt/``.

    Parameters
    ----------
    prompt_file : str
        Filename of the prompt template (e.g. ``zero_shot_binary.txt``).

    prompt_dir : Path, optional
        Directory containing prompt templates. Defaults to
        ``~/.packing-box/data/prompt/``. Override in tests to avoid
        filesystem dependencies.

    Attributes
    ----------
    _template : str or None
        Contents of the loaded prompt file (populated on first use).
    """

    def __init__(self, prompt_file, prompt_dir=_PROMPT_DIR):
        self.prompt_file = prompt_file
        self.prompt_dir = prompt_dir
        self._template = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def build_prompt(self, features_text):
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
        return self._load_template().format(features=features_text)

    def parse(self, response):
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
        if _NO.search(response):
            return 0
        if _YES.search(response):
            return 1
        return -1

    # ------------------------------------------------------------------
    # Template loading
    # ------------------------------------------------------------------

    def _load_template(self):
        if self._template is not None:
            return self._template
        path = self.prompt_dir / self.prompt_file
        if not path.exists():
            raise FileNotFoundError(
                f"[pboxllm] Prompt template not found: {path}\n"
                f"Expected location: {self.prompt_dir}"
            )
        self._template = path.read_text(encoding="utf-8")
        return self._template
