# -*- coding: UTF-8 -*-
import functools

__all__ = ["FeatureFormatter"]

_BULLET_HEADERS = {
    "compact_pe_structural": "COMPACT_PE_BUNDLE",
    "strings_summary": "STRINGS_SUMMARY",
}

_VALID_STYLES = {"flat", "structured_semantic_pe", "compact_pe_structural", "strings_summary"}


def _load_descriptions():
    """Load feature name -> description mapping from Features registry."""
    from pbox.core.executable import Features

    # Important: instantiate once so Features lazily computes its registry.
    Features()
    return dict(getattr(Features, "descriptions", {}) or {})


class FeatureFormatter:
    """Convert a row of feature values into a human-readable text block for the LLM.

    Each selected feature is rendered as one line::

        <Description from features.yml>: <value>

    If a feature name is not found in ``features.yml``, the raw feature name
    (underscores replaced with spaces) is used as the label instead.
    If a feature name is missing from the row entirely, the value is shown as ``N/A``.

    Parameters
    ----------
    feature_names : list of str
        Ordered list of feature names to include in the text block. These must match
        column names in the feature DataFrame produced by pbox.

    Attributes
    ----------
    _descriptions : dict
        Loaded mapping of feature name → description (populated on first ``format`` call).
    """

    def __init__(self, feature_names, representation_style="flat"):
        if representation_style not in _VALID_STYLES:
            raise ValueError(
                f"[FeatureFormatter] Unknown representation_style '{representation_style}'. "
                f"Valid styles: {sorted(_VALID_STYLES)}"
            )
        self.feature_names = feature_names
        self.representation_style = representation_style
        self._feature_index = {name: idx for idx, name in enumerate(feature_names)}
        self._descriptions = {}
        self._descriptions_loaded = False
        if representation_style in _BULLET_HEADERS:
            _header = _BULLET_HEADERS[representation_style]
            self._render = functools.partial(self._format_bullet_list, header=_header)
        elif representation_style == "structured_semantic_pe":
            self._render = self._format_structured_semantic_pe
        else:
            self._render = self._format_flat

    def format(self, row):
        """Format a single feature row as a text block.

        Parameters
        ----------
        row : dict or pandas.Series
            Feature values keyed by feature name.

        Returns
        -------
        str
            Multi-line text block, one feature per line.
        """
        if not self._descriptions_loaded:
            self._descriptions = _load_descriptions()
            self._descriptions_loaded = True
        return self._render(row)

    def _format_flat(self, row):
        lines = []
        for name in self.feature_names:
            label = self._descriptions.get(name, name.replace("_", " "))
            lines.append(f"{label}: {self._format_value(self._get_row_value(row, name))}")
        return "\n".join(lines)

    def _format_structured_semantic_pe(self, row):
        groups = {
            "entropy_and_entrypoint": [],
            "sections_and_structure": [],
            "imports_and_apis": [],
            "strings_and_content": [],
            "other_indicators": [],
        }
        for name in self.feature_names:
            item = (self._descriptions.get(name, name.replace("_", " ")), self._format_value(self._get_row_value(row, name)))
            n = name.lower()
            if "entropy" in n or "ep" in n or "entry" in n:
                groups["entropy_and_entrypoint"].append(item)
            elif "section" in n or "size_" in n or "headers" in n or "image" in n:
                groups["sections_and_structure"].append(item)
            elif "import" in n or "api" in n or "dll" in n or "iat" in n or "idt" in n:
                groups["imports_and_apis"].append(item)
            elif "string" in n or "url" in n:
                groups["strings_and_content"].append(item)
            else:
                groups["other_indicators"].append(item)
        lines = ["PE_STATIC_REPORT {"]
        for section, pairs in groups.items():
            if not pairs:
                continue
            lines.append(f"  {section}:")
            for label, value in pairs:
                lines.append(f"    - {label}: {value}")
        lines.append("}")
        return "\n".join(lines)

    def _format_bullet_list(self, row, header):
        lines = [header]
        for name in self.feature_names:
            label = self._descriptions.get(name, name.replace("_", " "))
            value = self._format_value(self._get_row_value(row, name))
            lines.append(f"- {label}: {value}")
        return "\n".join(lines)

    @staticmethod
    def _safe_index(row, idx):
        try:
            if hasattr(row, "iloc"):
                return row.iloc[idx]
            return row[idx]
        except Exception:
            return "N/A"

    def _get_row_value(self, row, name):
        try:
            return row[name]
        except (KeyError, IndexError):
            idx = self._feature_index.get(name)
            if idx is None:
                return "N/A"
            return self._safe_index(row, idx)
        except TypeError:
            idx = self._feature_index.get(name)
            if idx is None:
                return "N/A"
            return self._safe_index(row, idx)

    @staticmethod
    def _format_value(value):
        if isinstance(value, float) and not isinstance(value, bool):
            return f"{value:.4f}"
        if isinstance(value, (list, tuple, set)):
            seq = list(value)
            preview = ", ".join(map(str, seq[:8]))
            suffix = " ..." if len(seq) > 8 else ""
            return f"[{preview}{suffix}] (n={len(seq)})"
        if isinstance(value, dict):
            keys = list(value.keys())
            preview = ", ".join(map(str, keys[:8]))
            suffix = " ..." if len(keys) > 8 else ""
            return f"{{keys: {preview}{suffix}}} (n={len(keys)})"
        return value
