# -*- coding: UTF-8 -*-


__all__ = ["FeatureFormatter"]


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

    def __init__(self, feature_names):
        self.feature_names = feature_names
        self._descriptions = {}
        self._descriptions_loaded = False

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

        lines = []
        for name in self.feature_names:
            label = self._descriptions.get(name, name.replace("_", " "))
            try:
                value = row[name]
            except (KeyError, IndexError):
                value = "N/A"
            if isinstance(value, float) and not isinstance(value, bool):
                value = f"{value:.4f}"
            lines.append(f"{label}: {value}")
        return "\n".join(lines)
