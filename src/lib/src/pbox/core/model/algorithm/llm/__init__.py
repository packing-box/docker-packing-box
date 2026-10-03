# -*- coding: UTF-8 -*-
from tinyscript import logging, re
import numpy as np
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils._param_validation import Integral, Interval, Real, StrOptions
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from .backend import LLM
from .context import get_context
from .prompt import Prompt
from ....executable import Features


__all__ = ["LLMBinaryClassifier"]

# answers, in the order of the columns of the probabilities array: not packed (0), packed (1), unknown
_ANSWERS = "NY?"
_ANSWER_WORDS = ({"n", "no", "not", "unpacked", "false"}, {"y", "yes", "packed", "true"}, {"?"})
_ANSWER_VARIANTS = (["N", "No", "no", "n"], ["Y", "Yes", "yes", "y"], ["?"])
_GRAMMARS = {
    'label':      "root ::= [YN?]",
    'verbalized': "root ::= [YN?] \" \" prob\n"
                  "prob ::= \"0\" (\".\" [0-9] [0-9]?)? | \"1\" (\".0\" \"0\"?)?",
}
_PROBA = re.compile(r"(?<![\d.])(0(?:\.\d+)?|1(?:\.0+)?|\.\d+)(?![\d.])")
_TOP_LOGPROBS = 20
_VERBALIZED = "\nAfter your answer, give the probability (between 0 and 1) that the sample is packed, e.g. \"Y 0.85\"."




def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p)).reshape(-1, 1)


class LLMBinaryClassifier(ClassifierMixin, BaseEstimator):
    """LLM-based packing binary classifier.

    No gradient-based training occurs. ``fit`` loads the model into memory.

    Attributes
    ----------
    answer_ids_ : tuple of sets
        Token ids standing for each answer (not packed, packed, unknown) ; only for proba_method='logits'.

    backend_ : LLM
        The LLM backend model's object.

    calibrator_ : LogisticRegression or IsotonicRegression or None
        The fitted calibrator of the probabilities, if 'calibration' is set.

    classes_ : np.array([0, 1])
        0 is not packed, 1 is packed

    context_ : BaseContext
        The context management strategy instance, primed on 'fit' (see :mod:`.context`).

    prompt_ : Prompt
        The loaded prompt template.

    Parameters
    ----------
    model : str, default=None
        The reference to the local model cached into configured 'llm_cache' or '[owner]/[repository]/[filename].gguf'
         for download from the Hugging Face Hub.

    prompt_template : str, default="zero-shot-with-schema"
        The Jinja2 template of the prompt (see :class:`~.prompt.Prompt`) ; it determines the inference strategy
         through the variables it uses: '{{ schema }}' for describing the features and '{{ examples }}' for
         few-shot examples.

    context_manager : str, default="kv_cache"
        One of 'none', 'messages', 'kv_cache', 'llama_state' (see :func:`.context.get_context`) ; 'messages' turns the
         predictions into a single conversation, growing with every prediction.

    few_shots : int, default=0
        Number of training examples, balanced across classes, given in the prompt (only if the template uses
         '{{ examples }}').

    n_context : int or "auto", default="auto"
        Number of tokens of the context window ; "auto" uses the model's own trained context window.

    max_tokens : int, default=None
        Maximum number of generated tokens ; when None, configured 'max_output_tokens' is used.

    temperature : float, default=None
        Generation temperature ; when None, configured 'temperature' is used.

    feature_format : str, default="plain"
        One of 'plain', 'json', 'yaml', 'markdown'.

    feature_names : list, default=None
        Names of the features, in the order of the columns of X ; when set, these features are imposed to the model.

    float_precision : int, default=None
        Number of decimals for float feature values ; when None, configured 'float_precision' is used.

    proba_method : str, default="logits"
        How probabilities are obtained (each one gives P(not packed), P(packed) and P(unknown), the answers being Y, N
         or ?):
        - 'label':      parse the generated answer, i.e. probabilities are 0 or 1
        - 'logits':     evaluate the prompt and take the softmax of the next-token logits of the tokens standing for
                         Y, N and ?, without generating anything (cheapest and deterministic)
        - 'logprobs':   generate the answer and sum the probabilities of the most likely first tokens standing for Y,
                         N and ? (requires llama.cpp to keep all logits, i.e. more memory)
        - 'sampling':   generate 'n_samples' answers and take the frequency of each answer ('n_samples' times the
                         cost ; temperature=1. is used if the temperature is 0)
        - 'verbalized': ask the model for the probability that the sample is packed along with its answer (poorly
                         calibrated)

    constrained : bool, default=True
        Whether the generated answer is constrained with a grammar to Y, N or ? (followed by a probability for
         proba_method='verbalized') ; not applicable to proba_method='logits'.

    n_samples : int, default=10
        Number of generated answers per sample for proba_method='sampling'.

    calibration : str, default=None
        One of None, 'sigmoid' (Platt scaling) or 'isotonic' ; when set, the probabilities are calibrated on the
         labelled training samples during 'fit' (which then runs inference on the whole training set).

    unknown_threshold : float, default=.5
        A sample is predicted as unknown (-1) when P(unknown) is greater than or equal to this threshold.

    unknown_margin : float, default=.0
        A sample is predicted as unknown (-1) when P(packed) is within this margin around .5 ; with .0, only a
         probability of exactly .5 means unknown. Unknown samples get a probability of .5 from 'predict_proba'.

    Examples
    --------
    >>> from pbox.core.model.algorithm.llm import LLMBinaryClassifier
    >>> from pbox.helpers import make_test_dataset
    >>> X_train, y_train, X_test, y_test = make_test_dataset(2)
    >>> clf = LLMBinaryClassifier().fit(X_train, y_train)
    >>> clf.predict(X_test[:5, :])
    array([1, 0, 1, 1, 1])
    >>> clf.score(X_test, y_test)
    0.8...
    """
    classes_ = np.array([0, 1])
    _parameter_constraints = {
        'model':              [str, None],
        'prompt_template':    [str],
        'context_manager':    [StrOptions({"none", "messages", "kv_cache", "llama_state"})],
        'few_shots':          [Interval(Integral, 0, None, closed="left")],
        'n_context':          [StrOptions({"auto"}), Interval(Integral, 512, None, closed="left")],
        'max_tokens':         [Interval(Integral, 1, None, closed="left"), None],
        'temperature':        [Interval(Real, 0.0, 2.0, closed="both"), None],
        'feature_format':     [StrOptions({"plain", "json", "yaml", "markdown"})],
        'feature_names':      ["array-like", None],
        'float_precision':    [Interval(Integral, 0, None, closed="left"), None],
        'proba_method':       [StrOptions({"label", "logits", "logprobs", "sampling", "verbalized"})],
        'constrained':        ["boolean"],
        'n_samples':          [Interval(Integral, 1, None, closed="left")],
        'calibration':        [StrOptions({"sigmoid", "isotonic"}), None],
        'unknown_threshold':  [Interval(Real, 0., 1., closed="right")],
        'unknown_margin':     [Interval(Real, 0., .5, closed="left")],
    }
    
    def __init__(self, model=None, prompt_template="zero-shot-with-schema", context_manager="kv_cache", few_shots=0,
                 n_context="auto", max_tokens=None, temperature=None, feature_format="plain", feature_names=None,
                 float_precision=None, proba_method="logits", constrained=True, n_samples=10, calibration=None,
                 unknown_threshold=.5, unknown_margin=.0):
        # sklearn convention: __init__ only stores the given parameters as-is (get_params/clone rely on this) ;
        #  anything derived from them is computed in 'fit'
        self.model = model
        self.prompt_template = prompt_template
        self.context_manager = context_manager
        self.few_shots = few_shots
        self.n_context = n_context
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.feature_format = feature_format
        self.feature_names = feature_names
        self.float_precision = float_precision
        self.proba_method = proba_method
        self.constrained = constrained
        self.n_samples = n_samples
        self.calibration = calibration
        self.unknown_threshold = unknown_threshold
        self.unknown_margin = unknown_margin
    
    def __decide(self, P):
        """ From the (n_samples, 3) array of probabilities (not packed, packed, unknown), compute P(packed), calibrated
             if relevant, and whether each sample is unknown. """
        s = P[:, 0] + P[:, 1]
        p = np.divide(P[:, 1], s, out=np.full(len(P), .5), where=s > 0)
        if (c := getattr(self, "calibrator_", None)) is not None:
            p = c.predict_proba(_logit(p))[:, 1] if hasattr(c, "predict_proba") else c.predict(p)
        return p, (P[:, 2] >= self.unknown_threshold) | (np.abs(p - .5) <= self.unknown_margin)
    
    def __format_features(self, feature_values, names=True):
        """ Format a feature vector as a text block given the selected format. """
        f, lines = self.feature_format, []
        for name, value in zip(self.feature_names_, feature_values):
            if isinstance(value, (float, np.floating)):
                value = f"{value:.{self.float_precision_}f}"
            elif isinstance(value, np.generic):
                value = value.item()
            name = name if names else Features.descriptions.get(name, name)
            lines.append({'plain': "{}={}", 'json': "  \"{}\": {}", 'markdown': "{} | {}", 'yaml': "{}: {}"}[f] \
                         .format(name, value))
        if f == "json":
            return "{\n" + ",\n".join(lines) + "\n}"
        return ("Name | Value\n--- | ---\n" if f == "markdown" else "") + "\n".join(lines)
    
    def __infer(self, body):
        """ Run inference for a single sample, returning its probabilities (not packed, packed, unknown) and the answer
             to be recorded in the context. """
        prompt, method, P = self.context_.render() + body, self.proba_method, np.zeros(3)
        grammar = _GRAMMARS['verbalized' if method == "verbalized" else 'label'] if self.constrained else None
        ask = lambda **kw: self.backend_.complete(prompt, temperature=self.temperature_, grammar=grammar,
                                                  **{'max_tokens': self.max_tokens, **kw})
        if method == "logits":
            logits = self.backend_.next_token_logits(prompt)
            s = np.array([logsumexp(logits[ids]) if len(ids) else -np.inf for ids in self.answer_ids_])
            P = np.exp(s - s.max())
            return P / P.sum(), _ANSWERS[P.argmax()]
        if method == "logprobs":
            c = ask(max_tokens=1, logprobs=_TOP_LOGPROBS)
            for token, lp in (c['logprobs']['top_logprobs'][0] or {}).items():
                t = token.strip().lower()
                for i, words in enumerate(_ANSWER_WORDS):
                    if t in words:
                        P[i] += np.exp(lp)
                        break
            if P.sum() == 0.:
                P[2] = 1.
            return P / P.sum(), c['text'].strip()
        if method == "sampling":
            for _ in range(self.n_samples):
                P[self.prompt_.parse(ask()['text']) % 3] += 1  # -1 (unknown) -> 2
            return P / P.sum(), _ANSWERS[P.argmax()]
        answer = ask()['text'].strip()
        P[label := self.prompt_.parse(answer) % 3] = 1.
        if method == "verbalized" and label != 2 and (m := _PROBA.search(answer)):
            P[:] = 1. - (p := float(m.group(1))), p, 0.
        return P, answer
    
    def __probas(self, X):
        """ Compute the (n_samples, 3) array of probabilities (not packed, packed, unknown), each sample being inferred
             only once (i.e. 'predict' and 'predict_proba' are consistent and do not run inference twice). """
        check_is_fitted(self, attributes=["backend_", "context_", "prompt_"])
        P = np.zeros((len(X), 3))
        for i in range(len(X)):
            body = self.prompt_.build(self.__format_features(X.iloc[i] if hasattr(X, "iloc") else X[i]))
            if self.proba_method == "verbalized":
                body += _VERBALIZED
            if (r := self.probas_cache_.get(body)) is None:
                self.context_.restore(self.backend_)
                r, answer = self.__infer(body)
                self.context_.update(body, answer)
                self.probas_cache_[body] = r
            P[i] = r
        return P
    
    def fit(self, X, y=None):
        self._validate_params()
        if y is None:
            X = check_array(X, accept_sparse=False, ensure_2d=True, dtype=None)
        else:
            X, y = check_X_y(X, y, accept_sparse=False, ensure_2d=True, dtype=None)
        logger = logging.getLogger()
        self.feature_names_ = list(Features.names if self.feature_names is None else self.feature_names)
        if len(self.feature_names_) != X.shape[1]:
            raise ValueError(f"{X.shape[1]} features in the input data while {len(self.feature_names_)} feature names "
                             "were provided")
        self.n_features_in_ = X.shape[1]
        self.float_precision_ = config['float_precision'] if self.float_precision is None else self.float_precision
        # the inference strategy (with or without schema, zero- or few-shot) is inferred from the template's variables
        self.prompt_ = p = Prompt(self.prompt_template)
        if "schema" in p.variables:
            d = Features.descriptions
            p.schema = "Features:\n" + "\n".join(f"- {n}: {d.get(n, n)}" for n in self.feature_names_)
        if "examples" in p.variables:
            if self.few_shots == 0 or y is None:
                p.examples = []
            else:
                pools, p.examples, n = [list(np.flatnonzero(y == c)) for c in self.classes_], [], 0
                while n < self.few_shots and any(pools):
                    for pool in pools:
                        if n >= self.few_shots:
                            break
                        if not pool:
                            continue
                        idx = pool.pop(0)
                        examples.append({'sample': self.__format_features(X[idx]), 'label': "NY"[int(y[idx])]})
                        n += 1
            if len(p.examples) == 0:
                logger.warning(f"Prompt template '{self.prompt_template}' expects few-shot examples while none could "
                               "be selected (few_shots=0 or no label)")
        elif self.few_shots > 0:
            logger.debug(f"'few_shots={self.few_shots}' ignored as prompt template '{self.prompt_template}' does not "
                         "use '{{ examples }}'")
        self.temperature_ = config['temperature'] if self.temperature is None else self.temperature
        if self.proba_method == "sampling" and self.temperature_ == 0.:
            logger.debug("temperature=0 makes sampling deterministic ; using temperature=1. instead")
            self.temperature_ = 1.
        self.backend_ = LLM(self.model, self.n_context, logits_all=self.proba_method == "logprobs", logger=logger)
        self.backend_.model  # lazily loads the model ; also resolves n_context="auto" against the model's own
                             #  trained context window (see :class:`~.backend.LLM`)
        self.n_context_ = self.backend_.n_context  # the actual resolved number of tokens
        self.context_ = get_context(self.context_manager, backend=self.backend_, prefix=p.prefix, logger=logger)
        if self.proba_method == "logits":
            ids = [self.backend_.token_ids(*v, *[" " + x for x in v]) for v in _ANSWER_VARIANTS]
            # discard ids shared by several answers (i.e. a leading whitespace token)
            self.answer_ids_ = tuple(sorted(i - set().union(*[o for o in ids if o is not i])) for i in ids)
            if not all(self.answer_ids_[:2]):
                raise ValueError("Could not find distinct tokens for the answers Y and N in the model's vocabulary")
        self.probas_cache_, self.calibrator_ = {}, None
        if self.calibration is not None:
            if y is None or len(np.unique(y[mask := np.isin(y, self.classes_)])) < 2:
                logger.warning("Calibration requires labelled samples of both classes ; skipping")
            else:
                logger.info(f"Calibrating probabilities ({self.calibration}) on {mask.sum()} training samples...")
                p, unknown = self.__decide(self.__probas(X[mask]))
                p, t = p[~unknown], y[mask][~unknown]
                if len(np.unique(t)) < 2:
                    logger.warning("Calibration requires samples of both classes not predicted as unknown ; skipping")
                else:
                    if self.calibration == "sigmoid":
                        from sklearn.linear_model import LogisticRegression
                        self.calibrator_ = LogisticRegression().fit(_logit(p), t)
                    else:
                        from sklearn.isotonic import IsotonicRegression
                        self.calibrator_ = IsotonicRegression(y_min=0., y_max=1., out_of_bounds="clip").fit(p, t)
        return self
    
    def predict(self, X):
        """ Predict 0 (not packed), 1 (packed) or -1 (unknown, see 'unknown_threshold' and 'unknown_margin'). """
        p, unknown = self.__decide(self.__probas(X))
        return np.where(unknown, -1, (p > .5).astype(int))
    
    def predict_proba(self, X):
        """ Predict the probabilities of (not packed, packed) ; unknown samples get .5. """
        p, unknown = self.__decide(self.__probas(X))
        p[unknown] = .5
        return np.vstack([1 - p, p]).T
    
    @property
    def _feature_names(self):
        """ Features imposed to the model (see :meth:`pbox.core.model.BaseModel._prepare`) ; only defined when
             'feature_names' is set. """
        if self.feature_names is None:
            raise AttributeError("_feature_names")
        return list(self.feature_names)

