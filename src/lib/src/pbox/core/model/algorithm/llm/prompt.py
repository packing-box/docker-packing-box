# -*- coding: UTF-8 -*-
from tinyscript import re
from tinyscript.helpers import Path


__all__ = ["Prompt"]

_DEFAULT_PROMPT_DIR = Path(__file__).parent / "prompts"
_NO  = re.compile(r"\b(n|no|not[-_\s]packed|unpacked|false|0)\b", re.I)
_PROMPT_TEMPLATES_CACHE = {}
_YES = re.compile(r"\b(y|yes|packed|true|1)\b", re.I)
_SAMPLE_MARKER = "\x00SAMPLE\x00"


class Prompt:
    """ Load a Jinja2 prompt template and parse the LLM response into a binary label.

    The prompt template is rendered with the following variables:
    - ``sample``:   the formatted feature values of the sample to be classified (mandatory)
    - ``schema``:   the description of the features
    - ``examples``: a list of few-shot examples, each with ``sample`` and ``label`` ("Y" or "N") attributes

    Prompt templates are loaded from the configured 'prompts' folder (i.e. ``~/.packing-box/prompts/`` or the current
     experiment workspace's ``prompts`` subfolder) or, as a fallback, from the bundled ``prompts`` folder.
    """
    def __init__(self, path, schema="", examples=()):
        p = Path(str(path))
        if not p.suffix:
            p = Path(f"{p}.j2")
        self.path = config['prompts'].joinpath(p)
        if not self.path.exists():
            self.path = _DEFAULT_PROMPT_DIR.joinpath(p)
        if not self.path.exists():
            raise FileNotFoundError(f"Prompt template '{path}' not found in {config['prompts']} nor in "
                                    f"{_DEFAULT_PROMPT_DIR}")
        self.schema, self.examples = schema, list(examples)
    
    def __load(self):
        try:
            return _PROMPT_TEMPLATES_CACHE[str(self.path)]
        except KeyError:
            from jinja2 import Environment, meta, StrictUndefined
            env = Environment(keep_trailing_newline=True, undefined=StrictUndefined)
            src = self.path.read_text(encoding="utf-8")
            if "sample" not in (v := meta.find_undeclared_variables(env.parse(src))):
                raise ValueError(f"Bad prompt template '{self.path}' ; it must contain the '{{{{ sample }}}}' "
                                 "placeholder")
            t = _PROMPT_TEMPLATES_CACHE[str(self.path)] = (env.from_string(src), frozenset(v))
            return t
    
    def build(self, sample):
        """ Fill the prompt template with the formatted feature block of the sample. """
        return self.template.render(sample=sample, schema=self.schema, examples=self.examples)
    
    def parse(self, response):
        """ Parse the LLM raw response into a binary label (1: packed, 0: not packed, -1: unknown). """
        # first, rely on the leading word of the answer (e.g. "Y", "No, ...", "Packed") ; then search the whole answer
        if m := re.match(r"\W*([a-z]+(?:.packed)?)", response, re.I):
            if _NO.fullmatch(w := m.group(1)):
                return 0
            if _YES.fullmatch(w):
                return 1
        return 0 if _NO.search(response) else 1 if _YES.search(response) else -1
    
    @property
    def prefix(self):
        """ Static part of the rendered prompt, preceding the sample (i.e. for priming a KV cache). """
        return self.build(_SAMPLE_MARKER).split(_SAMPLE_MARKER, 1)[0]
    
    @property
    def template(self):
        return self.__load()[0]
    
    @property
    def variables(self):
        """ Variables used in the template (i.e. 'schema' or 'examples'), telling the inference strategy. """
        return self.__load()[1]

