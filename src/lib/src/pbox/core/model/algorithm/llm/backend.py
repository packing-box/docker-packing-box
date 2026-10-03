# -*- coding: UTF-8 -*-
from functools import lru_cache

import numpy as np
from tinyscript import logging
from tinyscript.helpers import positive_int


__all__ = ["LLM"]


# when the configured token is empty, None lets huggingface_hub fall back to HF_TOKEN or the locally saved token
_HF_TOKEN = lambda: config['huggingface_token'] or None


class LLM:
    """ Abstraction for LLM providing a text generation method.
    
    Attributes
    ----------
    model : Llama object, default=None
        The LLM backend model's object ; when calling this read-only property, the model is lazily loaded.
    
    Parameters
    ----------
    n_context : int or "auto", default="auto"
        The maximum number of tokens to be considered in the context ; when "auto", the underlying llama.cpp context
         is created with ``n_ctx=0``, which makes it size itself to the loaded model's own trained context window
         (``n_ctx_train``) ; the resolved value is then read back so that :attr:`n_context` reflects the actual
         number of tokens once the model is loaded.

    n_threads : int, default=config['nbr_threads']
        The number of threads to allocate to LLM text generation.

    logits_all : bool, default=False
        Whether llama.cpp shall keep the logits of every token, which is required for getting log-probabilities
         from :meth:`complete`.

    path : str
        The path to the LLM's GGUF file, relative to the configured 'llm_cache' ; it is either an existing file or a
         reference to a GGUF file from the Hugging Face Hub, formatted as '[owner]/[repository]/[filename].gguf', in
         which case the model is downloaded to '[llm_cache]/[owner]/[repository]/[filename].gguf' on first use.
    """
    def __init__(self, repo_id_or_path, n_context=None, n_threads=None, logits_all=False, logger=None):
        self.__llm = None
        self.logits_all = logits_all
        self.n_context = n_context or "auto"
        self.n_threads = n_threads or config['nbr_threads']
        self.path = repo_id_or_path
        self.logger = logger
    
    def ask(self, prompt, **kwargs):
        """ Run inference and return the raw generated text. """
        return self.complete(prompt, **kwargs)['text'].strip()
    
    def complete(self, prompt, max_tokens=None, temperature=None, grammar=None, logprobs=None):
        """ Run inference and return the first choice, holding the generated 'text' and, if 'logprobs' is set (requires
             logits_all=True), the 'logprobs' with the 'logprobs' most likely tokens at each generated position.

        :param grammar: GBNF grammar constraining the generated text
        """
        kw = {} if grammar is None else {'grammar': self.grammar(grammar)}
        if logprobs:
            kw['logprobs'] = logprobs
        return self.model(prompt, max_tokens=max_tokens or config['max_output_tokens'],
                          temperature=config['temperature'] if temperature is None else temperature, echo=False,
                          **kw)['choices'][0]
    
    def next_token_logits(self, prompt):
        """ Evaluate the prompt without generating anything and return the logits of the next token. """
        m = self.model
        tokens = m.tokenize(prompt.encode("utf-8"), add_bos=True, special=True)
        # reuse the longest prefix already evaluated (i.e. from a restored state or the previous call), leaving at
        #  least the last token to be evaluated so that its logits get computed
        n = 0
        for a, b in zip(m.input_ids[:m.n_tokens], tokens[:-1]):
            if a != b:
                break
            n += 1
        m.n_tokens = n
        m.eval(tokens[n:])
        return np.array(m.scores[m.n_tokens - 1], dtype=float)
    
    def token_ids(self, *texts):
        """ Return the set of ids of the first token of each text. """
        return {t[0] for s in texts if (t := self.model.tokenize(s.encode("utf-8"), add_bos=False, special=False))}
    
    @property
    def model(self):
        """ Model object, (lazily down)loaded on first use. """
        if self.__llm is None:
            from llama_cpp import Llama
            if not self.path.exists():
                try:
                    from huggingface_hub import hf_hub_download as dl
                    from tinyscript.helpers import Path
                    if self.logger:
                        self.logger.info(f"Model '{self.file}' not found in cache, downloading from '{self.repo}'...")
                    self.path = Path(dl(repo_id=self.repo, filename=self.file, token=_HF_TOKEN(),
                                        local_dir=config['llm_cache'].joinpath(self.repo)))
                except Exception as exc:
                    raise RuntimeError(f"Failed to download model '{self.file}' from '{self.repo}':"
                                       f" {exc}\nYou can also place the GGUF file manually at: {self.path}") from exc
            elif self.logger:
                self.logger.debug(f"Loading model '{self.file}' from cache ({self.n_context} context tokens ; "
                                  f"{self.n_threads} threads)")
            auto = self.__n_ctx == "auto"
            self.__llm = Llama(model_path=str(self.path), n_ctx=0 if auto else self.__n_ctx, n_threads=self.__n_threads,
                               logits_all=self.logits_all,
                               verbose=self.logger is not None and self.logger.isEnabledFor(logging.DEBUG))
            if auto:
                self.__n_ctx = self.__llm.n_ctx()  # llama.cpp resolved n_ctx=0 to the model's n_ctx_train
                if self.logger:
                    self.logger.debug(f"Auto-selected context window: {self.__n_ctx} tokens "
                                      f"(model's native n_ctx_train)")
        return self.__llm

    @property
    def n_context(self):
        return self.__n_ctx

    @n_context.setter
    def n_context(self, tokens):
        if tokens != "auto":
            positive_int(tokens, zero=False)
        self.__n_ctx = tokens
        self.__llm = None  # force resetting Llama instance with the new 'n_context' at next invocation of self.model

    @property
    def n_threads(self):
        return self.__n_threads

    @n_threads.setter
    def n_threads(self, threads):
        positive_int(threads, zero=False)
        self.__n_threads = threads
        self.__llm = None  # force resetting Llama instance with the new 'n_threads' at next invocation of self.model
    
    @property
    def path(self):
        return self.__path
    
    @path.setter
    def path(self, repo_id_or_path):
        if repo_id_or_path is None:
            raise ValueError("No model specified")
        self.__path = config['llm_cache'].joinpath(str(repo_id_or_path))
        self.repo, self.file = None, self.__path.basename
        self.__llm = None
        if not self.__path.exists():
            # Hugging Face Hub reference: [owner]/[repository]/[filename]
            if len(l := str(repo_id_or_path).split("/")) == 3 and all(len(s) > 0 for s in l):
                from huggingface_hub import file_exists
                repo, file = "/".join(l[:2]), l[2]
                if file_exists(repo, file, token=_HF_TOKEN()):
                    self.repo, self.file = repo, file
                else:
                    raise ValueError(f"Model '{repo_id_or_path}' does not exist on Hugging Face Hub")
            else:
                raise OSError(f"Model '{self.path}' does not exist locally")
    
    @staticmethod
    @lru_cache
    def grammar(gbnf):
        """ Compile (once) a GBNF grammar. """
        from llama_cpp import LlamaGrammar
        return LlamaGrammar.from_string(gbnf, verbose=False)
    
    @staticmethod
    def list():
        """ List cached models as references relative to the configured 'llm_cache'. """
        if not (root := config['llm_cache']).is_dir():
            return []
        return sorted(str(p.relative_to(root)) for p in root.rglob("*.gguf") if p.is_file())

