# -*- coding: UTF-8 -*-
from tinyscript import logging


__all__ = ["get_context"]


def get_context(context_manager, **kwargs):
    """ Context factory, instantiating the strategy matching a 'context_manager' classifier parameter value.

    Parameters
    ----------
    context_manager : str
        One of 'none', 'messages', 'kv_cache', 'llama_state' (see :class:`~.LLMBinaryClassifier`'s 'context_manager'
         parameter) ; 'messages' gives a :class:`ConversationContext`, growing with every prediction.
    """
    c = {'none': NullContext, 'messages': ConversationContext, 'kv_cache': KVCacheContext,
         'llama_state': LlamaStateContext}
    try:
        cls = c[cm := context_manager]
    except KeyError:
        raise ValueError(f"unknown context manager '{cm}' (should be one of: {', '.join(c.keys())})")
    return cls(**kwargs)


class BaseContext:
    """ Base class for context management strategies, controlling how state shared across successive per-sample LLM
         calls (system prompt, schema description, few-shot examples, conversation history) is primed, kept and,
         where relevant, grown between predictions.
    
    Parameters
    ----------
    logger : Logger, default=None
        Logger to report context lifecycle events (cache setup, ...).
    """
    def __init__(self, logger=None, prefix=None, **kwargs):
        self.logger = logger or logging.getLogger()
        self._messages = []
    
    def restore(self, *args, **kwargs):
        pass
    
    def render(self):
        """ Render the current messages as a plain text block to be prepended to the per-sample prompt. """
        if len(self.messages) == 0:
            return ""
        if len(self.messages) > 0:
            if isinstance(m := self.messages[0], str):
                return "\n".join(self.messages)
            elif isinstance(m, tuple) and len(m) == 2:
                return "".join(f"{role.capitalize()}: {content}\n" for role, content in self.messages)
        raise ValueError("unknown messages format (should be either list of strings or list of tuples (role, content)")
    
    def update(self, *args, **kwargs):
        pass
    
    @property
    def messages(self):
        return self._messages


class NullContext(BaseContext):
    """ No state kept from one sample to another ; every prompt is self-contained, i.e. built from the template alone,
         which may still provide the same few-shot examples at each classification. """
    pass


class MessageContext(BaseContext):
    """ Keep a fixed list of turns (i.e. few-shot examples) and render them ahead of every prompt. """
    def __init__(self, messages=None, **kwargs):
        super().__init__(**kwargs)
        self._messages.extend(messages or [])


class ConversationContext(MessageContext):
    """ Grow the message history with every prediction, turning the per-sample calls into a single, ever-growing
         conversation (context_manager='messages' ; see :func:`get_context`). """
    def update(self, user, assistant):
        """ Record a completed (user, assistant) turn ; no-op unless the context grows a history. """
        self._messages.append(("user", user))
        self._messages.append(("assistant", assistant))


class KVCacheContext(MessageContext):
    """ Rely on llama.cpp's own longest-common-prefix KV cache (``llama_cpp.LlamaRAMCache``) to avoid reprocessing
         the shared prompt prefix (system prompt, schema, few-shot examples) at every call. That prefix is still part
         of every per-sample prompt, as rendered from the template ; only its evaluation gets cached, which is where
         the speedup comes from, the same prefix recurring across samples.
    
    :param capacity_bytes : maximum size of the in-memory KV cache (default=2 << 30 (2 GiB))
    """
    def __init__(self, capacity_bytes=2 << 30, backend=None, **kwargs):
        from llama_cpp import LlamaRAMCache
        super().__init__(**kwargs)
        backend.model.set_cache(LlamaRAMCache(capacity_bytes=capacity_bytes))
        self.logger.debug(f"Enabled llama.cpp KV cache ({capacity_bytes} bytes)")


class LlamaStateContext(MessageContext):
    """ Explicitly prime and restore the llama.cpp KV cache state for the shared prompt prefix (system prompt, schema,
         few-shot examples), instead of relying on llama.cpp's own hash-based longest-prefix lookup
         (:class:`KVCacheContext`). The prefix is tokenized and evaluated once, on instantiation, and the resulting
         state (``llama_cpp.llama.LlamaState``) is saved ; that same single state is then force-loaded via
         ``Llama.load_state()`` before every per-sample call ('restore'), deterministically resetting the backend to
         just after the shared prefix regardless of what the previous call left it in, instead of trusting the
         longest-prefix match against whatever happens to still be resident. The prefix is still part of every
         per-sample prompt, as rendered from the template, but since its tokens already sit in the backend's restored
         KV cache, only the per-sample suffix actually gets (re-)evaluated.

    Unlike :class:`KVCacheContext`, no new state gets appended to a growing cache on every call ; a single state is
     primed once and reused, so there is no eviction/capacity to configure.
    """
    def __init__(self, backend=None, prefix="", **kwargs):
        super().__init__(**kwargs)
        self.__state = None
        if prefix := (prefix or self.render()):
            model = backend.model
            model.reset()
            model.eval(model.tokenize(prefix.encode("utf-8")))
            self.__state = model.save_state()
            self.logger.debug(f"Primed llama.cpp KV cache state for a {model.n_tokens}-token prefix")
    
    def restore(self, llm):
        """ Called before every per-sample inference to put the backend into the right state. """
        if self.__state is not None:
            llm.model.load_state(self.__state)

