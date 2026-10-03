# -*- coding: UTF-8 -*-


def estimate_ctx(static_tokens,max_sample,max_output,safety=256):
    required = static_tokens + max_sample + max_output + safety
    for c in (2048, 4096, 8192, 16384, 32768):
        if required <= c:
            return c
    return required

