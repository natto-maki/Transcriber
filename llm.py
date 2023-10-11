import dataclasses
import threading as th
import importlib

import main_types as t
import tools


handler_openai = "OpenAI"
handler_llama_cpp = "Llama.cpp"

handlers = [handler_openai, handler_llama_cpp]


@dataclasses.dataclass
class LmOptions:
    input_language: str = "ja"
    output_language: str = "ja"
    handler: str = handler_openai
    options: dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class OpenAiOptions:
    model_for_step1: str = "gpt-3.5-turbo-0613"
    model_for_step2: str = "gpt-4-0613"


@dataclasses.dataclass
class LlamaCppOptions:
    model: str = ""


class Handler:
    def qualify(self, sentences: list[t.Sentence], opt: LmOptions | None, timeout) -> tools.AsyncCallFuture:
        raise NotImplementedError()

    def low_latency_interpretation(self, in_language: str | None, out_language: str, text: str) -> str:
        return "[n/a]"


_lock0 = th.Lock()
_handler_cache = {}


def _get_handler(opt: LmOptions | None = None) -> Handler:
    if opt is None:
        opt = LmOptions()

    with _lock0:
        if opt.handler not in _handler_cache:
            module_name = "llm_openai"
            if opt.handler == handler_llama_cpp:
                module_name = "llm_llama_cpp"
            # else ...

            _handler_cache[opt.handler] = importlib.import_module(module_name).get_handler(opt)

        return _handler_cache[opt.handler]


def qualify(sentences: list[t.Sentence], opt: LmOptions | None = None, timeout=180.0) -> tools.AsyncCallFuture:
    return _get_handler(opt).qualify(sentences, opt, timeout)


def low_latency_interpretation(in_language: str | None, out_language: str, text: str, opt: LmOptions | None = None) \
        -> str:
    return _get_handler(opt).low_latency_interpretation(in_language, out_language, text)
