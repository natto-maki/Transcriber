import os
import subprocess
import logging
import dataclasses
import threading as th

# https://github.com/abetlen/llama-cpp-python
#
# prerequisites:
#   sudo apt install -y cmake clang lldb lld wget
#
# ubuntu+GPU
#   CUDACXX=/usr/local/cuda/bin/nvcc CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip3 install llama-cpp-python
# x86 Mac (very slow)
#   CMAKE_ARGS="-DLLAMA_METAL=off" pip3 install llama-cpp-python
# M1 Mac -> always aborted? (OOM?)
#   CMAKE_ARGS="-DLLAMA_METAL=on" pip3 install llama-cpp-python
#
# noinspection PyPackageRequirements
from llama_cpp import Llama

import llm
import main_types as t
import tools
import llm_tools


_model_table = {
    # https://huggingface.co/TFMC
    "ggml-model-q4_m": {
        "file": "ggml-model-q4_m.gguf",
        "url": "https://huggingface.co/TFMC/openbuddy-llama2-13b-v11.1-bf16-GGUF/resolve/main/ggml-model-q4_m.gguf"
    }
}

_default_model_name = "ggml-model-q4_m"
_model_dir = "models"

_lock0 = th.Lock()
_current_model_name = ""
_current_model: Llama | None = None


def _load_model(model_name: str | None):
    global _current_model_name, _current_model

    if _current_model_name == model_name:
        return

    _current_model_name = ""
    _current_model = None

    model_name = (model_name if model_name in _model_table else _default_model_name)
    m = _model_table[model_name]
    file_path = os.path.join(_model_dir, m["file"])

    os.makedirs(_model_dir, exist_ok=True)
    if not os.path.isfile(file_path):
        r0 = subprocess.run(["wget", m["url"]], cwd=_model_dir)
        if r0.returncode != 0:
            logging.error("Cannot download model file - %s" % m["url"])
            return

    _current_model = Llama(model_path=file_path, n_ctx=2048, n_gpu_layers=100)
    _current_model_name = model_name


def _invoke(messages, model_name: str | None = None):
    with _lock0:
        _load_model(model_name)
        if _current_model is None:
            return "[n/a]"

        prompt_list = []
        for m in messages:
            role = m["role"]
            if role == "system":
                prompt_list.append(m["content"])
            elif role == "user":
                prompt_list.append("USER: " + m["content"])
            elif role == "assistant":
                prompt_list.append("ASSISTANT: " + m["content"])
            else:
                raise ValueError()
        prompt_list.append("ASSISTANT: ")

        r0 = _current_model.create_completion(
            "\n".join(prompt_list),
            temperature=0.7, top_p=0.3, top_k=40, repeat_penalty=1.1, max_tokens=1024,
            stop=["ASSISTANT:", "USER:", "SYSTEM:"],
            stream=False)

        print("\n".join(prompt_list))
        print(r0)

        finish_reason = r0["choices"][0]["finish_reason"]
        if finish_reason != "stop":
            raise RuntimeError("LLM finished with unexpected reason: " + finish_reason)

        return r0["choices"][0]["text"]


_qualify_prompt = '''\
What is the main message in this conversation?
'''


def _qualify_procedure(sentences: list[t.Sentence], opt: llm.LmOptions | None = None) -> t.QualifiedResult:
    _ = opt

    has_embedding = (len([None for s in sentences if s.embedding is not None]) != 0)
    text = llm_tools.aggregate_sentences_with_embeddings(sentences) \
        if has_embedding else llm_tools.aggregate_sentences_no_embeddings(sentences)

    summaries = _invoke(messages=[
        {"role": "system", "content": _qualify_prompt},
        {"role": "user", "content": text}
    ])

    return t.QualifiedResult(
        corrected_sentences=sentences,
        summaries=summaries,
        action_items=[]
    )


def _handle_qualify(sentences: list[t.Sentence], opt: llm.LmOptions | None, timeout) -> tools.AsyncCallFuture:
    _ = timeout
    return tools.async_call(
        _qualify_procedure, [s.clone() for s in sentences], dataclasses.replace(opt), timeout=600.0)


class LlamaCppHandler(llm.Handler):
    def qualify(self, sentences: list[t.Sentence], opt: llm.LmOptions | None, timeout) -> tools.AsyncCallFuture:
        return _handle_qualify(sentences, opt, timeout)


def get_handler(opt: llm.LmOptions):
    _ = opt
    return LlamaCppHandler()
