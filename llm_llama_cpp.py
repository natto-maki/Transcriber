import os
import subprocess
import logging
import dataclasses

# https://github.com/abetlen/llama-cpp-python
#
# prerequisites:
#   sudo apt install -y cmake clang lldb lld wget
#
# ubuntu+GPU
#   CUDACXX=/usr/local/cuda/bin/nvcc CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip3 install llama-cpp-python
# x86 Mac (very slow)
#   CMAKE_ARGS="-DLLAMA_METAL=off" pip3 install llama-cpp-python
# M1 Mac -> always aborted?
#   CMAKE_ARGS="-DLLAMA_METAL=on" pip3 install llama-cpp-python
#
# noinspection PyPackageRequirements
from llama_cpp import Llama

import main_types as t
import tools
import llm_tools


# https://huggingface.co/TFMC
_model_file_name = "ggml-model-q4_m.gguf"
_model_file_url = "https://huggingface.co/TFMC/openbuddy-llama2-13b-v11.1-bf16-GGUF/resolve/main/ggml-model-q4_m.gguf"


@dataclasses.dataclass
class QualifyOptions:
    input_language: str = "ja"
    output_language: str = "ja"
    model: str = ""


def _check_dependency():
    if not os.path.isfile(_model_file_name):
        r0 = subprocess.run(["wget", _model_file_url])
        if r0.returncode != 0:
            raise RuntimeError("Cannot download model file - %s" % _model_file_url)


_check_dependency()
_model = Llama(model_path=_model_file_name, n_ctx=2048, n_gpu_layers=100)


def _invoke(messages, model_name: str | None = None):
    _ = model_name  # TODO model_name -> _model instance

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

    r0 = _model.create_completion(
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


def _qualify_procedure(sentences: list[t.Sentence], opt: QualifyOptions | None = None) -> t.QualifiedResult:
    if opt is None:
        opt = QualifyOptions()

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


def qualify(
        sentences: list[t.Sentence],
        opt: QualifyOptions | None = None, timeout=180.0) -> tools.AsyncCallFuture:
    _ = timeout
    return tools.async_call(
        _qualify_procedure, [s.clone() for s in sentences], dataclasses.replace(opt), timeout=600.0)


def low_latency_interpretation(in_language: str | None, out_language: str, text: str) -> str:
    return "[n/a]"
