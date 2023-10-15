import dataclasses
import logging
import urllib.error
import time
import re
import json
import threading as th
import concurrent.futures

import openai
import tiktoken
# noinspection PyPackageRequirements
import i18n

import main_types as t
import tools
import llm
import llm_tools

_implied_tokens_per_request = 3


def _num_tokens_from_messages(messages, model: str):
    """
    Return the number of tokens used by a list of messages.
    code from: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        logging.warning("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        logging.warning(
            "Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return _num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        logging.warning("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return _num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}"""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def _num_tokens_from_message(role: str, content: str, model: str):
    return (_num_tokens_from_messages([{"role": role, "content": content}], model)
            - _implied_tokens_per_request)


def _check_token_limits(messages, model_name):
    if _num_tokens_from_messages(messages, model_name) > 3096:
        raise RuntimeError(
            "The number of tokens exceeds the limit of 3096; messages = %s" %
            json.dumps(messages, indent=2, ensure_ascii=False))


def _invoke(messages, model_name: str):
    try_count = 0
    r0 = None
    while True:
        try_count += 1
        try:
            r0 = openai.ChatCompletion.create(
                model=model_name,
                messages=messages
            )
            break
        except openai.error.AuthenticationError as ex:
            raise ex
        except (urllib.error.HTTPError, openai.OpenAIError) as ex:
            if try_count >= 3:
                raise ex
            time.sleep(10)
            continue

    finish_reason = r0["choices"][0]["finish_reason"]
    if finish_reason != "stop":
        raise RuntimeError("API finished with unexpected reason: " + finish_reason)

    return r0["choices"][0]["message"]["content"]


def _invoke_with_retry(messages, model_name: str, post_process=None):
    _check_token_limits(messages, model_name)
    r0 = ""
    for _ in range(3):
        r0 = _invoke(messages, model_name)
        if post_process is None:
            return r0
        processed, r1 = post_process(r0)
        if not processed:
            logging.info("llm._invoke_with_retry: retry")
            continue
        return r1
    else:
        raise RuntimeError(
            "Invalid response returned more than the specified number of times;"
            " messages = %s, last response = \"%s\"" % (json.dumps(messages, indent=2, ensure_ascii=False), r0))


# deprecated. Keep definitions in order to be able to read configuration files created in older versions.
@dataclasses.dataclass
class QualifyOptions:
    input_language: str = "ja"
    output_language: str = "ja"
    model_for_step1: str = "gpt-3.5-turbo-0613"
    model_for_step2: str = "gpt-4-0613"


_qualify_p0_template_no_embeddings = '''\
The following %(source_language_descriptor)s text is a mechanical transcription of a conversation during a meeting.
Please correct this sentence and extract only what makes sense as a meeting conversation.
To do so, please correct what you assume to be transcription errors and remove fillers and rephrasing.
%(output_language_descriptor)s'''

_qualify_p0_template_with_embeddings = '''\
The following %(source_language_descriptor)s text is a mechanical transcription of a conversation during a meeting.
Please correct this sentence and extract only what makes sense as a meeting conversation.
To do so, please correct what you assume to be transcription errors and remove fillers and rephrasing.
Each line of input text is the speaker's name followed by ":" and then the content of the statement.
Output should maintain the same format.
%(output_language_descriptor)s'''

_qualify_p1_template = '''\
The following text is the transcribed minutes of a conversation during a meeting.
From this transcript, please extract a summary and action items.
The summary should include only proper nouns or the content of the discussion,
and should not be supplemented with general knowledge or known facts.
Action items should only include items explicitly mentioned by participants in the agenda, 
and should not include speculation.
Only one summary should be printed following the "Summary:" and the action item should be prefixed with "Action item:".
For example, the format is as follows:
%(output_example_descriptor)s
If there is no particular information to be output, or if there is not enough information for the summary,
just output "none".'''

_qualify_p2_template = '''\
The following %(source_language_descriptor)s text is a mechanical transcription of a conversation during a meeting.
From this transcript, please extract a summary and action items.
Transcription errors to closely pronounced words in the input text should be corrected, 
and fillers and rephrasing should be ignored.
The summary should include only proper nouns or the content of the discussion,
and should not be supplemented with general knowledge or known facts.
Action items should only include items explicitly mentioned by participants in the agenda, 
and should not include speculation.
Only one summary should be printed following the "Summary:" and the action item should be prefixed with "Action item:".
For example, the format is as follows:
%(output_example_descriptor)s
If there is no particular information to be output, or if there is not enough information for the summary,
just output "none".'''

_source_language_descriptor = {
    "en": "English",
    "ja": "Japanese"
}

_output_language_descriptor_for_p0 = {
    "en": {
        "default":
            "Output should be in English. Names of non-English spelling should not be converted to English, "
            "but should be retained in their original spelling.",
        "translate":
            "Please translate the output into English. "
            "Names of non-English spelling should not be converted to English, "
            "but should be retained in their original spelling.",
    },
    "ja": {
        "default":
            "出力は日本語にしてください。ただし、日本語表記ではない人名は日本語に変換せず、原表記を維持してください。",
        "translate":
            "日本語に翻訳して出力してください。ただし、日本語表記ではない人名は日本語に変換せず、原表記を維持してください。",
    }
}

_output_example_descriptor_for_p1 = {
    "en":
        "Summary: An example of summary. Please use sentence form, not a list of words.\n"
        "Action item: An example of an action item.\n"
        "Action item: There can be more than one action item.\n"
        "The words after \":\" should be written in English. "
        "However, names of non-English spelling should not be converted to English, "
        "but should be retained in their original spelling.",
    "ja":
        "Summary: 要点の例。文章にしてください。\n"
        "Action item: アクションアイテムの例\n"
        "Action item: アクションアイテムは複数になることもあります。\n"
        "\":\" 以降は日本語にしてください。ただし、日本語表記ではない人名は日本語に変換せず、原表記を維持してください。",
}


def _qualify_p0_system(opt: llm.LmOptions, with_embeddings: bool):
    return (_qualify_p0_template_with_embeddings if with_embeddings else _qualify_p0_template_no_embeddings) % {
        "source_language_descriptor": _source_language_descriptor[opt.input_language],
        "output_language_descriptor": _output_language_descriptor_for_p0[opt.output_language][
            "default" if opt.input_language == opt.output_language else "translate"]
    }


def _qualify_p1_system(opt: llm.LmOptions):
    return _qualify_p1_template % {
        "output_example_descriptor": _output_example_descriptor_for_p1[opt.output_language]
    }


def _qualify_p2_system(opt: llm.LmOptions):
    return _qualify_p2_template % {
        "source_language_descriptor": _source_language_descriptor[opt.input_language],
        "output_example_descriptor": _output_example_descriptor_for_p1[opt.output_language]
    }


def _correct_sentences_no_embeddings(sentences: list[t.Sentence], model_name: str, opt: llm.LmOptions) -> str:
    text = llm_tools.aggregate_sentences_no_embeddings(sentences)
    if len(text) == 0:
        return ""

    messages = [
        {"role": "system", "content": _qualify_p0_system(opt, with_embeddings=False)},
        {"role": "user", "content": text}
    ]

    return _invoke_with_retry(messages, model_name)


def _correct_sentences_with_embeddings(
        sentences: list[t.Sentence], model_name: str, opt: llm.LmOptions) -> list[t.Sentence]:

    text = llm_tools.aggregate_sentences_with_embeddings(sentences)
    if len(text) == 0:
        return []

    messages = [
        {"role": "system", "content": _qualify_p0_system(opt, with_embeddings=True)},
        {"role": "user", "content": text}
    ]

    name_to_id = {llm_tools.get_name(s): s.person_id for s in sentences if s.person_id != -1}

    def _post_process(r0_):
        r1_ = re.findall(r"([^:]+): (.+)\n*", r0_)
        return r1_ is not None and len(r1_) != 0, r1_

    r1 = _invoke_with_retry(messages, model_name, _post_process)

    return [t.Sentence(
        sentences[0].tm0, sentences[-1].tm1, e[1], person_name=e[0],
        person_id=name_to_id[e[0]] if e[0] in name_to_id else -1) for e in r1]


def _summarize_sub(sentences: list[t.Sentence] | str, model_name: str, prompt: str) -> tuple[str, list[str]]:
    text = llm_tools.aggregate_sentences_with_embeddings(sentences) if isinstance(sentences, list) else sentences
    if len(text) == 0:
        return "", []

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": text}
    ]

    def _is_none(v_):
        return re.search(r"[Nn]one\.?", v_) is not None

    def _post_process(r0_):
        r1_ = re.findall(r"[Ss]ummary: (.+)\n*", r0_)
        if r1_ is None or len(r1_) != 1:
            return False, None
        r1_[0] = "(なし)" if _is_none(r1_[0]) else r1_[0]
        r2_ = re.findall(r"[Aa]ction item: (.+)\n*", r0_)
        r2_ = [] if r2_ is None else list(filter(lambda e_: not _is_none(e_), r2_))
        return True, (r1_, r2_)

    r1, r2 = _invoke_with_retry(messages, model_name, _post_process)
    return r1[0], r2


def _summarize(sentences: list[t.Sentence] | str, model_name: str, opt: llm.LmOptions) -> tuple[str, list[str]]:
    return _summarize_sub(sentences, model_name, _qualify_p1_system(opt))


def _qualify(sentences: list[t.Sentence], model_name: str, opt: llm.LmOptions) -> tuple[str, list[str]]:
    return _summarize_sub(sentences, model_name, _qualify_p2_system(opt))


def _qualify_procedure(sentences: list[t.Sentence], opt: llm.LmOptions) -> t.QualifiedResult:
    has_embedding = (len([None for s in sentences if s.embedding is not None]) != 0)

    try:
        openai_opt = (opt.options[llm.handler_openai] if llm.handler_openai in opt.options else llm.OpenAiOptions())
        corrected = _correct_sentences_with_embeddings(sentences, openai_opt.model_for_step1, opt) \
            if has_embedding else _correct_sentences_no_embeddings(sentences, openai_opt.model_for_step1, opt)
        summaries, action_items = _summarize(corrected, openai_opt.model_for_step2, opt)
    except openai.error.AuthenticationError:
        return t.QualifiedResult(
            corrected_sentences=sentences,
            summaries=i18n.t('app.qualify_llm_authentication_error'),
            action_items=[]
        )

    return t.QualifiedResult(
        corrected_sentences=sentences,  # keep original
        summaries=summaries,
        action_items=action_items
    )


def _handle_qualify(sentences: list[t.Sentence], opt: llm.LmOptions, timeout) -> tools.AsyncCallFuture:
    return tools.async_call(
        _qualify_procedure, [s.clone() for s in sentences], dataclasses.replace(opt), timeout=timeout)


_interpret_p0_template = {
    "en": {
        "default": "Please translate it into clean English.",
        "ja": "Please translate it into clean English. Input text is in Japanese."
    },
    "ja": {
        "default": "Please translate it into clean Japanese.",
        "en": "Please translate it into clean Japanese. Input text is in English."
    }
}

_interpretation_executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)
_interpretation_task_count = th.Semaphore(1)


def _low_latency_interpretation_procedure(in_language: str | None, out_language: str, text: str) -> str:
    try:
        if out_language not in _interpret_p0_template:
            return "[unknown]"
        prompt_src_keyed = _interpret_p0_template[out_language]
        prompt = prompt_src_keyed[in_language if in_language in prompt_src_keyed else "default"]

        return _invoke([
            {"role": "system", "content": prompt},
            {"role": "user", "content": text}
        ], "gpt-3.5-turbo-0613")

    except Exception as ex:
        _ = ex
        return "[error]"


def _low_latency_interpretation_caller(ct):
    _interpretation_task_count.release()
    ct[1] = _low_latency_interpretation_procedure(*ct[2:])
    ct[0].release()


def _handle_low_latency_interpretation(in_language: str | None, out_language: str, text: str) -> str:
    if not _interpretation_task_count.acquire(blocking=False):
        return "[skip]"
    ct = [th.Semaphore(0), None, in_language, out_language, text]
    _interpretation_executor.submit(_low_latency_interpretation_caller, ct)
    if ct[0].acquire(timeout=15.0):
        return ct[1]
    return "[timeout]"


class _OpenAiHandler(llm.Handler):
    def qualify(self, sentences: list[t.Sentence], opt: llm.LmOptions, timeout) -> tools.AsyncCallFuture:
        return _handle_qualify(sentences, opt, timeout)

    def low_latency_interpretation(self, in_language: str | None, out_language: str, text: str) -> str:
        return _handle_low_latency_interpretation(in_language, out_language, text)


def get_handler(opt: llm.LmOptions):
    _ = opt
    return _OpenAiHandler()
