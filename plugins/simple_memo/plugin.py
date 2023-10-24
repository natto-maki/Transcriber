import logging
import datetime
import re
import concurrent.futures
import urllib.error
import html
import threading as th

import transcriber_plugin as pl
import gradio as gr

# The api_key of the openai module has already been set on the main module
import openai

# from . import import_local_file_using_this_style

_trigger_words_regex = [r"[Tt]ake a memo", r"(メモ|目も)(を)?(とって|取って|して|保存|作成|記録)"]
_timeout_for_end_of_memo = 5.0

_qualify_prompt = '''\
The following text is a written memo of what the user has said.
Please summarize only the main points from this memo and format it in Markdown format for output.
Do not supplement information not included in the original text.
The opening sentence "%(trigger_words)s" is an instruction to start writing the memo,
which should not be included in the output.
Output only the contents of the memo, do not add extra output before or after.
%(language)s
'''

_qualify_prompt_language = {
    "en": "Output should be in English.",
    "ja": "出力は日本語にしてください。"
}

_history_table_header = '''\
<main class="current">
<section class="historyBlock">
<table width="100%%">
<tr>
<th width="20%%">Time</th>
<th width="40%%">Memo</th>
<th width="40%%">Original text</th>
</tr>
'''

_history_table_footer = '''\
</table>
</section>
</main>
'''

_description_trigger_words = "\n".join(["<li>" + html.escape(regex) for regex in _trigger_words_regex])
_description = f'''\
<p>This is a sample plugin implementation that provides a simple memo recording function.</p>
<p>When the following trigger words appear during a conversation,
the subsequent sentences are saved as memos;
the statements are saved until a {_timeout_for_end_of_memo}-second silent interval is detected, 
and finally GPT-4 is used to extract the main points.</p>
<ul>
{_description_trigger_words}
</ul>
<p>This is just a sample implementation, so the generated notes will not be stored in storage.</p>
'''


class _SimpleMemo(pl.Plugin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__tm0 = -1.0
        self.__tm1 = -1.0
        self.__matched_trigger_words = ""
        self.__text = ""
        self.__executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        self.__lock0 = th.Lock()
        self.__generation = 1
        self.__history = []

    def injection_point(self) -> int:
        return pl.FLAG_SPEECH_SEGMENT | pl.FLAG_ADD_TAB

    def __qualify(self, generation, index, tm0, matched_trigger_words, text):
        messages = [
            {"role": "system", "content": _qualify_prompt % {
                "trigger_words": matched_trigger_words,
                "language": _qualify_prompt_language[self._output_language]}},
            {"role": "user", "content": text}
        ]
        try:
            r0 = openai.ChatCompletion.create(model="gpt-4-0613", messages=messages)
        except (urllib.error.HTTPError, openai.OpenAIError) as ex:
            logging.error("Failed to process text", exc_info=ex)
            return

        finish_reason = r0["choices"][0]["finish_reason"]
        if finish_reason != "stop":
            logging.error("API finished with unexpected reason: " + finish_reason)
            return

        qualified_text = r0["choices"][0]["message"]["content"]
        with self.__lock0:
            if generation == self.__generation:
                self.__history[index] = [tm0, qualified_text, text]

    def on_speech_segment(self, tm0: float, tm1: float, person_name: str | None, text: str | None):
        if self.__tm0 < 0.0:
            if tm0 >= 0.0:
                for pattern in _trigger_words_regex:
                    r = re.search(pattern, text)
                    if r is not None:
                        break
                else:
                    return
                self.__tm0 = tm0
                self.__tm1 = tm1
                self.__matched_trigger_words = r.group(0)
                self.__text = text[r.start(0):]
        else:
            if tm0 >= 0.0:
                self.__tm1 = tm1
                self.__text += text if len(self.__text) == 0 else " " + text
            else:
                if self.__tm1 + _timeout_for_end_of_memo < tm1:
                    with self.__lock0:
                        generation = self.__generation
                        index = len(self.__history)
                        self.__history.append([self.__tm0, "", self.__text])
                    self.__executor.submit(
                        self.__qualify, generation, index, self.__tm0, self.__matched_trigger_words, self.__text)
                    self.__tm0 = -1.0

    def tab_name(self) -> str:
        return "Simple Memo"

    def __get_history(self):
        with self.__lock0:
            if len(self.__history) == 0:
                return ""
            return _history_table_header + "\n".join([
                "<tr><td>%s</td><td>%s</td><td>%s</td></tr>" % (
                    datetime.datetime.fromtimestamp(e[0]).ctime(),
                    html.escape(e[1]),
                    html.escape(e[2])
                )
                for e in self.__history
            ]) + _history_table_footer

    def __clear_history(self):
        with self.__lock0:
            self.__generation += 1
            self.__history.clear()
        return ""

    def build_tab(self):
        gr.HTML(value=_description)
        f_clear = gr.Button("Clear")
        f_history = gr.HTML(value=self.__get_history, every=2)
        f_clear.click(self.__clear_history, None, [f_history])

    def read_state(self, read_parameters: str) -> str:
        return "parameter = " + read_parameters


def create(**kwargs):
    return _SimpleMemo(**kwargs)
