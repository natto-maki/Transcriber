import os
import logging
import sys
import time
import datetime
import html
import threading as th
import pickle
import locale
import dataclasses
import re
import concurrent.futures

# noinspection PyPackageRequirements
import i18n
import gradio as gr
import numpy as np
import openai
import sounddevice as sd
import soundfile as sf
# noinspection PyPackageRequirements
import iso639

import tools
import main_types as t
import llm_openai as llm
import main
import transcriber_plugin as pl


@dataclasses.dataclass
class UiConfiguration:
    language: str = "auto"
    show_input_status: bool = False
    show_statement_properties: bool = False
    openai_api_key: str = ""


_app_lock0 = th.Lock()
_app: main.Application | None = None
_conf: main.Configuration | None = None
_ui_conf: UiConfiguration | None = None
_last_called = 0.0
_live_checker_thread: th.Thread | None = None
_playback_audio_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)


def _keep_alive():
    global _last_called

    _last_called = time.time()
    if not _app.is_opened():
        with _app_lock0:
            logging.info("opening application...")
            _app.open()
            logging.info("application opened")


def _live_checker():
    while True:
        time.sleep(5.0)
        if _app.is_opened() and _last_called + 5.0 < time.time():
            with _app_lock0:
                logging.info("closing application...")
                _app.close()
                logging.info("application closed")


def _restart(conf: main.Configuration):
    global _app, _conf
    with _app_lock0:
        logging.info("restarting application...")
        if _app.is_opened():
            _app.close()
        _app = main.Application(conf, _ui_conf.language)
        _conf = _app.get_current_configuration()
        logging.info("application restarted")


block_css = '''\
footer {
  visibility: hidden;
}
main.current {
  height: calc(100vh - 120px);
  display: flex;
  flex-direction: column;
  overflow-y: hidden;
}
main.history {
  height: calc(100vh - 227px);
  display: flex;
  flex-direction: column;
}
main.personList {
  /* height: calc(100vh - 376px); */
  display: flex;
  flex-direction: column;
}
.historyBlock {
  flex: 1;
  overflow: auto;
}
table.sentences {
  width: 100%;
  margin-bottom: 0px !important;
}
table.sentences tr {
}
table.sentences tr td {
  border-bottom: 0px solid transparent !important;
  padding: 2px 0px !important;
}
table.sentences tr td.control {
  border-bottom: 0px solid transparent !important;
  padding: 2px 0px !important;
  text-align: right !important;
}
table.sentences tr td.sys_language_detected {
  background-color: var(--primary-500);
  padding: 2px 6px !important;
  border-top: 8px solid transparent !important;
  border-bottom: 8px solid transparent !important;
}
span.talker {
  color: #808080 !important;
  font-weight: bold;
}
span.talkerExtra {
  color: #808080 !important;
}
span.error {
  color: #FF4040 !important;
}
span.status {
  color: #808080 !important;
}
span.suppressed {
  color: #808080 !important;
}
span.processing {
  color: #A0A0A0 !important;
}
span.interpretation {
  color: var(--primary-200) !important;
}
span.playback_audio {
}
span.playback_audio:hover {
  background: rgba(255,255,255,.4);
}
span.playback_audio:active {
  background: rgba(255,255,255,.8);
}
img.playback_audio_base {
  width: 24px;
  height: 24px;
  outline: none;
  margin: 0px;
  padding: 0px;
  position: relative;
  display: inline-block;
  fill: #808080;
}
img.playback_audio {
}
img.playback_audio:hover {
  background: rgba(255,255,255,.4);
}
img.playback_audio:active {
  background: rgba(255,255,255,.8);
}
'''

text_table_header = '''\
<main class="current">
<section class="historyBlock">
<table width="100%%">
<tr>
<th width="60px">%(time)s</th>
<th width="calc(45%% - 60px)">%(summary)s</th>
<th width="55%%">%(conversation)s</th>
</tr>
'''

text_table_footer = '''\
</table>
</section>
</main>
'''

move_to_last_js = '''\
() => {
  const d = document.getElementById("lastRow");
  d.scrollIntoView(false);
}
'''

history_text_table_header = '''\
<main class="history">
<section class="historyBlock">
<table width="100%%">
<tr>
<th width="60px">%(time)s</th>
<th width="calc(45%% - 60px)">%(summary)s</th>
<th width="55%%">%(conversation)s</th>
</tr>
'''

history_text_table_footer = '''\
</table>
</section>
</main>
'''

person_list_table_header = '''\
<main class="personList">
<section class="historyBlock">
<table width="100%%">
<tr>
<th width="20%%">%(person_id)s</th>
<th width="25%%">%(superseded_by)s</th>
<th width="20%%">%(name)s</th>
<th width="35%%">%(last_mapped_time)s</th>
</tr>
'''

person_list_table_footer = '''\
</table>
</section>
</main>
'''


def _can_be_merged(s0: t.Sentence, s1: t.Sentence):
    if s0.person_id == -1 or s0.person_id != s1.person_id:
        return False
    if s0.si_state is None and s1.si_state is None:
        return True
    if s0.si_state is None or s1.si_state is None:
        return False
    return True


def _merge_sentences(sentences: list[t.Sentence], merge_interval=10.0):
    ret: list[t.Sentence] = []
    for s in sentences:
        if len(ret) != 0:
            s0 = ret[-1]
            if s0.tm1 + merge_interval > s.tm0 and _can_be_merged(s0, s):
                s0.merge(s)
                continue
        ret.append(s.clone())
    return ret


def _merge_sentences_with_no_embeddings(sentences: list[t.Sentence], merge_interval=10.0):
    ret: list[t.Sentence] = []
    for s in sentences:
        if s.sentence_type == t.SentenceType.Sentence and len(ret) != 0:
            s0 = ret[-1]
            if s0.sentence_type == t.SentenceType.Sentence and s0.tm1 + merge_interval > s.tm0:
                s0.merge(s)
                continue
        ret.append(s.clone())
    return ret


_playback_audio_file_template = '''\
<img src="file/resources/playback_audio.png" class="playback_audio_base playback_audio"
  onclick='fetch("/api/playback_audio/", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({data: ["%(audio_file_names)s"]})
  });'>
'''

_playback_audio_file_template_for_span = '''\
<span class="playback_audio"
  onclick='fetch("/api/playback_audio/", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({data: ["%(audio_file_names)s"]})
  });'>
'''


def _playback_audio_task(files):
    logging.info("_playback_audio_task: %s" % files)
    audio_data_list = []
    for file in files.split(","):
        try:
            with sf.SoundFile(file, mode="r") as f:
                sampling_rate = f.samplerate
                audio_data_list.append(f.read(dtype="float32"))
        except Exception as ex:
            logging.info("Cannot open file - %s" % file, exc_info=ex)
            return

    if len(audio_data_list) == 0:
        return
    audio_data = np.concatenate(audio_data_list)

    app = _app
    app.suppress_audio_lock()
    try:
        time.sleep(0.5)
        sd.play(audio_data, samplerate=sampling_rate, blocking=True)
        time.sleep(0.5)
    except Exception as ex:
        logging.error("An error occurred while playing the audio file - %s" % files, exc_info=ex)
    finally:
        app.suppress_audio_unlock()


def _playback_audio(files):
    _playback_audio_executor.submit(_playback_audio_task, files)


def _get_valid_audio_files(s: t.Sentence) -> list[str] | None:
    if s.prop is None or s.prop.audio_file_name_list is None:
        return None
    ret = [name for name in s.prop.audio_file_name_list if os.path.isfile(name)]
    return ret if len(ret) != 0 else None


def _has_sentence_specific_properties(s: t.Sentence):
    return s.embedding is not None or _get_valid_audio_files(s) is not None


def _has_inline_playback(s: t.Sentence):
    return s.prop is not None and \
        s.prop.audio_file_name_list is not None and len(s.prop.audio_file_name_list) != 0 and \
        s.prop.audio_file_prop_list is not None and \
        len(s.prop.audio_file_name_list) == len(s.prop.audio_file_prop_list)


def _render_text(s: t.Sentence, length_limit: int | None = None) -> str:
    if not _has_inline_playback(s):
        return html.escape(s.text)

    if length_limit is None:
        length_limit = len(s.text)

    out = []
    audio_index = 0
    offset = 0
    while offset < length_limit and audio_index < len(s.prop.audio_file_prop_list):
        name = s.prop.audio_file_name_list[audio_index]
        prop = s.prop.audio_file_prop_list[audio_index]
        audio_index += 1

        if prop is None:
            continue
        if offset < prop.offset:
            out.append(html.escape(s.text[offset:prop.offset]))
            offset = prop.offset
            if offset >= length_limit:
                break

        is_valid = os.path.isfile(name)
        if is_valid:
            out.append(_playback_audio_file_template_for_span % {"audio_file_names": name})
        offset = prop.offset + prop.length
        out.append(html.escape(s.text[prop.offset:offset]))
        if is_valid:
            out.append("</span>")

    if offset < length_limit:
        out.append(html.escape(s.text[offset:length_limit]))

    return "".join(out)


def _get_language_name(code: str) -> str:
    key = "lang." + code
    name = i18n.t(key)
    if name != key:
        return name
    try:
        return iso639.Lang(code).name
    except Exception as ex:
        _ = ex
        return "?Unknown"


def _output_sentences(sentences: list[t.Sentence], show_properties=False):
    has_embedding = (len([None for s in sentences if s.embedding is not None]) != 0)
    out = []
    for s in (_merge_sentences(sentences) if has_embedding else _merge_sentences_with_no_embeddings(sentences)):
        if s.sentence_type == t.SentenceType.SentenceSeparator:
            continue

        s_audio_file = None
        if not _has_inline_playback(s):
            valid_files = _get_valid_audio_files(s)
            if valid_files is not None:
                s_audio_file = _playback_audio_file_template % {"audio_file_names": ",".join(valid_files)}

        s_td_class = ""
        if s.sentence_type == t.SentenceType.LanguageDetected:
            s_td_class = " class=\"sys_language_detected\""

        s_talker = ""
        if has_embedding:
            s_talker = "<span class=\"talker\">%s</span>" % (
                html.escape(s.person_name) if s.person_id != -1 else t.unknown_person_display_name)

        if s.si_state is None:
            s_rendered_text = _render_text(s)
        else:
            s_text = " ".join(s.si_state.processed_org)
            s_processing = " ".join([text for text in [s.si_state.processing] + s.si_state.waiting if text != ""])
            s_interpretation = s.si_state.processed_int
            s_rendered_text = "%s%s%s" % (
                _render_text(s, len(s_text)),
                " <span class=\"processing\">" + html.escape(s_processing) + "</span>"
                if s_processing != "" else "",
                "<br/><span class=\"interpretation\">" + html.escape(s_interpretation) + "</span>"
                if s_interpretation != "" else "")

        out.append(("<tr><td%s>" if s_audio_file is not None else "<tr><td%s colspan=\"2\">") % s_td_class)

        if s.sentence_type == t.SentenceType.LanguageDetected:
            out.append(i18n.t('app.text_language_detected') % {
                "old_language": _get_language_name(s.payload["old_language"]),
                "new_language": _get_language_name(s.payload["new_language"])})
        elif show_properties:
            s_tm = time.localtime(s.tm0)
            s_prop = _output_properties(s.prop) if s.prop is not None else ""
            out.append(
                "%s<span class=\"talkerExtra\">%02d:%02d.%02d%s</span><br/>%s" % (
                    s_talker + "  " if s_talker != "" else "", s_tm.tm_hour, s_tm.tm_min, s_tm.tm_sec, s_prop,
                    s_rendered_text))
        else:
            out.append("%s%s" % (s_talker + "<br/>" if s_talker != "" else "", s_rendered_text))

        out.append("</td>")
        if s_audio_file is not None:
            out.append("<td class=\"control\" width=\"32px\">%s</td>" % s_audio_file)
        out.append("</tr>")

    return "<table class=\"sentences\">" + "\n".join(out) + "</table>"


def _scaled_db(v):
    return (6.0 / np.log10(2.0)) * np.log10(max(v, sys.float_info.epsilon))


def _output_properties(prop: t.AdditionalProperties):
    return " Va:%.2f Vm:%.2f Lv:%.1fdB(%.1fdB)" % (
        prop.vad_ave_level, prop.vad_max_level, _scaled_db(prop.audio_level), _scaled_db(prop.segment_audio_level))


def _output_text(reader, include_anker=False):
    text = ""
    group_count = reader.group_count()
    for index in range(group_count):
        g = reader.ref_group(index)

        text += "<tr>" if not include_anker or index + 1 < group_count else "<tr id=\"lastRow\">"

        tm = time.localtime(g.sentences[0].tm0)
        text += "<td>%02d:%02d</td>" % (tm.tm_hour, tm.tm_min)

        if g.qualified is None:
            if g.state == t.SENTENCE_QUALIFY_ERROR:
                text += "<td><span class=\"error\">%s</span></td>" % i18n.t("app.text_error_in_qualifying")
                text += "<td><small>%s</small></td>" % _output_sentences(
                    g.sentences, show_properties=_ui_conf.show_statement_properties)
            else:
                text += "<td colspan=\"2\">%s</td>" % _output_sentences(
                    g.sentences, show_properties=_ui_conf.show_statement_properties)

        else:
            if g.state == t.SENTENCE_QUALIFY_ERROR:
                text += "<td><font color=\"red\">%s</font></td>" % i18n.t("app.text_error_in_qualifying")
            elif g.qualified is not None:
                action_items = ""
                if g.qualified.action_items is not None and len(g.qualified.action_items) != 0:
                    action_items += "<br/>\n<br/>\n"
                    for action_item in g.qualified.action_items:
                        action_items += "<strong>%s</strong> %s<br/>\n" % (
                            i18n.t("app.text_action_item"), html.escape(action_item))
                text += "<td>%s%s</td>" % (html.escape(g.qualified.summaries), action_items)
            else:
                text += "<td>...</td>"

            # "<td><small>%s</small></td>"
            text += "<td>%s</td>" % _output_sentences(
                g.qualified.corrected_sentences, show_properties=_ui_conf.show_statement_properties)

        text += "</tr>"

    return text


def _interval_update():
    if _app.log_file_may_changed():
        _restart(_load_configuration())
    _keep_alive()

    additional_rows = ""
    if _ui_conf.show_input_status:
        additional_rows = "<tr id=\"lastRow\"><td colspan=\"3\"><span class=\"status\">"
        m = _app.measure()
        additional_rows += "input: latency %.3fs, peak %.1fdB, RMS %.1fdB, %s%s<br/>" % (
            m.latency, m.peak_db, m.rms_db, "active" if m.woke else "sleep",
            ":VAD(max %.2f, ave %.2f)" % (m.vad_max, m.vad_ave) if m.woke else "")
        additional_rows += "</span></td></tr>"

    return text_table_header % {
        "time": i18n.t("app.text_table_header_time"),
        "summary": i18n.t("app.text_table_header_summary"),
        "conversation": i18n.t("app.text_table_header_conversation")
    } + _output_text(_app, include_anker=not _ui_conf.show_input_status) + additional_rows + text_table_footer


def _get_histories():
    return [
        datetime.datetime.strptime(
            "%04d %02d %02d" % (index // 10000, index // 100 % 100, index % 100),
            "%Y %m %d").strftime(i18n.t("app.date_format"))
        for index in _app.list_history()
    ]


def _update_history(selector):
    tm = datetime.datetime.strptime(selector, i18n.t("app.date_format"))
    date_index = tm.year * 10000 + tm.month * 100 + tm.day
    return history_text_table_header % {
        "time": i18n.t("app.text_table_header_time"),
        "summary": i18n.t("app.text_table_header_summary"),
        "conversation": i18n.t("app.text_table_header_conversation")
    } + _output_text(_app.open_history(date_index)) + history_text_table_footer


def _reload_history(f_history_selector):
    return gr.Dropdown.update(choices=_get_histories()), \
        _update_history(f_history_selector) if len(f_history_selector) != 0 else ""


def _get_person(p):
    return "%s (ID:%d)" % (p.name, p.person_id) if p is not None else None


def _get_persons():
    return [_get_person(p) for p in _app.get_persons()]


def _resolve_person(choice: str | None):
    if choice is None:
        return None
    r = re.search(r"\(ID:(\d+)\)$", choice)
    if r is None:
        return None
    person_id = int(r.group(1))
    for p in _app.get_persons():
        if p.person_id == person_id:
            return p
    return None


def _output_person_list():
    person_list = _app.get_persons()
    if len(person_list) == 0:
        return "<p>no data</p>"

    text = person_list_table_header % {
        "person_id": i18n.t("app.diarization_table_person_id"),
        "superseded_by": i18n.t("app.diarization_table_superseded_by"),
        "name": i18n.t("app.diarization_table_name"),
        "last_mapped_time": i18n.t("app.diarization_table_last_mapped_time")
    }

    person_list.sort(key=lambda p_: -p_.last_mapped_time)
    tm0 = time.localtime()
    for p in person_list:
        def _write_cell(content_, active_=True):
            nonlocal text
            active_ = (active_ and p.superseded_by == -1 and p.last_mapped_time >= 0.0)
            text += "<td>" if active_ else "<td><span class=\"suppressed\">"
            text += content_
            text += "</td>" if active_ else "</span></td>"

        _write_cell("%d" % p.person_id)
        _write_cell(
            "%d(%s)" % (p.superseded_by, html.escape(
                next(p_ for p_ in person_list if p_.person_id == p.superseded_by).name))
            if p.superseded_by != -1 else "")
        _write_cell(html.escape(p.name), not p.is_default)

        if p.last_mapped_time < 0.0:
            _write_cell(i18n.t("app.diarization_table_not_mapped"))
        else:
            tm = time.localtime(p.last_mapped_time)
            if tm0.tm_year == tm.tm_year and tm0.tm_yday == tm.tm_yday:
                _write_cell("%02d:%02d (%d minutes ago)" % (
                    tm.tm_hour, tm.tm_min, (tm0.tm_hour * 60 + tm0.tm_min) - (tm.tm_hour * 60 + tm.tm_min)))
            else:
                _write_cell("%s %02d:%02d" % (
                    datetime.datetime.fromtimestamp(p.last_mapped_time).strftime(i18n.t("app.date_format")),
                    tm.tm_hour, tm.tm_min))
        text += "</tr>\n"

    text += person_list_table_footer
    return text


def _pre_update_diarization():
    return gr.Button.update(interactive=False)


def _update_diarization(f_person_selector, f_update_diarization_with_plot):
    ret_f_person_selector = gr.Dropdown.update(
        choices=_get_persons(),
        value=_get_person(_resolve_person(f_person_selector)))
    ret_f_person_plot = gr.Plot.update(value=_app.plot_db(_conf.embedding_type), visible=True) \
        if f_update_diarization_with_plot and _conf.embedding_type is not None else gr.Plot.update(visible=False)
    return ret_f_person_selector, _output_person_list(), ret_f_person_plot, gr.Button.update(interactive=True)


def _select_person(f_person_selector):
    p = _resolve_person(f_person_selector)
    return p.name if p is not None else ""


def _rename_person_name(f_person_selector, f_person_new_name):
    if len(f_person_new_name) != 0:
        p = _resolve_person(f_person_selector)
        if p is not None:
            logging.info("rename ID:%d \"%s\" -> \"%s\"", p.person_id, p.name, f_person_new_name)
            _app.rename(p.person_id, f_person_new_name)
    return gr.Dropdown.update(
        choices=_get_persons(),
        value=_get_person(_resolve_person(f_person_selector))), _output_person_list()


def _erase_person(f_person_selector):
    p = _resolve_person(f_person_selector)
    if p is not None:
        logging.info("erase ID:%d \"%s\"", p.person_id, p.name)
        _app.erase(p.person_id)
    return gr.Dropdown.update(choices=_get_persons(), value=None), _output_person_list()


def _encode_embedding_type(embedding_type: str | None):
    return embedding_type if embedding_type == "speechbrain" or embedding_type == "pyannote" \
        else i18n.t('app.conf_embedding_type_none')


def _get_embedding_types():
    return [_encode_embedding_type(None), "speechbrain", "pyannote"]


def _resolve_embedding_type(name: str | None):
    return name if name == "speechbrain" or name == "pyannote" else None


def _find_plugins():
    return [plugin_name for dir_name, plugin_name in _app.find_installed_plugins()]


def _pre_apply_configuration():
    return gr.Button.update(interactive=False)


def _apply_configuration(
        f_conf_enable_plugins, f_conf_input_language,
        f_conf_enable_auto_detect_language, f_conf_enable_simultaneous_interpretation, f_conf_output_language,
        f_conf_ui_language, f_conf_ui_show_input_status, f_conf_ui_show_statement_properties,
        f_conf_input_devices, f_conf_device,
        f_conf_vad_threshold, f_conf_vad_pre_hold, f_conf_vad_post_hold, f_conf_vad_post_apply,
        f_conf_vad_soft_limit_length, f_conf_vad_hard_limit_length,
        f_conf_vad_wakeup_peak_threshold_db, f_conf_vad_wakeup_release,
        f_conf_embedding_type, f_conf_transcribe_min_duration, f_conf_transcribe_min_segment_duration,
        f_conf_keep_audio_file, f_conf_keep_audio_file_for,
        f_conf_max_hold_embeddings,
        f_conf_openai_api_key,
        f_conf_qualify_soft_limit, f_conf_qualify_hard_limit, f_conf_qualify_silent_interval,
        f_conf_qualify_merge_interval, f_conf_qualify_merge_threshold,
        f_conf_qualify_llm_model_name_step1, f_conf_qualify_llm_model_name_step2,
        *f_conf_args):

    global _ui_conf
    f_conf_args = list(f_conf_args)

    ui_conf: UiConfiguration = dataclasses.replace(_ui_conf)
    ui_conf.language = f_conf_ui_language
    ui_conf.show_input_status = f_conf_ui_show_input_status
    ui_conf.show_statement_properties = f_conf_ui_show_statement_properties
    ui_conf.openai_api_key = f_conf_openai_api_key

    conf = main.Configuration()
    conf.input_devices = f_conf_input_devices if len(f_conf_input_devices) != 0 else None
    conf.device = f_conf_device
    conf.language = f_conf_input_language
    conf.enable_auto_detect_language = f_conf_enable_auto_detect_language
    conf.enable_simultaneous_interpretation = f_conf_enable_simultaneous_interpretation

    conf.vad_threshold = f_conf_vad_threshold
    conf.vad_pre_hold = f_conf_vad_pre_hold
    conf.vad_post_hold = f_conf_vad_post_hold
    conf.vad_post_apply = f_conf_vad_post_apply
    conf.vad_soft_limit_length = f_conf_vad_soft_limit_length
    conf.vad_hard_limit_length = f_conf_vad_hard_limit_length
    conf.vad_wakeup_peak_threshold_db = f_conf_vad_wakeup_peak_threshold_db
    conf.vad_wakeup_release = f_conf_vad_wakeup_release

    conf.embedding_type = _resolve_embedding_type(f_conf_embedding_type)
    conf.transcribe_min_duration = f_conf_transcribe_min_duration
    conf.transcribe_min_segment_duration = f_conf_transcribe_min_segment_duration
    conf.keep_audio_file_for = f_conf_keep_audio_file_for * (3600 * 24) if f_conf_keep_audio_file else -1.0

    for i, emb_c in enumerate([conf.emb_sb, conf.emb_pn]):
        emb_c.threshold = f_conf_args.pop(0)
        emb_c.dbscan_eps = f_conf_args.pop(0)
        emb_c.dbscan_min_samples = f_conf_args.pop(0)
        emb_c.min_matched_embeddings_to_inherit_cluster = f_conf_args.pop(0)
        emb_c.min_matched_embeddings_to_match_person = f_conf_args.pop(0)

    conf.max_hold_embeddings = int(f_conf_max_hold_embeddings)

    conf.qualify_soft_limit = f_conf_qualify_soft_limit
    conf.qualify_hard_limit = f_conf_qualify_hard_limit
    conf.qualify_silent_interval = f_conf_qualify_silent_interval
    conf.qualify_merge_interval = f_conf_qualify_merge_interval
    conf.qualify_merge_threshold = f_conf_qualify_merge_threshold

    conf.llm_opt = llm.QualifyOptions(
        input_language=f_conf_input_language, output_language=f_conf_output_language,
        model_for_step1=f_conf_qualify_llm_model_name_step1,
        model_for_step2=f_conf_qualify_llm_model_name_step2
    )

    conf.disabled_plugins = [plugin for plugin in _find_plugins() if plugin not in f_conf_enable_plugins]

    _ui_conf = ui_conf
    _restart(conf)
    with tools.SafeWrite(os.path.join(main.data_dir_name, "config.pickle"), "wb") as f:
        pickle.dump({"conf": conf}, f.stream)
    with tools.SafeWrite(os.path.join(main.data_dir_name, "ui_config.pickle"), "wb") as f:
        pickle.dump({"conf": ui_conf}, f.stream)

    return gr.Button.update(interactive=True)


def _load_configuration():
    conf = main.Configuration()
    conf_file_path = os.path.join(main.data_dir_name, "config.pickle")
    if os.path.isfile(conf_file_path):
        with open(conf_file_path, "rb") as f:
            d = pickle.load(f)
            conf = d["conf"]
    return conf


def _load_ui_configuration():
    conf = UiConfiguration()
    conf_file_path = os.path.join(main.data_dir_name, "ui_config.pickle")
    if os.path.isfile(conf_file_path):
        with open(conf_file_path, "rb") as f:
            d = pickle.load(f)
            conf = d["conf"]
    return conf


def app_main(args=None):
    global _app, _conf, _live_checker_thread
    _ = args

    os.makedirs(main.data_dir_name, exist_ok=True)
    tools.recover_files(main.data_dir_name)

    _app = main.Application(_load_configuration(), _ui_conf.language)
    _conf = _app.get_current_configuration()

    _live_checker_thread = th.Thread(target=_live_checker)
    _live_checker_thread.start()

    with gr.Blocks(title=i18n.t("app.application_name"), css=block_css) as demo:
        f_api_playback_audio = gr.Button(visible=False)
        f_api_playback_audio_files = gr.Textbox(visible=False)
        f_api_playback_audio.click(_playback_audio, [f_api_playback_audio_files], None, api_name="playback_audio")

        with gr.Tab(i18n.t("app.tab_current")):
            f_text = gr.HTML()

        with gr.Tab(i18n.t("app.tab_history")):
            with gr.Row():
                with gr.Column(scale=5):
                    f_history_selector = gr.Dropdown(
                        label=i18n.t("app.history_date"), allow_custom_value=False, choices=_get_histories())
                with gr.Column(scale=1):
                    f_history_reload = gr.Button(value=i18n.t("app.history_reload"))
            f_history_text = gr.HTML()

        with gr.Tab(i18n.t("app.tab_diarization")):
            f_update_diarization = gr.Button(value=i18n.t("app.diarization_update"))
            f_update_diarization_with_plot = gr.Checkbox(
                label=i18n.t('app.diarization_update_with_plot'), value=False)
            f_person_plot = gr.Plot(visible=False)
            with gr.Row():
                with gr.Column(scale=5):
                    with gr.Group():
                        f_person_selector = gr.Dropdown(
                            label=i18n.t("app.diarization_person_selector"), allow_custom_value=False,
                            choices=_get_persons())
                        f_person_new_name = gr.Textbox(label=i18n.t("app.diarization_new_name"))
                with gr.Column(scale=1):
                    f_person_rename = gr.Button(value=i18n.t("app.diarization_person_rename"), variant="primary")
                    f_person_erase = gr.Button(value=i18n.t("app.diarization_person_erase"))
            f_person_list = gr.HTML(value=_output_person_list())

        plugins = _app.ref_plugins()
        plugins_have_tab = {name: p for name, p in plugins.items() if p.injection_point() & pl.FLAG_ADD_TAB}
        for name in sorted(plugins_have_tab.keys()):
            p = plugins_have_tab[name]
            with gr.Tab(p.tab_name()):
                p.build_tab()

        with gr.Tab(i18n.t('app.tab_configuration')):
            f_conf_args = []
            gr.Markdown(i18n.t('app.conf_apply_desc'))
            f_conf_apply = gr.Button(value=i18n.t('app.conf_apply'), variant="primary")
            with gr.Group():
                f_conf_enable_plugins = gr.CheckboxGroup(
                    visible=(len(_find_plugins()) != 0),
                    label=i18n.t('app.conf_enable_plugins'),
                    choices=_find_plugins(),
                    value=[plugin for plugin in _find_plugins() if plugin not in _conf.disabled_plugins])
            with gr.Group():
                f_conf_input_language = gr.Dropdown(
                    label=i18n.t('app.conf_input_language'),
                    multiselect=False, allow_custom_value=False,
                    choices=["en", "ja"], value=_conf.language)
                f_conf_enable_auto_detect_language = gr.Checkbox(
                    label=i18n.t('app.conf_enable_auto_detect_language'),
                    value=_conf.enable_auto_detect_language)
                f_conf_enable_simultaneous_interpretation = gr.Checkbox(
                    label=i18n.t('app.conf_enable_simultaneous_interpretation'),
                    value=_conf.enable_simultaneous_interpretation)
                f_conf_output_language = gr.Dropdown(
                    label=i18n.t('app.conf_output_language'),
                    multiselect=False, allow_custom_value=False,
                    choices=["en", "ja"], value=_conf.llm_opt.output_language)
            with gr.Group():
                f_conf_ui_language = gr.Dropdown(
                    label=i18n.t('app.conf_ui_language'),
                    multiselect=False, allow_custom_value=False,
                    choices=["auto", "en", "ja"], value=_ui_conf.language)
            with gr.Group():
                devices_by_name = {d["name"]: d for d in main.AudioInput.query_valid_input_devices()}
                f_conf_input_devices = gr.CheckboxGroup(
                    label=i18n.t('app.conf_input_devices'),
                    choices=list(devices_by_name.keys()),
                    value=None if _conf.input_devices is None else
                    [d for d in _conf.input_devices if d in devices_by_name.keys()])
                f_conf_ui_show_input_status = gr.Checkbox(
                    label=i18n.t('app.conf_ui_show_input_status'),
                    value=_ui_conf.show_input_status)
                f_conf_ui_show_statement_properties = gr.Checkbox(
                    label=i18n.t('app.conf_ui_show_statement_properties'),
                    value=_ui_conf.show_statement_properties)
                f_conf_device = gr.Dropdown(
                    label=i18n.t('app.conf_device'),
                    multiselect=False, allow_custom_value=True,
                    choices=["cpu", "gpu"], value=_conf.device)
            with gr.Group():
                f_conf_keep_audio_file = gr.Checkbox(
                    label=i18n.t('app.conf_keep_audio_file'),
                    value=(_conf.keep_audio_file_for >= 0.0))
                f_conf_keep_audio_file_for = gr.Slider(
                    label=i18n.t('app.conf_keep_audio_file_for'),
                    minimum=1.0, maximum=14.0, step=1.0,
                    value=float(int(_conf.keep_audio_file_for / (3600 * 24)))
                    if _conf.keep_audio_file_for >= 0.0 else 1.0)
                with gr.Accordion(i18n.t('app.conf_vad_group'), open=False):
                    f_conf_vad_threshold = gr.Slider(
                        label=i18n.t('app.conf_vad_threshold'),
                        minimum=0.1, maximum=0.9, value=_conf.vad_threshold, step=0.01)
                    f_conf_vad_pre_hold = gr.Slider(
                        label=i18n.t('app.conf_vad_pre_hold'),
                        minimum=0.0, maximum=1.0, value=_conf.vad_pre_hold, step=0.01)
                    f_conf_vad_post_hold = gr.Slider(
                        label=i18n.t('app.conf_vad_post_hold'),
                        minimum=0.0, maximum=2.0, value=_conf.vad_post_hold, step=0.01)
                    f_conf_vad_post_apply = gr.Slider(
                        label=i18n.t('app.conf_vad_post_apply'),
                        minimum=0.0, maximum=2.0, value=_conf.vad_post_apply, step=0.01)
                    f_conf_vad_soft_limit_length = gr.Slider(
                        label=i18n.t('app.conf_vad_soft_limit_length'),
                        minimum=10.0, maximum=120.0, value=_conf.vad_soft_limit_length, step=1.0)
                    f_conf_vad_hard_limit_length = gr.Slider(
                        label=i18n.t('app.conf_vad_hard_limit_length'),
                        minimum=10.0, maximum=120.0, value=_conf.vad_soft_limit_length, step=1.0)
                    f_conf_vad_wakeup_peak_threshold_db = gr.Slider(
                        label=i18n.t('app.conf_vad_wakeup_peak_threshold_db'),
                        minimum=-80.0, maximum=0.0, value=_conf.vad_wakeup_peak_threshold_db, step=1.0)
                    f_conf_vad_wakeup_release = gr.Slider(
                        label=i18n.t('app.conf_vad_wakeup_release'),
                        minimum=0.0, maximum=20.0, value=_conf.vad_wakeup_release, step=0.1)
            with gr.Group():
                f_conf_embedding_type = gr.Dropdown(
                    label=i18n.t('app.conf_embedding_type'),
                    multiselect=False, allow_custom_value=False,
                    choices=_get_embedding_types(), value=_encode_embedding_type(_conf.embedding_type))
                with gr.Accordion(i18n.t('app.conf_emb_group'), open=False):
                    f_conf_transcribe_min_duration = gr.Slider(
                        label=i18n.t('app.conf_transcribe_min_duration'),
                        minimum=0.1, maximum=5.0, value=_conf.transcribe_min_duration, step=0.1)
                    f_conf_transcribe_min_segment_duration = gr.Slider(
                        label=i18n.t('app.conf_transcribe_min_segment_duration'),
                        minimum=0.1, maximum=5.0, value=_conf.transcribe_min_segment_duration, step=0.1)
                    for label, emb_c in [('app.conf_emb_sb_group', _conf.emb_sb),
                                         ('app.conf_emb_pn_group', _conf.emb_pn)]:
                        with gr.Accordion(i18n.t(label), open=False):
                            f_conf_args += [
                                gr.Slider(
                                    label=i18n.t('app.conf_emb_threshold'),
                                    minimum=0.1, maximum=0.9, value=emb_c.threshold, step=0.01),
                                gr.Slider(
                                    label=i18n.t('app.conf_emb_dbscan_eps'),
                                    minimum=0.1, maximum=0.9, value=emb_c.dbscan_eps, step=0.01),
                                gr.Slider(
                                    label=i18n.t('app.conf_emb_dbscan_min_samples'),
                                    minimum=2.0, maximum=10.0, value=emb_c.dbscan_min_samples, step=1.0),
                                gr.Slider(
                                    label=i18n.t('app.conf_emb_min_matched_embeddings_to_inherit_cluster'),
                                    minimum=2.0, maximum=10.0,
                                    value=emb_c.min_matched_embeddings_to_inherit_cluster, step=1.0),
                                gr.Slider(
                                    label=i18n.t('app.conf_emb_min_matched_embeddings_to_match_person'),
                                    minimum=2.0, maximum=10.0,
                                    value=emb_c.min_matched_embeddings_to_match_person, step=1.0),
                            ]
                    f_conf_max_hold_embeddings = gr.Slider(
                        label=i18n.t('app.conf_max_hold_embeddings'),
                        minimum=5.0, maximum=100.0, value=float(_conf.max_hold_embeddings), step=1.0)
            with gr.Group():
                f_conf_openai_api_key = gr.Textbox(
                    label=i18n.t("app.conf_openai_api_key"), type="password", value=_ui_conf.openai_api_key)
                with gr.Accordion(i18n.t('app.conf_qualify_group'), open=False):
                    f_conf_qualify_soft_limit = gr.Slider(
                        label=i18n.t('app.conf_qualify_soft_limit'),
                        minimum=60.0, maximum=600.0, value=_conf.qualify_soft_limit, step=10.0)
                    f_conf_qualify_hard_limit = gr.Slider(
                        label=i18n.t('app.conf_qualify_hard_limit'),
                        minimum=60.0, maximum=600.0, value=_conf.qualify_hard_limit, step=10.0)
                    f_conf_qualify_silent_interval = gr.Slider(
                        label=i18n.t('app.conf_qualify_silent_interval'),
                        minimum=1.0, maximum=60.0, value=_conf.qualify_silent_interval, step=1.0)
                    f_conf_qualify_merge_interval = gr.Slider(
                        label=i18n.t('app.conf_qualify_merge_interval'),
                        minimum=1.0, maximum=60.0, value=_conf.qualify_merge_interval, step=1.0)
                    f_conf_qualify_merge_threshold = gr.Slider(
                        label=i18n.t('app.conf_qualify_merge_threshold'),
                        minimum=0.1, maximum=0.9, value=_conf.qualify_merge_threshold, step=0.01)

                    llm_model_choices = ["gpt-4-0613", "gpt-4-0314", "gpt-3.5-turbo-0613"]
                    f_conf_qualify_llm_model_name_step1 = gr.Dropdown(
                        label=i18n.t('app.conf_qualify_llm_model_name_step1'),
                        multiselect=False, allow_custom_value=False,
                        choices=llm_model_choices,
                        value=_conf.llm_opt.model_for_step1)
                    f_conf_qualify_llm_model_name_step2 = gr.Dropdown(
                        label=i18n.t('app.conf_qualify_llm_model_name_step2'),
                        multiselect=False, allow_custom_value=False,
                        choices=llm_model_choices,
                        value=_conf.llm_opt.model_for_step2)

        f_text.change(None, None, None, _js=move_to_last_js)
        f_history_selector.select(_update_history, [f_history_selector], [f_history_text])
        f_history_reload.click(_reload_history, [f_history_selector], [f_history_selector, f_history_text])
        f_update_diarization.click(_pre_update_diarization, None, [f_update_diarization]).then(
            _update_diarization,
            [f_person_selector, f_update_diarization_with_plot],
            [f_person_selector, f_person_list, f_person_plot, f_update_diarization])
        f_person_selector.select(_select_person, [f_person_selector], [f_person_new_name])
        f_person_rename.click(
            _rename_person_name, [f_person_selector, f_person_new_name], [f_person_selector, f_person_list])
        f_person_erase.click(
            _erase_person, [f_person_selector], [f_person_selector, f_person_list])
        f_conf_apply.click(_pre_apply_configuration, None, [f_conf_apply]).then(_apply_configuration, [
            f_conf_enable_plugins, f_conf_input_language,
            f_conf_enable_auto_detect_language, f_conf_enable_simultaneous_interpretation, f_conf_output_language,
            f_conf_ui_language, f_conf_ui_show_input_status, f_conf_ui_show_statement_properties,
            f_conf_input_devices, f_conf_device,
            f_conf_vad_threshold, f_conf_vad_pre_hold, f_conf_vad_post_hold, f_conf_vad_post_apply,
            f_conf_vad_soft_limit_length, f_conf_vad_hard_limit_length,
            f_conf_vad_wakeup_peak_threshold_db, f_conf_vad_wakeup_release,
            f_conf_embedding_type, f_conf_transcribe_min_duration, f_conf_transcribe_min_segment_duration,
            f_conf_keep_audio_file, f_conf_keep_audio_file_for,
            f_conf_max_hold_embeddings,
            f_conf_openai_api_key,
            f_conf_qualify_soft_limit, f_conf_qualify_hard_limit, f_conf_qualify_silent_interval,
            f_conf_qualify_merge_interval, f_conf_qualify_merge_threshold,
            f_conf_qualify_llm_model_name_step1, f_conf_qualify_llm_model_name_step2,
            *f_conf_args], [f_conf_apply])

        demo.load(_interval_update, None, [f_text], every=1)

    demo.queue().launch(server_name="0.0.0.0")  # TODO network opts


if __name__ == '__main__':
    _ui_conf = _load_ui_configuration()

    logging.basicConfig(
        format='%(asctime)s: %(name)s:%(funcName)s:%(lineno)d %(levelname)s: %(message)s', level=logging.INFO)
    logging.getLogger().handlers[0].addFilter(lambda record: record.name != "httpx")

    language_code = _ui_conf.language
    if language_code == "auto":
        lc = locale.getlocale()
        language_code = lc[0][0:2] if lc is not None and lc[0] is not None and len(lc[0]) >= 2 else 'en'
    logging.info("language_code = %s" % language_code)

    i18n.load_path.append('./i18n')
    i18n.set('locale', language_code)
    i18n.set('fallback', 'en')

    if len(_ui_conf.openai_api_key) != 0:
        openai.api_key = _ui_conf.openai_api_key

    app_main()
