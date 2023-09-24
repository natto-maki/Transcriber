import dataclasses
import enum
import copy

import numpy as np

unknown_person_name = "unknown"
unknown_person_display_name = "???"


def _merge_list(target: list | None, value: list | None):
    if target is None and value is None:
        return None
    return (target if target is not None else []) + (value if value is not None else [])


def _merge_str(target: str, value: str):
    return target + (" " if target != "" else "") + value


class SentenceType(enum.Enum):
    Sentence = 0
    LanguageDetected = 1  # old_language, new_language
    SentenceSeparator = 2


@dataclasses.dataclass
class AudioFileProperties:
    offset: int = 0
    length: int = 0

    def clone(self):
        return copy.deepcopy(self)


@dataclasses.dataclass
class AdditionalProperties:
    source: str = ""
    vad_ave_level: float = 0.0
    vad_max_level: float = 0.0
    audio_level: float = 0.0
    segment_audio_level: float = 0.0
    audio_file_name_list: list[str] | None = None
    audio_file_prop_list: list[AudioFileProperties | None] | None = None
    language: str = ""

    def clone(self):
        return copy.deepcopy(self)

    def append_audio_file(self, audio_file_name: str | None, audio_file_prop: AudioFileProperties | None = None):
        if audio_file_name is not None:
            self.audio_file_name_list = _merge_list(self.audio_file_name_list, [audio_file_name])
            self.audio_file_prop_list = _merge_list(self.audio_file_prop_list, [audio_file_prop])
        return self


@dataclasses.dataclass
class SimultaneousInterpretationState:
    processed_org: list[str] | None = None
    processed_int: str = ""
    processing: str = ""
    waiting: list[str] | None = None

    def clone(self):
        return copy.deepcopy(self)

    def merge(self, r):
        # note: If there is sentences awaiting translation in `self` and
        # sentences already translated in `r`, the order in the display will be swapped.
        self.processed_org = _merge_list(self.processed_org, r.processed_org)
        self.processed_int = _merge_str(self.processed_int, r.processed_int)
        self.processing = _merge_str(self.processing, r.processing)
        self.waiting = _merge_list(self.waiting, r.waiting)
        return self


@dataclasses.dataclass
class Sentence:
    tm0: float
    tm1: float
    text: str
    si_state: SimultaneousInterpretationState | None = None

    sentence_type: SentenceType = SentenceType.Sentence
    payload: dict | None = None

    # note: If sentence_type is other than Sentence, these fields have always initial values
    embedding: np.ndarray | None = None
    person_id: int = -1
    person_name: str = unknown_person_name

    prop: AdditionalProperties | None = None

    def clone(self):
        return copy.deepcopy(self)

    def merge(self, s):
        self.text = _merge_str(self.text, s.text)
        self.tm1 = s.tm1

        if self.si_state is not None and s.si_state is not None:
            self.si_state.merge(s.si_state)

        if s.prop is not None and s.prop.audio_file_name_list is not None:
            if self.prop is None:
                self.prop = AdditionalProperties()
            offset_delta = len(self.text) - len(s.text)
            for i, name in enumerate(s.prop.audio_file_name_list):
                prop = (s.prop.audio_file_prop_list[i]
                        if s.prop.audio_file_prop_list is not None and i < len(s.prop.audio_file_prop_list)
                        else None)
                if prop is not None:
                    prop = prop.clone()
                    prop.offset += offset_delta
                self.prop.append_audio_file(name, prop)
        return self

    def add_text(self, text: str, audio_file_name: str | None = None):
        self.text = _merge_str(self.text, text)
        if self.prop is None:
            return
        if audio_file_name is not None:
            total_length = len(self.text)
            text_length = len(text)
            self.prop.append_audio_file(audio_file_name, AudioFileProperties(
                offset=total_length - text_length, length=text_length))
        return self


@dataclasses.dataclass
class QualifiedResult:
    corrected_sentences: list[Sentence] | None = None
    summaries: str | None = None
    action_items: list[str] | None = None


SENTENCE_QUALIFIED = 1
SENTENCE_QUALIFYING = 2
SENTENCE_BUFFER = 3
SENTENCE_QUALIFY_ERROR = 4


@dataclasses.dataclass
class SentenceGroup:
    state: int
    sentences: list[Sentence]
    qualified: QualifiedResult | None = None
