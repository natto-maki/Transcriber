import dataclasses
import enum
import numpy as np

unknown_person_name = "unknown"
unknown_person_display_name = "???"


class SentenceType(enum.Enum):
    Sentence = 0
    LanguageDetected = 1  # old_language, new_language


@dataclasses.dataclass
class AdditionalProperties:
    source: str = ""
    vad_ave_level: float = 0.0
    vad_max_level: float = 0.0
    audio_level: float = 0.0
    segment_audio_level: float = 0.0
    audio_file_name_list: list[str] | None = None
    language: str = ""

    def clone(self):
        r = dataclasses.replace(self)
        if self.audio_file_name_list is not None:
            r.audio_file_name_list = list(self.audio_file_name_list)
        return r

    def append_audio_file(self, audio_file_name: str | None):
        if audio_file_name is None:
            return
        if self.audio_file_name_list is None:
            self.audio_file_name_list = []
        self.audio_file_name_list.append(audio_file_name)


@dataclasses.dataclass
class Sentence:
    tm0: float
    tm1: float
    text: str

    sentence_type: SentenceType = SentenceType.Sentence
    payload: dict | None = None

    # note: If sentence_type is other than Sentence, these fields have always initial values
    embedding: np.ndarray | None = None
    person_id: int = -1
    person_name: str = unknown_person_name

    prop: AdditionalProperties | None = None

    def clone(self):
        r = dataclasses.replace(self)
        if self.prop is not None:
            r.prop = self.prop.clone()
        return r

    def merge(self, s):
        self.text += " " + s.text
        self.tm1 = s.tm1
        if s.prop is not None and s.prop.audio_file_name_list is not None:
            if self.prop is None:
                self.prop = AdditionalProperties()
            for name in s.prop.audio_file_name_list:
                self.prop.append_audio_file(name)


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
