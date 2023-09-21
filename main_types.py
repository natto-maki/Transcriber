import dataclasses
import numpy as np

unknown_person_name = "unknown"
unknown_person_display_name = "???"


@dataclasses.dataclass
class AdditionalProperties:
    source: str = ""
    vad_ave_level: float = 0.0
    vad_max_level: float = 0.0
    audio_level: float = 0.0
    segment_audio_level: float = 0.0
    audio_file_name: str | None = None
    audio_file_name_list: list[str] | None = None

    def clone(self):
        r = dataclasses.replace(self)
        if self.audio_file_name_list is not None:
            r.audio_file_name_list = list(self.audio_file_name_list)
        return r

    def append_audio_file(self, audio_file_name: str | None):
        if audio_file_name is None:
            return
        if self.audio_file_name is None:
            self.audio_file_name = audio_file_name
        if self.audio_file_name_list is None:
            self.audio_file_name_list = []
        self.audio_file_name_list.append(audio_file_name)


@dataclasses.dataclass
class Sentence:
    tm0: float
    tm1: float
    text: str
    embedding: np.ndarray | None = None
    person_id: int = -1
    person_name: str = unknown_person_name
    prop: AdditionalProperties | None = None

    def clone(self):
        r = dataclasses.replace(self)
        if self.prop is not None:
            r.prop = self.prop.clone()
        return r


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
