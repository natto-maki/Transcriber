from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class TranscribeRequest(_message.Message):
    __slots__ = ["audio_data", "language", "get_embedding", "min_segment_duration"]
    AUDIO_DATA_FIELD_NUMBER: _ClassVar[int]
    LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    GET_EMBEDDING_FIELD_NUMBER: _ClassVar[int]
    MIN_SEGMENT_DURATION_FIELD_NUMBER: _ClassVar[int]
    audio_data: bytes
    language: str
    get_embedding: str
    min_segment_duration: float
    def __init__(self, audio_data: _Optional[bytes] = ..., language: _Optional[str] = ..., get_embedding: _Optional[str] = ..., min_segment_duration: _Optional[float] = ...) -> None: ...

class TranscribeResponse(_message.Message):
    __slots__ = ["segments", "embeddings"]
    SEGMENTS_FIELD_NUMBER: _ClassVar[int]
    EMBEDDINGS_FIELD_NUMBER: _ClassVar[int]
    segments: str
    embeddings: _containers.RepeatedScalarFieldContainer[bytes]
    def __init__(self, segments: _Optional[str] = ..., embeddings: _Optional[_Iterable[bytes]] = ...) -> None: ...

class DetectLanguageRequest(_message.Message):
    __slots__ = ["audio_data"]
    AUDIO_DATA_FIELD_NUMBER: _ClassVar[int]
    audio_data: bytes
    def __init__(self, audio_data: _Optional[bytes] = ...) -> None: ...

class DetectLanguageResponse(_message.Message):
    __slots__ = ["detected_languages"]
    DETECTED_LANGUAGES_FIELD_NUMBER: _ClassVar[int]
    detected_languages: str
    def __init__(self, detected_languages: _Optional[str] = ...) -> None: ...
