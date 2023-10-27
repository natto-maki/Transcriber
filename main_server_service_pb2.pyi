from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class BaseResponse(_message.Message):
    __slots__ = ["result"]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    result: str
    def __init__(self, result: _Optional[str] = ...) -> None: ...

class OpenRequest(_message.Message):
    __slots__ = ["session_id"]
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    def __init__(self, session_id: _Optional[str] = ...) -> None: ...

class CloseRequest(_message.Message):
    __slots__ = ["session_id"]
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    def __init__(self, session_id: _Optional[str] = ...) -> None: ...

class PushRequest(_message.Message):
    __slots__ = ["session_id", "audio_data"]
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    AUDIO_DATA_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    audio_data: bytes
    def __init__(self, session_id: _Optional[str] = ..., audio_data: _Optional[bytes] = ...) -> None: ...

class ReadRequest(_message.Message):
    __slots__ = ["session_id", "source", "read_parameters", "begin_time", "end_time"]
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    READ_PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    BEGIN_TIME_FIELD_NUMBER: _ClassVar[int]
    END_TIME_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    source: str
    read_parameters: str
    begin_time: float
    end_time: float
    def __init__(self, session_id: _Optional[str] = ..., source: _Optional[str] = ..., read_parameters: _Optional[str] = ..., begin_time: _Optional[float] = ..., end_time: _Optional[float] = ...) -> None: ...

class ReadResponse(_message.Message):
    __slots__ = ["result", "payload"]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    result: str
    payload: str
    def __init__(self, result: _Optional[str] = ..., payload: _Optional[str] = ...) -> None: ...
