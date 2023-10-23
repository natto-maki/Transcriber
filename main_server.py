import time
import threading as th
import concurrent.futures
from collections import deque
import dataclasses

# noinspection PyPackageRequirements
from fastapi import FastAPI
# noinspection PyPackageRequirements
from pydantic import BaseModel
import numpy as np

import common
import main

sampling_rate = common.sampling_rate
frame_size = common.frame_size


class RemoteAudioInput(main.ConcurrentContextManagerImpl):
    def __init__(self, buffer_samples=2048, **kwargs):
        super().__init__(**kwargs)
        self.__buffer_samples = buffer_samples

        self.__lock0 = th.Lock()
        self.__is_opened = False
        self.__buffer = deque()
        self.__size = 0
        self.__offset = 0

    def push(self, audio_frame: np.ndarray):
        with self.__lock0:
            if self.__is_opened and self.__size < self.__buffer_samples * 2:
                self.__buffer.append(audio_frame)
                self.__size += len(audio_frame)

    def _handler(self):
        tm0 = time.time()
        output_frame_count = 0
        while not self._should_stop():
            output = np.zeros((frame_size,), dtype=np.float32)
            with self.__lock0:
                if self.__size < frame_size:
                    self.__buffer.append(np.full(
                        (frame_size - self.__size,), 0.0 if self.__size == 0 else self.__buffer[-1][-1],
                        dtype=np.float32))
                    self.__size = frame_size

                offset = 0
                while offset < frame_size:
                    add = min(frame_size - offset, len(self.__buffer[0]) - self.__offset)
                    output[offset:offset + add] = self.__buffer[0][self.__offset:self.__offset + add]
                    offset += add

                    self.__size -= add
                    self.__offset += add
                    if self.__offset >= len(self.__buffer[0]):
                        self.__buffer.popleft()
                        self.__offset = 0

            self._invoke_callback(tm0 + output_frame_count * frame_size / sampling_rate, output)

            output_frame_count += 1
            tm1 = tm0 + output_frame_count * frame_size / sampling_rate
            tm_current = time.time()
            if tm_current < tm1:
                time.sleep(tm1 - tm_current)

    def open(self):
        with self.__lock0:
            self.__is_opened = True
            self.__buffer = deque([np.zeros((self.__buffer_samples,), dtype=np.float32)])
            self.__size = self.__buffer_samples
            self.__offset = 0
        super().open()

    def close(self):
        super().close()
        with self.__lock0:
            self.__is_opened = False


@dataclasses.dataclass
class Session:
    audio_input: RemoteAudioInput | None = None
    app: main.Application | None = None
    app_thread: concurrent.futures.ThreadPoolExecutor | None = None
    last_access: float = 0.0


_app = FastAPI()

_lock0 = th.Lock()
_sessions: dict[str, Session] = {}


def _start_app(s: Session):
    s.app = main.Application(audio_input=s.audio_input)
    s.app.open()


def _stop_app(s: Session):
    if s.app.is_opened():
        s.app.close()


def _push(s: Session, audio_frame: np.ndarray):
    s.audio_input.push(audio_frame)


def _check_session(session_id: str) -> Session:
    with _lock0:
        if session_id in _sessions:
            return _sessions[session_id]
        s = Session()
        s.audio_input = RemoteAudioInput()
        s.app_thread = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        _sessions[session_id] = s

    s.app_thread.submit(_start_app, s)
    return s


@_app.get("/open/{session_id}")
def open_session(session_id: str):
    _check_session(session_id)


@_app.get("/close/{session_id}")
def close_session(session_id: str):
    with _lock0:
        if session_id in _sessions:
            s = _sessions[session_id]
            s.app_thread.submit(_stop_app, s)
            del _sessions[session_id]


class AudioFrame(BaseModel):
    samples: list[int]


@_app.get("/push/{session_id}")
def push_audio_frame(session_id: str, audio_frame: AudioFrame):
    s = _check_session(session_id)
    s.app_thread.submit(_push, s, np.array([v / 32768 for v in audio_frame.samples], dtype=np.float32))
