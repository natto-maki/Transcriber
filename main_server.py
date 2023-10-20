import time
import threading as th
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
    last_access: float = 0.0


_app = FastAPI()

_lock0 = th.Lock()
_sessions: dict[str, Session] = {}


@_app.get("/open/{session_id}")
def open_session(session_id: str):
    with _lock0:
        if session_id not in _sessions:
            audio_input = RemoteAudioInput()
            # TODO yaml
            _sessions[session_id] = Session(
                audio_input=audio_input,
                app=main.Application(audio_input=audio_input))


@_app.get("/close/{session_id}")
def close_session(session_id: str):
    with _lock0:
        if session_id in _sessions:
            del _sessions[session_id]


class AudioFrame(BaseModel):
    samples: list[int]


@_app.get("/push/{session_id}")
def push_audio_frame(session_id: str, audio_frame: AudioFrame):
    pass
