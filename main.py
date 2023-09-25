"""
Transcriber
"""
import re
import sys
import os
import glob
import time
import threading as th
import logging
import pickle
import json
import concurrent.futures
from collections import deque
import dataclasses
import importlib

# noinspection PyPackageRequirements
import grpc
import transcriber_service_pb2
import transcriber_service_pb2_grpc

import numpy as np
import sounddevice as sd
import soundfile as sf
import onnxruntime
import torch
from faster_whisper import WhisperModel
from speechbrain.pretrained import EncoderClassifier
# noinspection PyPackageRequirements
from pyannote.audio import Model
# noinspection PyPackageRequirements
from pyannote.audio import Inference

import tools
import main_types as t
import emb_db as db
import llm_openai as llm
import transcriber_plugin as pl
import transcriber_hack


# workaround for an error on x86 Mac
# "OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized."
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


sampling_rate = 16000
frame_size = 512
data_dir_name = "data"


class ContextManagerImpl:
    def __init__(self, **kwargs):
        self._stream = None
        self.__callbacks = []
        self.__opened = False

    def add_callback(self, callback, **kwargs):
        self.__callbacks.append([callback, kwargs])
        return self

    def _has_callback(self):
        return len(self.__callbacks) != 0

    def _invoke_callback(self, *args, **kwargs):
        if not self.__opened:
            return
        for e in self.__callbacks:
            e[0](*args, **kwargs, **e[1])

    def open(self):
        self.__opened = True
        if self._stream is not None:
            self._stream.open()

    def close(self):
        self.__opened = False
        if self._stream is not None:
            self._stream.close()

    def is_opened(self):
        return self.__opened

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class ConcurrentContextManagerImpl(ContextManagerImpl):
    def __init__(self, finished_callback=None, **kwargs):
        super().__init__(**kwargs)
        self.__finished_callback = finished_callback
        self.__stop = th.Event()
        self.__thread = None

    def _should_stop(self):
        return self.__stop.is_set()

    def _handler(self):
        raise NotImplementedError

    def __invoke_handler(self):
        if not self._handler():
            return
        if self.__finished_callback is not None:
            self.__finished_callback()

    def open(self):
        self.__stop.clear()
        self.__thread = th.Thread(target=self.__invoke_handler)
        self.__thread.start()
        super().open()

    def close(self):
        super().close()
        self.__stop.set()
        self.__thread.join()
        self.__thread = None


class MultithreadContextManagerImpl(ContextManagerImpl):
    def __init__(self, acceptable_delay=120.0, **kwargs):
        super().__init__(**kwargs)
        self.__acceptable_delay = acceptable_delay

        self.__thread = None
        self.__mutex = th.Lock()
        self.__semaphore = th.Semaphore(0)
        self.__requests = deque()

    def _init_process(self):
        pass

    def _request_handler(self, *args, **kwargs):
        raise NotImplementedError

    def __push_request(self, *args):
        with self.__mutex:
            self.__requests.append([*args])
        self.__semaphore.release()

    def _send_request(self, *args, **kwargs):
        self.__push_request(0, time.time(), args, kwargs)

    def __poll_request(self):
        with self.__mutex:
            ret = self.__requests.popleft()
        return ret

    def __open(self):
        self._init_process()
        while True:
            self.__semaphore.acquire()
            command, tm0, args, kwargs = self.__poll_request()

            if command == -1:
                break
            tm1 = time.time()

            self._request_handler(*args, **kwargs)

            if tm1 - tm0 > self.__acceptable_delay:
                skipped = 0
                while self.__semaphore.acquire(blocking=False):
                    command, _, _, _ = self.__poll_request()
                    if command == -1:
                        skipped = -1
                        break
                    skipped += 1
                if skipped == -1:
                    break
                logging.warning("Due to processing delay, %d requests has been skipped" % skipped)

    def open(self):
        self.__thread = th.Thread(target=self.__open)
        self.__thread.start()
        super().open()

    def close(self):
        super().close()
        self.__push_request(-1, None, None, None)
        self.__thread.join()
        self.__thread = None
        self.__requests.clear()


class CaptureCallback(ContextManagerImpl):
    def __init__(self, stream, file_name, **kwargs):
        super().__init__(**kwargs)
        self._stream = stream.add_callback(self._capture_callback)
        self.__file_name = file_name

        self.__start_time = 0

        self.__lock0 = th.Lock()
        self.__history = []

    def _capture_callback(self, *args, **kwargs):
        with self.__lock0:
            self.__history.append([time.time(), args, kwargs])
            self._invoke_callback(*args, **kwargs)

    def open(self):
        self.__start_time = time.time()
        super().open()

    def close(self):
        super().close()
        with open(self.__file_name, "wb") as f:
            pickle.dump({"start_time": self.__start_time, "end_time": time.time(), "history": self.__history}, f)


class RecallCallback(ConcurrentContextManagerImpl):
    def __init__(self, file_name, real_time=True, **kwargs):
        super().__init__(**kwargs)
        self.__real_time = real_time

        with open(file_name, "rb") as f:
            d = pickle.load(f)
            self.__start_time = d["start_time"]
            self.__end_time = d["end_time"]
            self.__history = d["history"]

        self.__open_time = 0
        self.__index = 0
        self.__stop = th.Event()
        self.__thread = None

    def _handler(self):
        tm0 = time.time()
        for h in self.__history:
            if self._should_stop():
                return False
            timestamp, args, kwargs = h
            if self.__real_time:
                delta = (timestamp - self.__start_time) - (time.time() - tm0)
                if delta > 0:
                    time.sleep(delta)
            self._invoke_callback(*args, **kwargs)

        return True

    def open(self):
        self.__open_time = time.time()
        self.__index = 0
        super().open()


class AudioInput(ContextManagerImpl):
    def __init__(self, selected_device, **kwargs):
        super().__init__(**kwargs)
        self.__selected_device = selected_device
        self.__audio_stream = None
        self.__time_delta = None

    @staticmethod
    def query_valid_input_devices():
        valid_devices = []
        host_apis = sd.query_hostapis()
        for device in sd.query_devices():
            if device["max_input_channels"] > 0:
                device["host_api_name"] = host_apis[device["hostapi"]]["name"]
                valid_devices.append(device)
        return valid_devices

    def __audio_callback(self, audio_data: np.ndarray, frames: int, tm, status):
        _ = frames
        _ = status
        if self.__time_delta is None:
            self.__time_delta = time.time() - tm.inputBufferAdcTime
        if self._has_callback():
            self._invoke_callback(self.__time_delta + tm.inputBufferAdcTime, audio_data.flatten())

    def open(self):
        self.__audio_stream = sd.InputStream(
            device=self.__selected_device, channels=1, samplerate=sampling_rate, dtype="float32",
            blocksize=frame_size, callback=self.__audio_callback
        )
        super().open()
        self.__audio_stream.start()

    def close(self):
        super().close()
        if self.__audio_stream is not None:
            self.__audio_stream.close()


class _WrapCallback:
    def __init__(self, callback, **additional_kwargs):
        self.__callback = callback
        self.__additional_kwargs = additional_kwargs

    def __call__(self, *args, **kwargs):
        return self.__callback(*args, **kwargs, **self.__additional_kwargs)


class MultipleAudioInput(ContextManagerImpl):
    def __init__(self, audio_inputs: list, target_latency=0.125, **kwargs):
        assert len(audio_inputs) >= 2
        super().__init__(**kwargs)
        self.__audio_inputs = audio_inputs
        self.__target_latency = target_latency

        for index, audio_input in enumerate(self.__audio_inputs):
            audio_input.add_callback(_WrapCallback(self.__audio_callback, index=index))

        self.__lock0 = th.Lock()
        self.__buffers: list[deque[list[float, np.ndarray]] | None] = []
        self.__suppress_warning_until = time.time() + 8.0

    def __audio_callback(self, timestamp: float, audio_data: np.ndarray, index: int):
        if index != 0:
            with self.__lock0:
                q = self.__buffers[index]
                q.append([timestamp, audio_data])
                while q[0][0] + self.__target_latency * 2 < timestamp:
                    q.popleft()
                return

        target_timestamp = timestamp - self.__target_latency
        with self.__lock0:
            for i in range(1, len(self.__audio_inputs)):
                q = self.__buffers[i]
                if len(q) == 0:
                    if self.__suppress_warning_until < time.time():
                        logging.warning("sync error at audio #%d; no samples" % i)
                    continue
                nearest_index = -1
                nearest_delta = sys.float_info.max
                for ei, e0 in enumerate(q):
                    delta = abs(e0[0] - target_timestamp)
                    if delta < nearest_delta:
                        nearest_index = ei
                        nearest_delta = delta
                if nearest_index == -1 or nearest_delta > self.__target_latency * 0.5:
                    if self.__suppress_warning_until < time.time():
                        logging.warning("sync error at audio #%d; target = %f, available = [%f, %f]" % (
                            i, target_timestamp, q[0][0], q[-1][0]
                        ))
                    continue

                audio_data += q[nearest_index][1]

        self._invoke_callback(timestamp, audio_data)

    def open(self):
        self.__buffers = [None if index == 0 else deque() for index in range(len(self.__audio_inputs))]
        super().open()
        for audio_input in self.__audio_inputs:
            audio_input.open()

    def close(self):
        for audio_input in self.__audio_inputs:
            audio_input.close()
        super().close()


class AudioFileInput(ConcurrentContextManagerImpl):
    def __init__(self, source_file, additional_wait=0.0, **kwargs):
        super().__init__(**kwargs)
        self.__additional_wait = additional_wait

        with sf.SoundFile(source_file, mode="r") as f:
            if f.samplerate != sampling_rate or f.channels != 1:
                raise ValueError("Audio file formats not supported; detected fs=%d, ch=%d (required fs=%d, ch=1)" % (
                    f.samplerate, f.channels, sampling_rate))
            self.__audio_data = f.read(dtype="float32")

    def _handler(self):
        tm0 = time.time_ns() / 1000000000
        for offset in range(0, len(self.__audio_data) + int(self.__additional_wait * sampling_rate), frame_size):
            if self._should_stop():
                return False

            if offset + frame_size <= len(self.__audio_data):
                audio_data = self.__audio_data[offset:offset + frame_size]
            else:
                audio_data = np.zeros((frame_size,), dtype=np.float32)
                if offset < len(self.__audio_data):
                    audio_data[0:len(self.__audio_data) - offset] = self.__audio_data[offset:]

            tm1 = time.time_ns() / 1000000000
            required_wait = tm0 + offset / sampling_rate - tm1
            if required_wait > 0:
                time.sleep(required_wait)
            self._invoke_callback(offset / sampling_rate, audio_data)

        return True


class SuppressAudioInput(ContextManagerImpl):
    def __init__(self, stream, **kwargs):
        super().__init__(**kwargs)
        self.__lock0 = th.Lock()
        self.__suppress_count = 0
        self._stream = stream.add_callback(self.__audio_callback)

    def __audio_callback(self, timestamp: float, audio_data: np.ndarray):
        with self.__lock0:
            suppressing = (self.__suppress_count > 0)
        self._invoke_callback(
            timestamp, np.zeros(audio_data.shape, dtype=np.float32) if suppressing else audio_data)

    def lock(self):
        with self.__lock0:
            self.__suppress_count += 1

    def unlock(self):
        with self.__lock0:
            self.__suppress_count -= 1


class SharedModel:
    def __init__(self):
        self.__model = None
        self.__ref_count = 0
        self.__lock0 = th.Lock()
        self.__semaphore = th.BoundedSemaphore(1)

    def open(self, factory):
        with self.__lock0:
            if self.__ref_count == 0:
                self.__model = factory()
            self.__ref_count += 1
        return self

    def ref(self):
        with self.__lock0:
            if self.__ref_count == 0:
                raise RuntimeError()
            return self.__model

    def __enter__(self):
        self.__semaphore.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__semaphore.release()
        return False


@dataclasses.dataclass
class _VadFrame:
    audio_data: np.ndarray
    vad_level: float


@dataclasses.dataclass
class VadMeasureResult:
    latency: float
    peak_db: float
    rms_db: float
    woke: bool
    vad_max: float
    vad_ave: float


_vadMeasureWindow_peak = 0
_vadMeasureWindow_rms = 1
_vadMeasureWindow_woke = 2
_vadMeasureWindow_vad = 3


class VoiceActivityDetector(MultithreadContextManagerImpl):
    def __init__(self, stream, threshold=0.25, pre_hold=0.05, post_hold=0.25, post_apply=0.20,
                 soft_limit_length=30.0, hard_limit_length=60.0,
                 wakeup_peak_threshold_db=-36.0, wakeup_release=3.0,
                 keep_alive_interval=-1.0,
                 **kwargs):

        def __frames(time_):
            return (int(time_ * sampling_rate) + frame_size - 1) // frame_size

        super().__init__(acceptable_delay=1.0, **kwargs)
        self.__threshold = threshold
        self.__pre_hold_frames = __frames(pre_hold)
        self.__post_hold_frames0 = __frames(post_hold)
        self.__post_apply_frames = __frames(post_apply) if post_apply < post_hold else self.__post_hold_frames0
        self.__soft_limit1_frames = __frames(soft_limit_length)
        self.__soft_limit2_frames = __frames((soft_limit_length + hard_limit_length) // 2)
        self.__hard_limit_frames = __frames(hard_limit_length)
        self.__wakeup_peak_threshold = self.__db_to_level(wakeup_peak_threshold_db)
        self.__wakeup_release_frames = __frames(wakeup_release)
        self.__keep_alive_interval = keep_alive_interval

        self._stream = stream.add_callback(self.__audio_callback)

        self.__session = SharedModel().open(self.__open_model)
        self.__h: np.ndarray | None = None
        self.__c: np.ndarray | None = None

        self.__queueing = False
        self.__queue: deque[_VadFrame] = deque()
        self.__timestamp = 0.0
        self.__waiting_retirement_frames = 0
        self.__soft_limit_activated = 0
        self.__post_hold_frames1 = 0
        self.__last_keep_alive_sent = -1.0

        self.__left_wakeup_frames = 0

        self.__lock_measure = th.Lock()
        self.__measure_latency = 0.0
        measure_length = sampling_rate // frame_size
        self.__measure_window = np.zeros((4, measure_length,), dtype=np.float32)

    @staticmethod
    def __db_to_level(v):
        return np.power(10.0, v * np.log10(2.0) / 6.0)

    @staticmethod
    def __scaled_clipped_db(v):
        return max(-80.0, (6.0 / np.log10(2.0)) * np.log10(max(v, sys.float_info.epsilon)))

    @staticmethod
    def __open_model():
        options = onnxruntime.SessionOptions()
        options.log_severity_level = 4
        return onnxruntime.InferenceSession(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources", "silero_vad.onnx"),
            sess_options=options
        )

    def __available_frames(self):
        return (len(self.__queue) - self.__pre_hold_frames - self.__post_hold_frames1 +
                min(self.__post_apply_frames, self.__post_hold_frames1))

    def __retirement(self):
        frames = [self.__queue.popleft() for _ in range(self.__available_frames())]
        if self._has_callback():
            vad_level_list = [f.vad_level for f in frames]
            self._invoke_callback(
                self.__timestamp, np.concatenate([f.audio_data for f in frames]),
                t.AdditionalProperties(
                    vad_ave_level=np.average(vad_level_list), vad_max_level=np.max(vad_level_list)))
        while len(self.__queue) > self.__pre_hold_frames:
            self.__queue.popleft()

        self.__queueing = False
        self.__soft_limit_activated = 0
        self.__post_hold_frames1 = self.__post_hold_frames0

    def __send_keep_alive(self, timestamp: float, voice_detected: bool):
        if self.__keep_alive_interval < 0.0:
            return
        if voice_detected or self.__last_keep_alive_sent < 0.0:
            self.__last_keep_alive_sent = timestamp
            return
        if self.__last_keep_alive_sent + self.__keep_alive_interval < timestamp:
            self._invoke_callback(timestamp, None, None)
            self.__last_keep_alive_sent = timestamp

    def _request_handler(self, timestamp: float, audio_data: np.ndarray):
        with self.__lock_measure:
            self.__measure_latency = max(0.0, time.time() - timestamp)

            self.__measure_window = np.roll(self.__measure_window, 1, axis=1)
            self.__measure_window[:, 0] = 0.0

            peak = np.max(audio_data)
            self.__measure_window[_vadMeasureWindow_peak][0] = peak
            self.__measure_window[_vadMeasureWindow_rms][0] = np.sqrt(np.average(np.square(audio_data)))

            if peak >= self.__wakeup_peak_threshold:
                self.__left_wakeup_frames = self.__wakeup_release_frames
            if self.__left_wakeup_frames <= 0:
                if not self.__queueing:
                    self.__queue.append(_VadFrame(audio_data=audio_data, vad_level=0.0))
                    self.__queue.popleft()
                    self.__send_keep_alive(timestamp, False)
                    return
            else:
                self.__left_wakeup_frames -= 1
            self.__measure_window[_vadMeasureWindow_woke][0] = 1.0

        self.__send_keep_alive(timestamp, self.__queueing)

        with self.__session:
            out, h, c = self.__session.ref().run(None, {
                "input": audio_data.reshape(1, -1),
                "sr": np.array([sampling_rate], dtype=np.int64),
                "h": self.__h,
                "c": self.__c
            })
        self.__h = h
        self.__c = c
        out = float(out)
        has_voice = (out > self.__threshold)

        with self.__lock_measure:
            if not self.__queueing:
                self.__measure_window[_vadMeasureWindow_vad][:] = 0.0
            self.__measure_window[_vadMeasureWindow_vad][0] = out

        self.__queue.append(_VadFrame(audio_data=audio_data, vad_level=out))
        if not has_voice and not self.__queueing:
            self.__queue.popleft()

        if self.__queueing and self.__available_frames() >= self.__hard_limit_frames:
            self.__retirement()

        elif has_voice:
            if not self.__queueing:
                self.__timestamp = timestamp - self.__pre_hold_frames * frame_size / sampling_rate

            self.__queueing = True
            self.__waiting_retirement_frames = 0

            for level, limit_frames in [[1, self.__soft_limit1_frames], [2, self.__soft_limit2_frames]]:
                if self.__soft_limit_activated < level and len(self.__queue) >= limit_frames:
                    self.__soft_limit_activated = level
                    self.__post_hold_frames1 = max(1, self.__post_hold_frames1 // 2)

        elif self.__queueing:
            self.__waiting_retirement_frames += 1
            if self.__waiting_retirement_frames >= self.__post_hold_frames1 + self.__pre_hold_frames:
                self.__retirement()

    def __audio_callback(self, timestamp: float, audio_data: np.ndarray):
        self._send_request(timestamp, audio_data)

    def measure(self) -> VadMeasureResult:
        with self.__lock_measure:
            return VadMeasureResult(
                self.__measure_latency,
                self.__scaled_clipped_db(np.max(self.__measure_window[_vadMeasureWindow_peak])),
                self.__scaled_clipped_db(np.sqrt(np.average(np.square(self.__measure_window[_vadMeasureWindow_rms])))),
                np.max(self.__measure_window[_vadMeasureWindow_woke]) > 0.5,
                np.max(self.__measure_window[_vadMeasureWindow_vad]),
                np.average(self.__measure_window[_vadMeasureWindow_vad]))

    def open(self):
        self.__h = np.zeros((2, 1, 64), dtype=np.float32)
        self.__c = np.zeros((2, 1, 64), dtype=np.float32)

        self.__queueing = False
        self.__queue = deque([
            _VadFrame(audio_data=np.zeros((frame_size,), dtype=np.float32), vad_level=0)
            for _ in range(self.__pre_hold_frames)])
        self.__timestamp = 0.0
        self.__soft_limit_activated = 0
        self.__post_hold_frames1 = self.__post_hold_frames0

        self.__left_wakeup_frames = 0

        self.__measure_latency = 0.0
        self.__measure_window[:, :] = 0.0

        super().open()


@dataclasses.dataclass
class LanguageDetectionState:
    current_language: str = ""
    language_probs: dict[str, float] = dataclasses.field(default_factory=dict)
    guard_period: int = 0


class Transcriber(MultithreadContextManagerImpl):
    def __init__(self, stream, device="cpu", language="ja", auto_detect_language=True,
                 auto_detect_upper_threshold=0.9, auto_detect_lower_threshold=0.9,
                 auto_detect_guard_period=4,
                 embedding_type=None, min_duration=2.0, min_segment_duration=1.0,
                 save_audio_dir=None, **kwargs):

        super().__init__(**kwargs)
        self.__device = device
        self.__language = language
        self.__auto_detect_language = auto_detect_language
        self.__auto_detect_upper_threshold = auto_detect_upper_threshold
        self.__auto_detect_lower_threshold = auto_detect_lower_threshold
        self.__auto_detect_guard_period = auto_detect_guard_period
        self.__embedding_type = embedding_type
        self.__min_duration_in_samples = int(min_duration * sampling_rate)
        self.__min_segment_duration = min_segment_duration
        self.__save_audio_dir = save_audio_dir

        self.__model = SharedModel()
        self.__embedding_model = SharedModel()
        self.__channel = None
        self.__stub = None

        self.__current_language = self.__language
        self.__detected_languages = deque()
        self.__guard_period = self.__auto_detect_guard_period
        self.__language_probs = {}

        self._stream = stream.add_callback(self.__sentence_callback)

        # It appears that model opening must be done from the main thread, or it will fail.
        if self.__use_remote():
            return
        self.__model.open(self.__open_transcriber_model)
        if self.__embedding_type is not None:
            self.__embedding_model.open(self.__open_embedding_model)

    def __use_remote(self):
        return not self.__device.startswith("cpu") and not self.__device.startswith("gpu")

    def __open_transcriber_model(self):
        if self.__device == "gpu":
            return WhisperModel("large-v2", device="cuda", compute_type="float16")
        else:
            return WhisperModel("small", device="cpu", compute_type="int8")

    def __open_embedding_model(self):
        if self.__embedding_type == "speechbrain":
            return EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                run_opts={"device": "cuda"} if self.__device == "gpu" else None)
        elif self.__embedding_type == "pyannote":
            return Inference(Model.from_pretrained(
                "resources/pynannote_embedding_pytorch_model.bin"), window="whole")
        else:
            raise ValueError()

    def __update_language(self, audio_data: np.ndarray) -> str:
        if len(audio_data) > sampling_rate * 30:
            audio_data = audio_data[:sampling_rate * 30]

        if self.__channel is None:
            with self.__model:
                self.__detected_languages.append(
                    (transcriber_hack.detect_language(self.__model.ref(), audio_data), len(audio_data)))
        else:
            try:
                response = self.__stub.DetectLanguage(transcriber_service_pb2.DetectLanguageRequest(
                    audio_data=audio_data.tobytes()))
                languages = json.loads(response.detected_languages)
            except Exception as ex:
                languages = {}
                logging.error("Transcriber: exception raised", exc_info=ex)

            if len(languages) != 0:
                self.__detected_languages.append((languages, len(audio_data)))

        while (len(self.__detected_languages) > 8 or
               (len(self.__detected_languages) > 2 and
                sum([duration for _, duration in self.__detected_languages]) > sampling_rate * 90)):
            self.__detected_languages.popleft()

        if self.__guard_period > 0:
            self.__guard_period -= 1
            return self.__current_language

        language_probs = {}
        for prob_table, duration in self.__detected_languages:
            for language, prob in prob_table:
                e = language_probs.setdefault(language, [0.0, 0])
                e[0] += duration * prob
                e[1] += duration
        self.__language_probs = {language: e[0] / e[1] for language, e in language_probs.items()}

        if len(language_probs) == 0:
            return self.__current_language

        top_language, top_ratio = max(self.__language_probs.items(), key=lambda e_: e_[1])

        old_language = self.__current_language
        self.__current_language = (
            top_language if top_ratio > self.__auto_detect_upper_threshold else
            self.__language if top_ratio < self.__auto_detect_lower_threshold else
            self.__current_language)
        if old_language != self.__current_language:
            self.__guard_period = self.__auto_detect_guard_period

        return old_language

    def _request_handler(
            self, timestamp: float, raw_audio_data: np.ndarray | None, prop: t.AdditionalProperties | None):

        if raw_audio_data is None:
            self._invoke_callback(t.Sentence(-1.0, timestamp, ""))
            return

        if len(raw_audio_data) < self.__min_duration_in_samples:
            return

        prop.audio_level = max(np.sqrt(np.average(np.square(raw_audio_data))), sys.float_info.epsilon)
        audio_data = raw_audio_data * (0.25 / prop.audio_level)  # around -12dB

        tm0 = time.time_ns()

        if self.__auto_detect_language:
            old_language = self.__update_language(audio_data)
            if self.__current_language != old_language:
                self._invoke_callback(t.Sentence(
                    timestamp, timestamp, "", sentence_type=t.SentenceType.LanguageDetected, payload={
                        "old_language": old_language, "new_language": self.__current_language}))

        if self.__channel is None:
            with self.__model:
                segments, _ = self.__model.ref().transcribe(audio_data, beam_size=5, language=self.__current_language)
            segments_j = [[s.start, s.end, s.text] for s in segments
                          if s.end - s.start >= self.__min_segment_duration]

            embeddings = [None for _ in range(len(segments_j))]
            if self.__embedding_type == "speechbrain":
                for i, s in enumerate(segments_j):
                    audio_tensor = torch.from_numpy(audio_data[int(s[0] * sampling_rate):int(s[1] * sampling_rate)])
                    try:
                        with self.__embedding_model:
                            embeddings[i] = (self.__embedding_model.ref().encode_batch(audio_tensor)
                                             .cpu().detach().numpy().flatten())
                    except Exception as ex:
                        embeddings[i] = None
                        logging.error("Transcriber: exception raised", exc_info=ex)

            elif self.__embedding_type == "pyannote":
                for i, s in enumerate(segments_j):
                    # Seems to accept ndarray and Tensor, but cannot confirm working properly → go through .wav
                    with sf.SoundFile(
                            "tmp.wav", mode="w",
                            samplerate=sampling_rate, channels=1, subtype="FLOAT") as f:
                        f.write(audio_data[int(s[0] * sampling_rate):int(s[1] * sampling_rate)])
                    try:
                        with self.__embedding_model:
                            embeddings[i] = self.__embedding_model.ref()("tmp.wav").flatten().astype(np.float32)
                    except Exception as ex:
                        embeddings[i] = None
                        logging.error("Transcriber: exception raised", exc_info=ex)

        else:
            response = None
            try:
                response = self.__stub.Transcribe(transcriber_service_pb2.TranscribeRequest(
                    audio_data=audio_data.tobytes(), language=self.__current_language,
                    get_embedding=self.__embedding_type if self.__embedding_type is not None else "",
                    min_segment_duration=self.__min_segment_duration))
                segments_j = json.loads(response.segments)
            except Exception as ex:
                segments_j = []
                logging.error("Transcriber: exception raised", exc_info=ex)

            embeddings = [None for _ in range(len(segments_j))]
            if self.__embedding_type is not None and response is not None:
                for i, s in enumerate(segments_j):
                    embeddings[i] = np.frombuffer(response.embeddings[i], dtype=np.float32)

        tm1 = time.time_ns()
        processing_time = (tm1 - tm0) / 1000000000

        if processing_time > len(audio_data) / sampling_rate:
            logging.warning(
                "Processing time is taking longer than the length of the audio - %fs processing vs. %fs audio" %
                (processing_time, len(audio_data) / sampling_rate))

        if self._has_callback() or self.__save_audio_dir is not None:
            for i, s in enumerate(segments_j):
                segment_audio_data = raw_audio_data[int(s[0] * sampling_rate):int(s[1] * sampling_rate)]

                audio_file_name = None
                if self.__save_audio_dir is not None:
                    audio_file_name = os.path.join(self.__save_audio_dir, "%f.wav" % (timestamp + s[0]))
                    with sf.SoundFile(
                            audio_file_name, mode="w",
                            samplerate=sampling_rate, channels=1, subtype="FLOAT") as f:
                        f.write(segment_audio_data)

                s_prop = dataclasses.replace(prop)
                s_prop.segment_audio_level = max(
                    np.sqrt(np.average(np.square(segment_audio_data))), sys.float_info.epsilon)
                s_prop.language = self.__current_language

                self._invoke_callback(t.Sentence(
                    timestamp + s[0], timestamp + s[1], "", embedding=embeddings[i], prop=s_prop).add_text(
                    s[2], audio_file_name))

    def __sentence_callback(
            self, timestamp: float, audio_data: np.ndarray | None, prop: t.AdditionalProperties | None):
        self._send_request(timestamp, audio_data, prop)

    def ref_language_detection_state(self) -> LanguageDetectionState:
        ret = LanguageDetectionState()
        ret.current_language = self.__current_language
        ret.language_probs = self.__language_probs
        ret.guard_period = self.__guard_period
        return ret

    def open(self):
        if self.__use_remote():
            self.__channel = grpc.insecure_channel(self.__device)
            self.__stub = transcriber_service_pb2_grpc.TranscriberServiceStub(self.__channel)
        super().open()

    def close(self):
        super().close()
        if self.__channel is not None:
            self.__stub = None
            self.__channel.close()
            self.__channel = None


class InitialDiarization(MultithreadContextManagerImpl):
    def __init__(self, stream, backend, **kwargs):
        super().__init__(**kwargs)
        self.__backend = backend
        self._stream = stream.add_callback(self.__sentence_callback)

    def _request_handler(self, s: t.Sentence):
        if self.__backend is not None and s.embedding is not None:
            e = self.__backend.map([s.embedding])[0]
            s.person_id = e[0]
            s.person_name = e[1] if e[0] != -1 else t.unknown_person_name

        self._invoke_callback(s)

    def __sentence_callback(self, s: t.Sentence):
        self._send_request(s)


class DiarizationAndQualify(MultithreadContextManagerImpl):
    """
    note: Callbacks from this class may also be called after close()
    """
    def __init__(self, stream, backend, file_name=None, soft_limit=180.0, hard_limit=300.0, silent_interval=20.0,
                 merge_interval=10.0, merge_threshold=0.3, llm_opt: llm.QualifyOptions | None = None,
                 auto_sync=True, enable_simultaneous_interpretation=True,
                 separator_interval_on_interpretation_enabled=30.0, **kwargs):

        super().__init__(**kwargs)
        self.__backend = backend
        self.__file_name = file_name
        self.__soft_limit = soft_limit
        self.__hard_limit = hard_limit
        self.__silent_interval = silent_interval
        self.__merge_interval = merge_interval
        self.__merge_threshold = merge_threshold
        self.__llm_opt = dataclasses.replace(llm_opt) if llm_opt is not None else llm.QualifyOptions()
        self.__default_input_language = self.__llm_opt.input_language
        self.__auto_sync = auto_sync
        self.__enable_simultaneous_interpretation = enable_simultaneous_interpretation
        self.__separator_interval_on_interpretation_enabled = separator_interval_on_interpretation_enabled

        self.__lock0 = th.Lock()
        self.__lock1 = th.Lock()
        self.__history: list[t.SentenceGroup] = []
        self.__backend_generation = None

        self.__freeze_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.__callback_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.__interpretation_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        tools.recover_file(self.__file_name)
        if os.path.isfile(self.__file_name):
            with open(self.__file_name, "rb") as f:
                d = pickle.load(f)
            self.__history = d["history"]

        self._stream = stream.add_callback(self.__sentence_callback)

    def __delayed_callback(self, gr_index: int):
        self._invoke_callback(gr_index)
        if self.__auto_sync:
            self.sync()

    def __freeze(self, gr_index: int, gr: t.SentenceGroup):
        language_count = {}
        for s in gr.sentences:
            if s.prop is not None and s.prop.language != "":
                language_count.setdefault(s.prop.language, [0])[0] += 1
        self.__llm_opt.input_language = max(language_count.items(), key=lambda e_: e_[1][0])[0] \
            if len(language_count) != 0 else self.__default_input_language

        try:
            qualified = llm.qualify(gr.sentences, opt=self.__llm_opt)
            with self.__lock0:
                gr.qualified = qualified
                gr.state = t.SENTENCE_QUALIFIED
        except Exception as ex:
            with self.__lock0:
                gr.qualified = None
                gr.state = t.SENTENCE_QUALIFY_ERROR
            logging.error("failed to qualify sentence group", exc_info=ex)

        self.__callback_executor.submit(self.__delayed_callback, gr_index)

    def __freeze_last_group(self):
        assert self.__lock0.locked()
        gr_index = len(self.__history) - 1
        gr = self.__history[gr_index]
        gr.state = t.SENTENCE_QUALIFYING
        self.__freeze_executor.submit(self.__freeze, gr_index, gr)

    def __interpret(self, gr_index: int, s: t.Sentence):
        with self.__lock0:
            if len(s.si_state.waiting) == 0:
                return
            in_language = s.prop.language if s.prop is not None and s.prop.language != "" else None
            in_text = " ".join(s.si_state.processed_org + s.si_state.waiting)
            s.si_state.processing = " ".join(s.si_state.waiting)
            s.si_state.waiting.clear()

        out_text = llm.low_latency_interpretation(in_language, self.__llm_opt.output_language, in_text)

        with self.__lock0:
            s.si_state.processed_org.append(s.si_state.processing)
            s.si_state.processed_int = out_text
            s.si_state.processing = ""

        self.__callback_executor.submit(self.__delayed_callback, gr_index)

    def __initiate_interpret_on_sentence_created(self, gr_index: int, s: t.Sentence):
        assert self.__lock0.locked()
        if not self.__enable_simultaneous_interpretation:
            return
        if s.sentence_type != t.SentenceType.Sentence:
            return
        if s.prop is None or s.prop.language == "" or s.prop.language == self.__llm_opt.output_language:
            return
        s.si_state = t.SimultaneousInterpretationState(processed_org=[], waiting=[s.text])
        self.__interpretation_executor.submit(self.__interpret, gr_index, s)

    def __initiate_interpret_on_merged(self, gr_index: int, s: t.Sentence, s_add: t.Sentence) -> bool:
        assert self.__lock0.locked()
        if not self.__enable_simultaneous_interpretation:
            return False
        if s.si_state is None:
            return False
        s.si_state.waiting.append(s_add.text)
        self.__interpretation_executor.submit(self.__interpret, gr_index, s)
        return 0.0 < self.__separator_interval_on_interpretation_enabled < s.tm1 - s.tm0

    def _request_handler(self, received_time: float, s: t.Sentence):
        _ = received_time

        with self.__lock0:
            if s.tm0 < 0.0:
                if len(self.__history) != 0 and self.__history[-1].state == t.SENTENCE_BUFFER and \
                        self.__history[-1].sentences[-1].tm1 + self.__silent_interval <= s.tm1:
                    self.__freeze_last_group()
                return

            if len(self.__history) != 0 and self.__history[-1].state == t.SENTENCE_BUFFER:
                gr = self.__history[-1]
                next_gr_length = s.tm1 - gr.sentences[0].tm0
                if next_gr_length > self.__soft_limit and (
                        s.person_id == -1 or gr.sentences[-1].person_id != s.person_id):
                    self.__freeze_last_group()
                elif gr.sentences[-1].tm1 + self.__silent_interval <= s.tm0:
                    self.__freeze_last_group()
                elif next_gr_length > self.__hard_limit:
                    self.__freeze_last_group()

            if len(self.__history) == 0 or self.__history[-1].state != t.SENTENCE_BUFFER:
                self.__history.append(t.SentenceGroup(t.SENTENCE_BUFFER, []))

            gr_index = len(self.__history) - 1
            gr = self.__history[-1]
            merged = False
            if s.embedding is not None and len(gr.sentences) != 0:
                s0 = gr.sentences[-1]
                if s0.embedding is not None and s0.tm1 + self.__merge_interval > s.tm0 and \
                        self.__backend.metrics(s0.embedding, s.embedding) < self.__merge_threshold:
                    s0.merge(s)
                    if self.__initiate_interpret_on_merged(gr_index, s0, s):
                        gr.sentences.append(t.Sentence(
                            s0.tm1, s0.tm1, "", sentence_type=t.SentenceType.SentenceSeparator))
                    merged = True

            if not merged:
                s0 = s.clone()
                gr.sentences.append(s0)
                self.__initiate_interpret_on_sentence_created(gr_index, s0)

        self.__callback_executor.submit(self.__delayed_callback, gr_index)

    def __sentence_callback(self, s: t.Sentence):
        self._send_request(time.time(), s)

    @staticmethod
    def __map_sentences(sentences: list[t.Sentence] | None, embeddings):
        if sentences is None or len(sentences) == 0:
            return []
        offsets = []
        for i, s in enumerate(sentences):
            if s.embedding is None:
                offsets.append(-1)
            else:
                offsets.append(len(embeddings))
                embeddings.append(s.embedding)
        return offsets

    @staticmethod
    def __apply_mapping_result(sentences: list[t.Sentence] | None, offsets, mapping_result):
        if sentences is None or len(sentences) == 0:
            return
        for i in range(len(offsets)):
            s = sentences[i]
            if offsets[i] == -1 or s.embedding is None:
                continue
            r = mapping_result[offsets[i]]
            s.person_id = r[0]
            s.person_name = r[1]

    def __check_backend_update(self):
        with self.__lock1:
            current_generation = self.__backend.get_generation()
            if self.__backend_generation == current_generation:
                return

            with self.__lock0:
                sentences0_to_offsets = []
                sentences1_to_offsets = []
                embeddings = []
                for gr_index, gr in enumerate(self.__history):
                    sentences0_to_offsets.append(self.__map_sentences(gr.sentences, embeddings))
                    sentences1_to_offsets.append(self.__map_sentences(
                        gr.qualified.corrected_sentences if gr.qualified is not None else None, embeddings))

            mapping_result = self.__backend.map(embeddings, update=False) if len(embeddings) != 0 else []

            with self.__lock0:
                # Note that the state may have changed while unlocking lock0 and
                # the number of history may have changed.
                for gr_index in range(len(sentences0_to_offsets)):
                    gr = self.__history[gr_index]
                    self.__apply_mapping_result(gr.sentences, sentences0_to_offsets[gr_index], mapping_result)
                    self.__apply_mapping_result(
                        gr.qualified.corrected_sentences if gr.qualified is not None else None,
                        sentences1_to_offsets[gr_index], mapping_result)

            self.__backend_generation = current_generation

    def open(self):
        # Reprocess groups for which qualifying process was not completed before the previous exit
        with self.__lock0:
            for gr_index, gr in enumerate(self.__history):
                if gr.state == t.SENTENCE_QUALIFYING or gr.state == t.SENTENCE_BUFFER:
                    gr.state = t.SENTENCE_QUALIFYING
                    self.__freeze_executor.submit(self.__freeze, gr_index, gr)
        super().open()

    def group_count(self) -> int:
        with self.__lock0:
            return len(self.__history)

    def ref_group(self, gr_index: int) -> t.SentenceGroup:
        self.__check_backend_update()
        with self.__lock0:
            return dataclasses.replace(self.__history[gr_index])

    def sync(self):
        with self.__lock0:
            with tools.SafeWrite(self.__file_name, "wb") as f:
                pickle.dump({"history": self.__history}, f.stream)


@dataclasses.dataclass
class EmbeddingDatabaseConfiguration:
    threshold: float = 0.6
    dbscan_eps: float = 0.4
    dbscan_min_samples: int = 6
    min_matched_embeddings_to_inherit_cluster: int = 6
    min_matched_embeddings_to_match_person: int = 6


@dataclasses.dataclass
class Configuration:
    input_devices: list[str] | None = None
    device: str = "cpu"  # "cpu" "gpu" or access point
    language: str = "ja"  # copied to llm_opt.input_language
    enable_auto_detect_language: bool = True
    enable_simultaneous_interpretation: bool = False

    vad_threshold: float = 0.5
    vad_pre_hold: float = 0.05
    vad_post_hold: float = 0.50
    vad_post_apply: float = 0.20
    vad_soft_limit_length: float = 30.0
    vad_hard_limit_length: float = 60.0
    vad_wakeup_peak_threshold_db: float = -48.0
    vad_wakeup_release: float = 3.0

    embedding_type: str | None = None  # "speechbrain" "pyannote"
    transcribe_min_duration: float = 2.0
    transcribe_min_segment_duration: float = 1.0
    keep_audio_file_for: float = -1.0

    emb_sb: EmbeddingDatabaseConfiguration = dataclasses.field(default_factory=EmbeddingDatabaseConfiguration)
    emb_pn: EmbeddingDatabaseConfiguration = dataclasses.field(default_factory=EmbeddingDatabaseConfiguration)
    max_hold_embeddings: int = 40

    qualify_soft_limit: float = 180.0
    qualify_hard_limit: float = 300.0
    qualify_silent_interval: float = 20.0
    qualify_merge_interval: float = 10.0
    qualify_merge_threshold: float = 0.3

    llm_opt: llm.QualifyOptions | None = None

    disabled_plugins: list[str] = dataclasses.field(default_factory=lambda: ["simple_memo"])


class Reader:
    def __init__(self, file_name):
        with open(file_name, "rb") as f:
            d = pickle.load(f)
        self.__history = d["history"]

    def group_count(self) -> int:
        return len(self.__history)

    def ref_group(self, gr_index: int) -> t.SentenceGroup:
        return self.__history[gr_index]


_plugin_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)


class _WrapAudioCallback:
    def __init__(self, device_index, callback):
        self.__device_index = device_index
        self.__callback = callback

    def __call__(self, timestamp: float, audio_data: np.ndarray):
        _plugin_executor.submit(self.__callback, self.__device_index, timestamp, audio_data)


class _WrapVadCallback:
    def __init__(self, callback):
        self.__callback = callback

    def __call__(self, timestamp: float, audio_data: np.ndarray | None, prop: t.AdditionalProperties | None):
        _ = prop
        _plugin_executor.submit(self.__callback, timestamp, audio_data)


class _WrapSegmentCallback:
    def __init__(self, callback):
        self.__callback = callback

    def __call__(self, s: t.Sentence):
        _plugin_executor.submit(self.__callback, s.tm0, s.tm1, s.person_name if s.person_id != -1 else None, s.text)


class Application:
    def __init__(self, conf=Configuration(), ui_language="en"):
        self.__conf = dataclasses.replace(conf)
        self.__ui_language = ui_language

        devices = AudioInput.query_valid_input_devices()
        devices_by_name = {d["name"]: d for d in devices}
        if self.__conf.input_devices is not None:
            self.__conf.input_devices = [d for d in self.__conf.input_devices if d in devices_by_name.keys()]
        if self.__conf.input_devices is None or len(self.__conf.input_devices) == 0:
            default_index = sd.default.device[0]
            default_device = next(d for d in devices if d["index"] == default_index)
            self.__conf.input_devices = [default_device["name"]]

        if self.__conf.llm_opt is None:
            self.__conf.llm_opt = llm.QualifyOptions()
        self.__conf.llm_opt.input_language = self.__conf.language

        self.__audio: dict[str, AudioInput] = {
            name: AudioInput(selected_device=next(d for d in devices if d["name"] == name)["index"])
            for name in self.__conf.input_devices}

        self.__audio_mux = MultipleAudioInput(list(self.__audio.values())) if len(self.__audio) >= 2 \
            else next(iter(self.__audio.values()))

        self.__suppress_audio = SuppressAudioInput(self.__audio_mux)

        self.__vad = VoiceActivityDetector(
            self.__suppress_audio,
            threshold=self.__conf.vad_threshold, pre_hold=self.__conf.vad_pre_hold,
            post_hold=self.__conf.vad_post_hold, post_apply=self.__conf.vad_post_apply,
            soft_limit_length=self.__conf.vad_soft_limit_length,
            hard_limit_length=self.__conf.vad_hard_limit_length,
            wakeup_peak_threshold_db=self.__conf.vad_wakeup_peak_threshold_db,
            wakeup_release=self.__conf.vad_wakeup_release,
            keep_alive_interval=1.0)

        self.__save_audio_dir = os.path.join(data_dir_name, "audio")
        os.makedirs(self.__save_audio_dir, exist_ok=True)
        self.__transcriber = Transcriber(
            self.__vad, device=self.__conf.device, language=self.__conf.language,
            auto_detect_language=self.__conf.enable_auto_detect_language,
            embedding_type=self.__conf.embedding_type,
            min_duration=self.__conf.transcribe_min_duration,
            min_segment_duration=self.__conf.transcribe_min_segment_duration,
            save_audio_dir=self.__save_audio_dir if self.__conf.keep_audio_file_for >= 0.0 else None)

        self.__db = db.HybridEmbeddingDatabase(
            data_dir_name,
            param_for_speechbrain=self.__get_embedding_database_param(self.__conf, self.__conf.emb_sb),
            param_for_pyannote=self.__get_embedding_database_param(self.__conf, self.__conf.emb_pn))

        self.__initial_diarization = InitialDiarization(
            self.__transcriber, self.__db)

        self.__qualifier_file_name = self.__get_qualifier_file_name()
        self.__qualifier = DiarizationAndQualify(
            self.__initial_diarization, self.__db,
            file_name=os.path.join(data_dir_name, self.__qualifier_file_name),
            soft_limit=self.__conf.qualify_soft_limit, hard_limit=self.__conf.qualify_hard_limit,
            silent_interval=self.__conf.qualify_silent_interval,
            merge_interval=self.__conf.qualify_merge_interval, merge_threshold=self.__conf.qualify_merge_threshold,
            llm_opt=self.__conf.llm_opt,
            enable_simultaneous_interpretation=self.__conf.enable_simultaneous_interpretation)

        self.__sync_stop: th.Semaphore | None = None
        self.__sync_thread = None
        self.__opened = False

        # Generate a file where the status is saved if it was the first run of the day
        self.__qualifier.sync()

        self.__plugins = {}
        self.__launch_plugins()

        self.__clean_audio_dir()

    @staticmethod
    def __get_embedding_database_param(c: Configuration, ec: EmbeddingDatabaseConfiguration):
        return {
            "threshold": ec.threshold,
            "dbscan_eps": ec.dbscan_eps,
            "dbscan_min_samples": ec.dbscan_min_samples,
            "min_matched_embeddings_to_inherit_cluster": ec.min_matched_embeddings_to_inherit_cluster,
            "min_matched_embeddings_to_match_person": ec.min_matched_embeddings_to_match_person,
            "max_hold_embeddings": c.max_hold_embeddings
        }

    @staticmethod
    def find_installed_plugins():
        ret = []
        for dir_name in glob.glob("plugins/*"):
            if not os.path.isdir(dir_name):
                continue
            plugin_name = os.path.basename(dir_name)
            if plugin_name.startswith(".") or plugin_name.startswith("_"):
                continue
            ret.append((dir_name, plugin_name))
        return ret

    def __launch_plugins(self):
        for dir_name, plugin_name in self.find_installed_plugins():
            if plugin_name in self.__conf.disabled_plugins:
                continue

            try:
                logging.info("Launching plugin \"%s\"" % dir_name)
                m = importlib.import_module(".".join(os.path.split(dir_name)))
                plugin_data_dir = os.path.join(data_dir_name, "plugins", plugin_name)
                os.makedirs(plugin_data_dir, exist_ok=True)
                p: pl.Plugin = m.create(
                    __sampling_rate=sampling_rate, __ui_language=self.__ui_language, __data_dir=plugin_data_dir,
                    __input_language=self.__conf.llm_opt.input_language,
                    __output_language=self.__conf.llm_opt.output_language
                )
            except Exception as ex:
                logging.error("Failed to launch plugin \"%s\"" % dir_name, exc_info=ex)
                continue

            self.__plugins[plugin_name] = p

            flags = p.injection_point()
            if flags & pl.FLAG_AUDIO:
                for device_index, name in enumerate(sorted(self.__audio.keys())):
                    self.__audio[name].add_callback(_WrapAudioCallback(device_index, p.on_audio_frame))
            if flags & pl.FLAG_VAD:
                self.__vad.add_callback(_WrapVadCallback(p.on_vad_frame))
            if flags & pl.FLAG_SPEECH_SEGMENT:
                self.__initial_diarization.add_callback(_WrapSegmentCallback(p.on_speech_segment))

    def __clean_audio_dir(self):
        tm0 = time.time()
        for file_name in glob.glob(os.path.join(self.__save_audio_dir, "*.wav")):
            r = re.match(r"(\d+\.\d+)\.wav", os.path.basename(file_name))
            if r is not None and self.__conf.keep_audio_file_for >= 0.0 and \
                    float(r.group(1)) + self.__conf.keep_audio_file_for < tm0:
                os.remove(file_name)

    def get_current_configuration(self):
        return dataclasses.replace(self.__conf)

    def log_file_may_changed(self):
        return self.__qualifier_file_name != self.__get_qualifier_file_name()

    @staticmethod
    def __get_qualifier_file_name():
        tm = time.localtime()
        return "q.%04d-%02d-%02d.pickle" % (tm.tm_year, tm.tm_mon, tm.tm_mday)

    def __sync(self):
        self.__qualifier.sync()
        self.__db.sync()

    def __interval_sync(self):
        while True:
            if self.__sync_stop.acquire(timeout=60.0):
                break
            self.__sync()

    def current_configuration(self):
        return self.__conf

    def ref_plugins(self) -> dict[str, pl.Plugin]:
        return self.__plugins

    def open(self):
        self.__qualifier.open()

        self.__sync_stop = th.Semaphore(0)
        self.__sync_thread = th.Thread(target=self.__interval_sync)
        self.__sync_thread.start()

        self.__opened = True

    def close(self):
        self.__opened = False

        self.__sync_stop.release()
        self.__sync_thread.join()
        self.__sync_thread = None

        self.__qualifier.close()

    def is_opened(self):
        return self.__opened

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def suppress_audio_lock(self):
        self.__suppress_audio.lock()

    def suppress_audio_unlock(self):
        self.__suppress_audio.unlock()

    def measure(self) -> VadMeasureResult:
        return self.__vad.measure()

    def ref_language_detection_state(self) -> LanguageDetectionState:
        return self.__transcriber.ref_language_detection_state()

    def map(self, embeddings: list[np.ndarray], update=True) -> list[[int, str | None]]:
        return self.__db.map(embeddings, update=update)

    def get_persons(self) -> list[db.Person]:
        return self.__db.get_persons()

    def rename(self, person_id: int, new_name: str):
        self.__db.rename(person_id, new_name)
        self.__sync()

    def erase(self, person_id: int):
        self.__db.erase(person_id)
        self.__sync()

    def plot_db(self, embedding_type: str):
        return self.__db.plot(embedding_type)

    def add_group_updated_callback(self, callback):
        self.__qualifier.add_callback(callback)

    def group_count(self) -> int:
        return self.__qualifier.group_count()

    def ref_group(self, gr_index: int) -> t.SentenceGroup:
        return self.__qualifier.ref_group(gr_index)

    @staticmethod
    def list_history():
        ret = []
        for file in glob.glob(os.path.join(data_dir_name, "q.*")):
            r0 = re.search(r"q.(\d{4})-(\d{2})-(\d{2}).pickle$", file)
            if r0 is not None:
                ret.append(int(r0.group(1)) * 10000 + int(r0.group(2)) * 100 + int(r0.group(3)))
        ret.sort()
        return ret

    @staticmethod
    def open_history(date_index: int):
        file_name = "q.%04d-%02d-%02d.pickle" % (date_index // 10000, date_index // 100 % 100, date_index % 100)
        file_name = os.path.join(data_dir_name, file_name)
        if not os.path.isfile(file_name):
            raise ValueError("file \"%s\" not found")
        return Reader(file_name)
