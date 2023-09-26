import logging
import json
import dataclasses
import threading as th

# noinspection PyPackageRequirements
import grpc
import transcriber_service_pb2
import transcriber_service_pb2_grpc

import numpy as np
import soundfile as sf
import torch
from faster_whisper import WhisperModel
from speechbrain.pretrained import EncoderClassifier
# noinspection PyPackageRequirements
from pyannote.audio import Model
# noinspection PyPackageRequirements
from pyannote.audio import Inference

import common
import tools
import transcriber_hack


sampling_rate = common.sampling_rate


@dataclasses.dataclass
class TranscribedSegment:
    tm0: float = 0.0
    tm1: float = 0.0
    text: str = ""
    embedding: np.ndarray | None = None


class Transcriber:
    def __init__(self, device="cpu", embedding_type=None, min_segment_duration=1.0):
        self.__device = device
        self.__embedding_type = embedding_type
        self.__min_segment_duration = min_segment_duration

        self.__model = tools.SharedModel()
        self.__embedding_model = tools.SharedModel()
        self.__channel = None
        self.__stub = None

        self.__lock0 = th.Lock()
        self.__detect_language_future = None
        self.__transcribe_future = None

        # It appears that opening model must be done from the main thread, or it will fail.
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

    @staticmethod
    def __timeout(audio_data: np.ndarray):
        return 30.0 + len(audio_data) / sampling_rate * 1.5

    def __detect_language_process(self, audio_data: np.ndarray) -> list[tuple[str, float]] | None:
        if self.__channel is None:
            with self.__model:
                languages = transcriber_hack.detect_language(self.__model.ref(), audio_data)
        else:
            try:
                response = self.__stub.DetectLanguage(transcriber_service_pb2.DetectLanguageRequest(
                    audio_data=audio_data.tobytes()))
                languages = json.loads(response.detected_languages)
            except Exception as ex:
                languages = {}
                logging.error("Transcriber: exception raised", exc_info=ex)

        return languages if languages is not None and len(languages) != 0 else None

    def detect_language(self, audio_data: np.ndarray) -> list[tuple[str, float]] | None:
        f = tools.async_call(self.__detect_language_process, audio_data, timeout=self.__timeout(audio_data))
        with self.__lock0:
            self.__detect_language_future = f
        return f.wait_result()

    def __transcribe_process(self, audio_data: np.ndarray, language: str) -> list[TranscribedSegment]:
        if self.__channel is None:
            with self.__model:
                raw_segments, _ = self.__model.ref().transcribe(audio_data, beam_size=5, language=language)
            segments = [TranscribedSegment(tm0=s.start, tm1=s.end, text=s.text) for s in raw_segments
                        if s.end - s.start >= self.__min_segment_duration]

            if self.__embedding_type == "speechbrain":
                for i, s in enumerate(segments):
                    audio_tensor = torch.from_numpy(audio_data[int(s.tm0 * sampling_rate):int(s.tm1 * sampling_rate)])
                    try:
                        with self.__embedding_model:
                            s.embedding = (self.__embedding_model.ref().encode_batch(audio_tensor)
                                           .cpu().detach().numpy().flatten())
                    except Exception as ex:
                        s.embedding = None
                        logging.error("Transcriber: exception raised", exc_info=ex)

            elif self.__embedding_type == "pyannote":
                for i, s in enumerate(segments):
                    # Seems to accept ndarray and Tensor, but cannot confirm working properly → go through .wav
                    with sf.SoundFile(
                            "tmp.wav", mode="w",
                            samplerate=sampling_rate, channels=1, subtype="FLOAT") as f:
                        f.write(audio_data[int(s.tm0 * sampling_rate):int(s.tm1 * sampling_rate)])
                    try:
                        with self.__embedding_model:
                            s.embedding = self.__embedding_model.ref()("tmp.wav").flatten().astype(np.float32)
                    except Exception as ex:
                        s.embedding = None
                        logging.error("Transcriber: exception raised", exc_info=ex)

        else:
            response = None
            try:
                response = self.__stub.Transcribe(transcriber_service_pb2.TranscribeRequest(
                    audio_data=audio_data.tobytes(), language=language,
                    get_embedding=self.__embedding_type if self.__embedding_type is not None else "",
                    min_segment_duration=self.__min_segment_duration))
                segments_j = json.loads(response.segments)
            except Exception as ex:
                segments_j = []
                logging.error("Transcriber: exception raised", exc_info=ex)

            segments = [TranscribedSegment(tm0=s[0], tm1=s[1], text=s[2]) for s in segments_j]

            if self.__embedding_type is not None and response is not None:
                for i, s in enumerate(segments_j):
                    segments[i].embedding = np.frombuffer(response.embeddings[i], dtype=np.float32)

        return segments

    def transcribe(self, audio_data: np.ndarray, language: str) -> list[TranscribedSegment] | None:
        f = tools.async_call(self.__transcribe_process, audio_data, language, timeout=self.__timeout(audio_data))
        with self.__lock0:
            self.__transcribe_future = f
        return f.wait_result()

    def open(self):
        if self.__use_remote():
            self.__channel = grpc.insecure_channel(self.__device)
            self.__stub = transcriber_service_pb2_grpc.TranscriberServiceStub(self.__channel)

    def close(self):
        with self.__lock0:
            if self.__detect_language_future is not None:
                self.__detect_language_future.cancel()
            if self.__transcribe_future is not None:
                self.__transcribe_future.cancel()
        if self.__channel is not None:
            self.__stub = None
            self.__channel.close()
            self.__channel = None
