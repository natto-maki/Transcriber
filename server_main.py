"""
以下が必要かも
export LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib
"""
import logging
import threading as th
from concurrent import futures
import json

# noinspection PyPackageRequirements
import grpc
import transcriber_service_pb2
import transcriber_service_pb2_grpc

import numpy as np
import torch
import soundfile as sf
from faster_whisper import WhisperModel
from speechbrain.pretrained import EncoderClassifier
# noinspection PyPackageRequirements
from pyannote.audio import Model
# noinspection PyPackageRequirements
from pyannote.audio import Inference

import transcriber_hack

sampling_rate = 16000


class Servicer(transcriber_service_pb2_grpc.TranscriberServiceServicer):
    def __init__(self, device="gpu"):
        self.mutex = th.Lock()
        if device == "gpu":
            self.model = WhisperModel("large-v2", device="cuda", compute_type="float16")
        else:
            self.model = WhisperModel("small", device="cpu", compute_type="int8")

        self.embedding_model0 = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": "cuda"} if device == "gpu" else None)

        self.embedding_model1 = Inference(Model.from_pretrained(
            "resources/pynannote_embedding_pytorch_model.bin"), window="whole")

    def Transcribe(self, request, context):
        self.mutex.acquire()
        try:
            audio_data = np.frombuffer(request.audio_data, dtype=np.float32)
            segments, _ = self.model.transcribe(audio_data, beam_size=5, language=request.language)
            segments_j = [[s.start, s.end, s.text] for s in segments
                          if s.end - s.start >= request.min_segment_duration]

            embeddings = []
            if request.get_embedding == "speechbrain":
                for s in segments_j:
                    audio_tensor = torch.from_numpy(audio_data[int(s[0] * sampling_rate):int(s[1] * sampling_rate)])
                    embeddings.append(self.embedding_model0.encode_batch(audio_tensor).cpu().detach().numpy()
                                      .flatten().astype(np.float32).tobytes())
            elif request.get_embedding == "pyannote":
                for s in segments_j:
                    with sf.SoundFile(
                            "tmp.wav", mode="w",
                            samplerate=sampling_rate, channels=1, subtype="FLOAT") as f:
                        f.write(audio_data[int(s[0] * sampling_rate):int(s[1] * sampling_rate)])
                    embeddings.append(self.embedding_model1("tmp.wav").flatten().astype(np.float32).tobytes())

            return transcriber_service_pb2.TranscribeResponse(segments=json.dumps(segments_j), embeddings=embeddings)

        finally:
            self.mutex.release()

    def DetectLanguage(self, request, context):
        self.mutex.acquire()
        try:
            audio_data = np.frombuffer(request.audio_data, dtype=np.float32)
            return transcriber_service_pb2.DetectLanguageResponse(
                detected_languages=json.dumps(transcriber_hack.detect_language(self.model, audio_data)))
        finally:
            self.mutex.release()


def serve():
    port = "7860"
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    transcriber_service_pb2_grpc.add_TranscriberServiceServicer_to_server(Servicer(), server)
    server.add_insecure_port("[::]:" + port)
    server.start()
    logging.info("Server started, listening on " + port)
    server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig()
    serve()
