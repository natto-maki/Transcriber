import logging
import os
import time
import threading as th
import concurrent.futures
from collections import deque
import dataclasses
import json

# noinspection PyPackageRequirements
import grpc
import main_server_service_pb2
import main_server_service_pb2_grpc

import numpy as np
# noinspection PyPackageRequirements
import yaml

import common
import main

sampling_rate = common.sampling_rate
frame_size = common.frame_size
conf_file_name = "main_server_conf.yaml"


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
    session_id: str
    audio_input: RemoteAudioInput | None = None
    app: main.Application | None = None
    app_thread: concurrent.futures.ThreadPoolExecutor | None = None  # TODO should be use multiprocess
    last_access: float = 0.0


class Servicer(main_server_service_pb2_grpc.MainServerServiceServicer):
    def __init__(self):
        self.__lock0 = th.Lock()
        self.__sessions: dict[str, Session] = {}

        self.__conf = main.Configuration()
        if os.path.isfile(conf_file_name):
            with open(conf_file_name, mode="r") as f:
                self.__conf = yaml.load(f, Loader=yaml.Loader)

    def __start_app(self, s: Session, sm: th.Semaphore | None):
        logging.info(s.session_id + ": attempting to open app")

        s.app = main.Application(
            conf=self.__conf, audio_input=s.audio_input, data_dir_name=os.path.join("data", s.session_id))

        # with open(conf_file_name, mode="w") as f:
        #     yaml.dump(s.app.get_current_configuration(), f)

        s.app.open()
        logging.info(s.session_id + ": app opened")
        if sm is not None:
            sm.release()

    @staticmethod
    def __stop_app(s: Session):
        logging.info(s.session_id + ": attempting to close app")
        if s.app.is_opened():
            s.app.close()
        logging.info(s.session_id + ": app closed")

    @staticmethod
    def __push(s: Session, audio_frame: np.ndarray):
        s.audio_input.push(audio_frame)

    @staticmethod
    def __read(s: Session, source: str, read_parameters: str, begin_time: int, end_time: int):
        if source == "":
            ret = []
            for group_index in range(s.app.group_count()):
                g = s.app.ref_group(group_index)
                sentences = g.sentences
                if g.qualified is not None and g.qualified.corrected_sentences is not None:
                    sentences = g.qualified.corrected_sentences
                for s in sentences:
                    if begin_time <= s.tm0 and (end_time == 0 or s.tm0 < end_time):
                        ret.append({"time": s.tm0, "text": s.text, "name": s.person_name})
            return json.dumps(ret)

        else:
            plugins = s.app.ref_plugins()
            if source in plugins:
                return plugins[source].read_state(read_parameters)

        return ""

    def __check_session(self, session_id: str, blocking=False) -> Session:
        with self.__lock0:
            if session_id in self.__sessions:
                return self.__sessions[session_id]
            s = Session(session_id)
            s.audio_input = RemoteAudioInput()
            s.app_thread = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            self.__sessions[session_id] = s

        sm = th.Semaphore(0) if blocking else None
        s.app_thread.submit(self.__start_app, s, sm)
        if sm is not None:
            sm.acquire()
        return s

    def Open(self, request, context):
        self.__check_session(request.session_id, blocking=True)
        return main_server_service_pb2.BaseResponse(result="")

    def Close(self, request, context):
        with self.__lock0:
            if request.session_id in self.__sessions:
                s = self.__sessions[request.session_id]
                s.app_thread.submit(self.__stop_app, s)
                del self.__sessions[request.session_id]
        return main_server_service_pb2.BaseResponse(result="")

    def Push(self, request, context):
        s = self.__check_session(request.session_id)
        s.app_thread.submit(self.__push, s, np.frombuffer(request.audio_data, dtype=np.float32))
        return main_server_service_pb2.BaseResponse(result="")

    def Read(self, request, context):
        s = self.__check_session(request.session_id)
        sm = th.Semaphore(0)
        ret = main_server_service_pb2.ReadResponse()

        def _handle_read():
            ret.payload = self.__read(s, request.source, request.read_parameters, request.begin_time, request.end_time)
            ret.result = ""
            sm.release()

        s.app_thread.submit(_handle_read)
        sm.acquire()
        return ret


def serve():
    port = "7860"
    server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=10))
    main_server_service_pb2_grpc.add_MainServerServiceServicer_to_server(Servicer(), server)
    server.add_insecure_port("[::]:" + port)
    server.start()
    logging.info("Server started, listening on " + port)
    server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s: %(name)s:%(funcName)s:%(lineno)d %(levelname)s: %(message)s', level=logging.INFO)
    serve()
