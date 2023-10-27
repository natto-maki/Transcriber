import argparse
import logging
import random
import time
import json

# noinspection PyPackageRequirements
import grpc
import main_server_service_pb2
import main_server_service_pb2_grpc

import soundfile as sf

sampling_rate = 16000
frame_size = 2048


def main():
    parser = argparse.ArgumentParser(description="Test code for connecting main_server")
    parser.add_argument(
        "--endpoint", metavar="ADDRESS:PORT", action="store",
        dest="endpoint", required=True)
    parser.add_argument(
        "--input", metavar="AUDIO_FILE", action="store",
        dest="input", required=True)
    opt = parser.parse_args()

    session_id = "".join(["%02x" % random.randint(0, 255) for _ in range(8)])
    logging.info("session_id = %s" % session_id)

    with sf.SoundFile(opt.input, mode="r") as f:
        if f.samplerate != sampling_rate or f.channels != 1:
            raise ValueError("Audio file formats not supported; detected fs=%d, ch=%d (required fs=%d, ch=1)" % (
                f.samplerate, f.channels, sampling_rate))
        audio_data = f.read(dtype="float32")

    channel = grpc.insecure_channel(opt.endpoint)
    stub = main_server_service_pb2_grpc.MainServerServiceStub(channel)

    r0 = stub.Open(main_server_service_pb2.OpenRequest(session_id=session_id))
    if r0.result != "":
        raise Exception("status_code = " + r0.result)

    tm0 = time.time()
    offset = 0
    while offset < len(audio_data):
        length = min(frame_size, len(audio_data) - offset)
        r0 = stub.Push(main_server_service_pb2.PushRequest(
            session_id=session_id, audio_data=audio_data[offset:offset + length].tobytes()))
        if r0.result != "":
            raise Exception("status_code = " + r0.result)

        offset += length
        tm1 = tm0 + offset / sampling_rate
        tm_current = time.time()
        if tm_current < tm1:
            time.sleep(tm1 - tm_current)

    last_read = 0
    for _ in range(4):
        r0 = stub.Read(main_server_service_pb2.ReadRequest(
            session_id=session_id, begin_time=last_read, end_time=0))
        if r0.result != "":
            raise Exception("status_code = " + r0.result)
        j = json.loads(r0.payload)
        if len(j) != 0:
            logging.info("sentences: \n" + "\n".join([json.dumps(je, ensure_ascii=False) for je in j]))
            last_read = j[-1]["time"] + 0.001
        time.sleep(3)

    r0 = stub.Read(main_server_service_pb2.ReadRequest(
            session_id=session_id, source="simple_memo", read_parameters="foo"))
    if r0.result != "":
        raise Exception("status_code = " + r0.result)
    logging.info("result: " + r0.payload)

    r0 = stub.Close(main_server_service_pb2.CloseRequest(session_id=session_id))
    if r0.result != "":
        raise Exception("status_code = " + r0.result)

    channel.close()


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s: %(name)s:%(funcName)s:%(lineno)d %(levelname)s: %(message)s', level=logging.INFO)
    main()
