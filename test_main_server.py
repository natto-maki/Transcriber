import argparse
import logging
import random

import requests
import soundfile as sf


def main():
    parser = argparse.ArgumentParser(description="Test code for connecting main_server")
    parser.add_argument(
        "--endpoint", metavar="URL", action="store",
        dest="endpoint", required=True)
    opt = parser.parse_args()

    session_id = "".join(["%02x" % random.randint(0, 255) for _ in range(8)])
    logging.info("session_id = %s" % session_id)


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s: %(name)s:%(funcName)s:%(lineno)d %(levelname)s: %(message)s', level=logging.INFO)
    main()
