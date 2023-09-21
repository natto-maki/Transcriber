import os
import glob
import logging


def _make_file_name_for_safe_write(file_name):
    return os.path.abspath(file_name), os.path.abspath(file_name + "_$1"), os.path.abspath(file_name + "_$2")


def _safe_remove(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def _safe_rename(src, dest):
    try:
        os.rename(src, dest)
    except OSError as ex:
        logging.critical(
            "File operation failed in unexpected way." +
            " Force exit due to a bug in the application or an error in the system.", exc_info=ex)
        quit(1)


def recover_file(file_name):
    f0, f1, f2 = _make_file_name_for_safe_write(file_name)
    _safe_remove(f1)
    if os.path.isfile(f2):
        _safe_remove(f0)
        _safe_rename(f2, f0)


def recover_files(dir_name):
    for file in glob.glob(os.path.join(os.path.abspath(dir_name), "*")):
        if file.endswith("_$1"):
            _safe_remove(file)
        elif file.endswith("_$2"):
            recover_file(file[:-3])


class SafeWrite:
    def __init__(self, file_name, mode="wb"):
        f0, f1, f2 = _make_file_name_for_safe_write(file_name)
        self._file_name0 = f0
        self._file_name1 = f1
        self._file_name2 = f2
        self._mode = mode
        self.stream = None
        recover_file(file_name)

    def __enter__(self):
        self.stream = open(self._file_name1, self._mode)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            _safe_remove(self._file_name1)
            return True

        try:
            self.stream.close()
            os.rename(self._file_name1, self._file_name2)
        except Exception as ex:
            _safe_remove(self._file_name1)
            _safe_remove(self._file_name2)
            raise ex

        _safe_remove(self._file_name0)
        _safe_rename(self._file_name2, self._file_name0)
        return False
