import os
import glob
import logging
import threading as th
import concurrent.futures


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


class AsyncCallFuture:
    def __init__(self, timeout: float):
        self.__semaphore = th.Semaphore(0)
        self.__result = None
        self.__timeout = timeout

    def set_result(self, result):
        self.__result = result
        self.__semaphore.release()

    def wait_result(self, on_timeout=None):
        if self.__semaphore.acquire(timeout=self.__timeout):
            return self.__result
        return on_timeout

    def cancel(self, result=None):
        self.set_result(result)


_async_call_executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)


def _async_call_caller(f: AsyncCallFuture, process_callable, args, kwargs):
    f.set_result(process_callable(*args, **kwargs))


def async_call(process_callable, *args, timeout=90.0, **kwargs) -> AsyncCallFuture:
    f = AsyncCallFuture(timeout)
    _async_call_executor.submit(_async_call_caller, f, process_callable, args, kwargs)
    return f
