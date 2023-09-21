import sys
import time
import dataclasses
import numpy as np

enable = False


@dataclasses.dataclass
class _Result:
    tm_sum1: float = 0
    tm_sum2: float = 0
    tm_min: float = sys.float_info.max
    tm_max: float = 0
    sample_count: int = 0


_result: dict[str, _Result] = {}


class Measure:
    def __init__(self, key: str):
        self.__key = key
        self.__tm0 = 0.0

    def __enter__(self):
        if enable:
            self.__tm0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not enable:
            return False

        global _result
        tm = time.perf_counter() - self.__tm0
        if self.__key not in _result:
            _result[self.__key] = _Result()
        r = _result[self.__key]
        r.tm_sum1 += tm
        r.tm_sum2 += tm * tm
        r.tm_min = min(r.tm_min, tm)
        r.tm_max = max(r.tm_max, tm)
        r.sample_count += 1
        return False


def dump_result() -> str:
    out = ""
    for k in sorted(_result.keys()):
        r = _result[k]
        ave = r.tm_sum1 / r.sample_count
        out += "%s: %d calls, average %fs, stddev %fs, min %fs, max %fs\n" % (
            k, r.sample_count, ave, np.sqrt(r.tm_sum2 / r.sample_count - ave * ave), r.tm_min, r.tm_max
        )
    return out
