import sys

from contextlib import contextmanager
from exception import TimeoutException
from signal import setitimer, ITIMER_REAL, signal, SIGALRM


@contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    setitimer(ITIMER_REAL, seconds)
    signal(SIGALRM, signal_handler)
    try:
        yield
    finally:
        setitimer(ITIMER_REAL, 0)


class Null:
    def write(self, x: str):
        pass

    def flush(self):
        pass


@contextmanager
def no_stdout():
    stdout_bk = sys.stdout
    sys.stdout = Null()
    yield
    sys.stdout = stdout_bk
