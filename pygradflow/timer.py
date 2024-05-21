import time


class SimpleTimer:
    def __init__(self):
        self.start = time.time()

    def elapsed(self):
        return time.time() - self.start

    def reset(self):
        self.start = time.time()


class Timer(SimpleTimer):
    def __init__(self, time_limit: float):
        super().__init__()
        self.time_limit = time_limit

    def remaining(self):
        return self.time_limit - self.elapsed()

    def reached_time_limit(self):
        return self.remaining() <= 0.0
