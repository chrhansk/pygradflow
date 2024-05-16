import time


class Timer:
    def __init__(self, time_limit: float):
        self.time_limit = time_limit
        self.start = time.time()

    def elapsed(self):
        return time.time() - self.start

    def remaining(self):
        return self.time_limit - self.elapsed()

    def reached_time_limit(self):
        return self.remaining() <= 0.0
