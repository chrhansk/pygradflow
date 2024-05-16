from collections import defaultdict
from enum import Enum, auto


class CallbackType(Enum):
    ComputedStep = auto()


class CallbackHandle:
    def __init__(self, callback_type: CallbackType, callback):
        self.callback_type = callback_type
        self.callback = callback

    def __call__(self, *args, **kwargs):
        return self.callback(*args, **kwargs)

    def __repr__(self) -> str:
        return "CallbackHandle(callback_type={0})".format(self.callback_type)


class Callbacks:
    def __init__(self):
        self._callbacks = defaultdict(lambda: list())

    def register(self, callback_type: CallbackType, callback):
        handle = CallbackHandle(callback_type, callback)
        self._callbacks[callback_type].append(handle)
        return handle

    def unregister(self, handle):
        self._callbacks[handle.callback_type].remove(handle)

    def __call__(self, callback_type, *args, **kwargs):
        for handle in self._callbacks[callback_type]:
            handle(*args, **kwargs)
