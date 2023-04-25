import hashlib
import os
from contextlib import AbstractContextManager
from functools import wraps
from typing import List


def compute_md5_hash_from_bytes(input_bytes: bytes) -> str:
    """Compute the MD5 hash of a byte array."""
    return str(hashlib.md5(input_bytes).hexdigest())


class chdir(AbstractContextManager):  # noqa: N801
    """Non thread-safe context manager to change the current working directory."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._old_cwd: List[str] = []

    def __enter__(self) -> None:
        self._old_cwd.append(os.getcwd())
        os.chdir(self.path)

    def __exit__(self, *excinfo) -> None:  # type: ignore
        os.chdir(self._old_cwd.pop())


# https://igeorgiev.eu/python/design-patterns/python-singleton-pattern-decorator/
def singleton(orig_cls):
    orig_new = orig_cls.__new__
    instance = None

    @wraps(orig_cls.__new__)
    def __new__(cls, *args, **kwargs):
        nonlocal instance
        if instance is None:
            instance = orig_new(cls)
            cls.__init__(instance, *args, **kwargs)
        return instance

    orig_cls.__new__ = __new__

    return orig_cls
