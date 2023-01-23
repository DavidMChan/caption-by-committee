from abc import ABC, abstractmethod
from typing import List, Optional

from PIL.Image import Image


class CaptionEngine(ABC):
    @abstractmethod
    def __init__(self, device: Optional[str] = None):
        raise NotImplementedError()

    @abstractmethod
    def __call__(self, raw_image: Image, n_captions: int = 1, temperature: Optional[float] = None) -> List[str]:
        # Takes an RGB image and returns a list of captions
        raise NotImplementedError()

    @abstractmethod
    def get_baseline_caption(self, raw_image: Image) -> str:
        raise NotImplementedError()
