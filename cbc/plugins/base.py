from abc import ABC, abstractmethod
from typing import Dict, Optional

from PIL.Image import Image


class ImagePlugin(ABC):
    def __init__(self, device: Optional[str] = None):
        self._device = device

    @abstractmethod
    def __call__(self, raw_image: Image) -> Dict[str, str]:
        raise NotImplementedError()


class TestPlugin(ImagePlugin):
    def __call__(self, raw_image: Image) -> Dict[str, str]:
        return {"prompt_body": "This is a test plugin.", "image_info": "This is a test plugin."}
