from typing import Dict, Type

from .base import ImagePlugin, TestPlugin
from .ocr import OcrPlugin, OcrNoCorrectionPlugin

IMAGE_PLUGINS: Dict[str, Type[ImagePlugin]] = {
    "test": TestPlugin,
    "ocr": OcrPlugin,
    "ocr_no_correction": OcrNoCorrectionPlugin,
}
