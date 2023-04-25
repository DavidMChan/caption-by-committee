import json
import logging
from typing import Dict, Optional

import numpy as np
import openai
from PIL.Image import Image

from cbc.lm import ChatGPT, OpenAI
from cbc.plugins.base import ImagePlugin

try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None

# PROMPT_BODY = "Be sure to incorporate the potential text into the summary if possible. "
PROMPT_BODY = "There is potential text that appears in the image. Try to incorporate the potential text into the summary. Do not list them as potential text, and instead, try to organize them into the sentences."


def _correct_ocr_with_chatgpt(ocr_token_list, retries=3):
    PROMPT = """Let's think step by step. A very bad OCR algorithm generated the tokens: {tokens}
    What do these tokens really say? Keep in mind that the tokens may not be in english, and may be out of order. Be liberal in your guessing, talk through your answers, and give the final output in JSON format, with a key "tokens" containing a (de-duplicated) list IN ENGLISH of strings corrected for spelling, capitalization, and content, and a key "language" giving the original language of the tokens."""

    prompt = PROMPT.format(tokens=ocr_token_list)
    for _ in range(retries):
        cp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=1024,
            n=1,
        )  # type: ignore
        OpenAI.USAGE += int(cp.usage.total_tokens) * ChatGPT.COST_PER_TOKEN
        content = None
        try:
            # Extract anything between the first pair of open and closing braces
            content = "{" + cp.choices[0].message.content.split("{", 1)[1].rsplit("}", 1)[0] + "}"
            content = content.strip().replace("\n", "")
            # Remove any unicode
            content = content.encode("ascii", "ignore").decode("ascii")
            logging.debug(f"ChatGPT output: {content}")
            json_content = json.loads(content)
            return (json_content["tokens"], json_content.get("language", "English"))
        except Exception as ex:
            logging.warning(ex)
            logging.warning(f"Tried to parse {content}")
            logging.warning(cp.choices[0].message.content)
            continue

    raise RuntimeError(f"Could not correct OCR tokens: {ocr_token_list}")


class OcrPlugin(ImagePlugin):
    def __init__(self, device: Optional[str] = None):
        super().__init__(device)

        if PaddleOCR is None:
            raise RuntimeError("PaddleOCR not installed. Please run `pip install paddleocr`.")

        self.ocr = PaddleOCR(use_angle_cls=True, lang="en")

    def __call__(self, raw_image: Image) -> Dict[str, str]:
        image_np = np.array(raw_image)
        results = self.ocr(image_np, cls=True)

        logging.debug(f"OCR results: {results}")

        # Extract OCR tokens
        ocr_before_correction = [r[0] for r in results[1]]
        # Correct OCR tokens with ChatGPT
        try:
            if len(ocr_before_correction) > 0:
                ocr_after_correction, language = _correct_ocr_with_chatgpt(ocr_before_correction)
            else:
                ocr_after_correction, language = [], "English"
        except Exception as ex:
            logging.warning(ex)
            ocr_after_correction, language = ocr_before_correction, "English"
        # Remove duplicates
        ocr_after_correction = [o.replace(".", " ").strip() for o in set(ocr_after_correction)]
        logging.debug(f"OCR after ChatGPT correction: {ocr_after_correction}")

        # Remove some common failure cases
        if len(ocr_after_correction) == 0:
            return {"prompt_body": "", "image_info": ""}
        elif ["the"] == ocr_after_correction:
            return {"prompt_body": "", "image_info": ""}
        elif (
            len(set(ocr_after_correction).intersection(set("the quick brown fox jumps over the lazy dog".split()))) > 7
        ):
            return {"prompt_body": "", "image_info": ""}
        elif len(set(ocr_after_correction).intersection(set("hello world".split()))) > 1:
            return {"prompt_body": "", "image_info": ""}

        image_info = f"Potential text in the image: {ocr_after_correction}\nText language: {language}"

        return {"prompt_body": PROMPT_BODY, "image_info": image_info}


class OcrNoCorrectionPlugin(OcrPlugin):
    def __call__(self, raw_image: Image) -> Dict[str, str]:
        image_np = np.array(raw_image)
        results = self.ocr(image_np, cls=True)

        # Extract OCR tokens
        ocr_tokens = [r[0] for r in results[1]]

        image_info = f"Potential text in the image: {ocr_tokens}"
        return {"prompt_body": PROMPT_BODY, "image_info": image_info}
