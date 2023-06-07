from typing import Dict, Optional

from PIL.Image import Image
try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None
import numpy as np
import json
import openai
import math
from cbc.plugins.base import ImagePlugin
from cbc.lm import OpenAI, ChatGPT
from sklearn.cluster import DBSCAN

PROMPT_BODY = "There may be potential text appearing in the image. If the potential text makes sense, try to incorporate it into the summary. ONLY include text from the list of potential text. In the summary, you MUST NOT INCLUDE 'potential text'."
CONFIDENCE_THRESHOLD = 0.9

def _correct_ocr_with_chatgpt(ocr_token_list, retries=3):
    PROMPT = """Let's think step by step. A very bad OCR algorithm generated the tokens: {tokens}
    What do these tokens really say? Be liberal in your guessing, talk through your answers, and give the final output in JSON format, with a single key "tokens" containing a list of corrected string for spelling, capitalization, and content."""

    prompt = PROMPT.format(tokens=ocr_token_list)
    for _ in range(retries):
        cp = openai.ChatCompletion.create(
                    model='gpt-3.5-turbo',
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
            content = '{' + cp.choices[0].message.content.split('{', 1)[1].rsplit('}', 1)[0] + '}'
            content = content.strip().replace('\n', '')
            # Remove any unicode 
            content = content.encode('ascii', 'ignore').decode('ascii')
            return json.loads(content)['tokens']
        except Exception as ex:
            print(ex)
            print('Tried to parse:', content)
            print(cp.choices[0].message.content)
            continue

    raise RuntimeError(f'Could not correct OCR tokens: {ocr_token_list}')

# Sort OCR tokens based on DBSCAN clustering
def cluster_and_sort_ocr_tokens(ocr_tokens, eps, min_samples=1):
    # Extract the center points of the bounding boxes
    center_points = [
        np.mean(np.array(token[0]), axis=0)
        for token in ocr_tokens
    ]

    # Perform clustering using DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(center_points)
    labels = clustering.labels_
    
    # Organize tokens into clusters
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(ocr_tokens[i])

    # Sort clusters by their mean y-coordinate (top to bottom) and then x-coordinate (left to right)
    sorted_cluster_keys = sorted(
        clusters.keys(),
        key=lambda k: (
            np.mean([np.mean(np.array(token[0]), axis=0)[1] for token in clusters[k]]),
            np.mean([np.mean(np.array(token[0]), axis=0)[0] for token in clusters[k]])
        )
    )
    
    # Sort tokens within each cluster by their y-coordinate (top to bottom) and then x-coordinate (left to right)
    sorted_ocr_tokens = []
    for key in sorted_cluster_keys:
        sorted_cluster = sorted(
            clusters[key],
            key=lambda token: (
                np.mean(np.array(token[0]), axis=0)[1],
                np.mean(np.array(token[0]), axis=0)[0]
            )
        )
        sorted_ocr_tokens.extend(sorted_cluster)

    return sorted_ocr_tokens

# Get the bounding box area of an OCR token
def get_ocr_area(coordinates):
    x1, y1 = coordinates[0]
    x2, y2 = coordinates[1]
    x3, y3 = coordinates[2]
    
    length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    width = math.sqrt((x3 - x2)**2 + (y3 - y2)**2)
    
    return length * width

# Set the area threshold to 0.2% of the image area
def get_area_threshold(image):
    image_width = len(image)
    image_length = len(image[0])
    return image_width * image_length / 500

# Set the cluster threshold to 1/8 of the image diagonal
def get_cluster_threshold(image):
    image_width = len(image)
    image_length = len(image[0])
    return np.sqrt(image_width**2 + image_length**2) / 8

# Filter OCR tokens by area and confidence, then sort based on clustering
def filter_and_sort_ocr(image, results):
    area_threshold = get_area_threshold(image)
    cluster_threshold = get_cluster_threshold(image)
    
    # Filter out OCR tokens with bounding box area <= area threshold or confidence <= confidence threshold
    ocr_filtered = []
    for i in range(len(results[0])):
        cur_ocr = results[0][i]
        coordinates = cur_ocr[0]
        ocr_area = get_ocr_area(coordinates)
        confidence = cur_ocr[1][1]
        if ocr_area > area_threshold and confidence > CONFIDENCE_THRESHOLD:
            ocr_filtered.append(cur_ocr)
            
    ocr_filtered_and_sorted = []
            
    if len(ocr_filtered) > 0:
        # Sort OCR tokens based on clustering
        ocr_filtered_and_sorted = cluster_and_sort_ocr_tokens(ocr_filtered, cluster_threshold)
        ocr_filtered_and_sorted = [ocr[1][0] for ocr in ocr_filtered_and_sorted]
    
    return ocr_filtered_and_sorted

class OcrPlugin(ImagePlugin):
    def __init__(self, device: Optional[str] = None):
        super().__init__(device)

        if PaddleOCR is None:
            raise RuntimeError("PaddleOCR not installed. Please run `pip install paddleocr`.")

        self.ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=True)
        
    def __call__(self, raw_image: Image) -> Dict[str, str]:
        image = np.array(raw_image)
        results = self.ocr.ocr(image, cls=True, det=True, rec=True)
        
        ocr_filtered_and_sorted = filter_and_sort_ocr(image, results)
        ocr_after_correction = []
        if len(ocr_filtered_and_sorted) > 0:
            try:
                # Correct OCR tokens with ChatGPT
                ocr_after_correction = _correct_ocr_with_chatgpt(ocr_filtered_and_sorted)
            except Exception:
                ocr_after_correction = ocr_filtered_and_sorted

        image_info = f"Potential text in the image: {ocr_after_correction}"
        return {"prompt_body": PROMPT_BODY, "image_info": image_info}
    
class OcrNoCorrectionPlugin(OcrPlugin):
    def __call__(self, raw_image: Image) -> Dict[str, str]:
        image = np.array(raw_image)
        results = self.ocr.ocr(image, cls=True, det=True, rec=True)
            
        ocr_filtered_and_sorted = filter_and_sort_ocr(image, results)
        
        image_info = f"Potential text in the image: {ocr_filtered_and_sorted}"
        return {"prompt_body": PROMPT_BODY, "image_info": image_info}
    