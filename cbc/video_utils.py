from typing import List, Optional

import cv2
from PIL import Image


def load_video(video_path: str, frames: Optional[int] = None) -> List[Image.Image]:
    """
    Load a video from a path.
    """
    cap = cv2.VideoCapture(video_path)
    try:
        all_frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to RGB PIL Image
            all_frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    finally:
        cap.release()

    if frames is None:
        return all_frames

    # Subsample {frames} equally
    return [all_frames[i] for i in range(0, len(all_frames), len(all_frames) // frames)]
