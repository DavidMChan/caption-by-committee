import io
import logging
import subprocess
import tempfile
import time
from typing import Any, Optional, Tuple, Union

import numpy as np
import requests
import soundfile as sf
import whisper
from whisper.utils import format_timestamp

from cbc.utils.python import compute_md5_hash_from_bytes

NO_AUDIO_VTT = """WEBVTT

1
00:00:00.000 --> 01:00:00.000
NO TRANSCRIPT AVAILABLE

"""


def gcp_transcribe(path: str) -> str:
    try:
        from google.cloud import speech
    except ImportError:
        raise ImportError("Please install google-cloud-speech to use this feature.")

    logging.debug("Loading audio...")
    inputs = load_audio_to_wav_bytes(path)
    if inputs is None:
        return ""

    logging.debug("Transcribing audio...")
    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(content=inputs)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
        enable_automatic_punctuation=True,
    )

    logging.debug("Sending audio request...")
    response = client.recognize(config=config, audio=audio)
    transcript = " ".join([result.alternatives[0].transcript for result in response.results])

    logging.debug("Transcription complete: {transcript}")

    return transcript


def aws_transcribe(path: Union[str, bytes]) -> Tuple[str, str]:
    try:
        import boto3
    except ImportError:
        raise ImportError("Please install boto3 to use this feature.")

    logging.debug("Loading audio...")
    try:
        inputs = path if isinstance(path, bytes) else load_audio_to_wav_bytes(path)
        if inputs is None:
            return "", NO_AUDIO_VTT
    except RuntimeError:
        return "", NO_AUDIO_VTT

    # Create a tempfile, upload it to S3, and then delete it
    with tempfile.NamedTemporaryFile(suffix=".wav") as temp:
        temp.write(inputs)
        temp.flush()
        s3 = boto3.client("s3")
        s3.upload_file(temp.name, "caption-by-committee", "temp.wav")

    logging.debug("Transcribing audio...")

    # Create a tempfile
    transcribe_client = boto3.client("transcribe")
    job_name = f"cbc-{compute_md5_hash_from_bytes(inputs)}"
    _create_aws_transcribe_job_if_necessary(transcribe_client, job_name)

    while (status := transcribe_client.get_transcription_job(TranscriptionJobName=job_name))["TranscriptionJob"][
        "TranscriptionJobStatus"
    ] not in (
        "COMPLETED",
        "FAILED",
    ):
        logging.debug("Transcription not yet complete... sleeping for 5 seconds.")
        time.sleep(5)

    # Delete the file from S3
    try:
        s3.delete_object(Bucket="caption-by-committee", Key="temp.wav")
    except s3.exceptions.NoSuchKey:
        # The file doesn't exist, so just continue
        pass

    # Get the transcript
    transcript, vtt = _extract_transcript_and_vtt_from_aws_job_result(status)

    # Remove the transcript
    # transcribe_client.delete_transcription_job(TranscriptionJobName=job_name)

    logging.debug(f"Transcription complete: {transcript}")

    return transcript, vtt


def _extract_transcript_and_vtt_from_aws_job_result(status: Any) -> Tuple[str, str]:
    transcript_url = status["TranscriptionJob"]["Transcript"]["TranscriptFileUri"]
    logging.debug(f"Transcript URI: {transcript_url}")

    # Fetch the transcript with requests
    req = requests.get(transcript_url)
    transcript = req.json()["results"]["transcripts"][0]["transcript"]

    # Get the VTT file
    try:
        vtt_url = status["TranscriptionJob"]["Subtitles"]["SubtitleFileUris"][0]
        logging.debug(f"VTT URI: {vtt_url}")
        req = requests.get(vtt_url)
        vtt = req.text
        logging.debug(f"VTT:\n{vtt}")
    except IndexError:
        # No VTT file available, so just return a dummy one
        vtt = NO_AUDIO_VTT
    return transcript, vtt


def _create_aws_transcribe_job_if_necessary(transcribe_client: Any, job_name: str) -> None:
    try:
        transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
        logging.debug("Transcription already exists. Skipping...")
    except transcribe_client.exceptions.BadRequestException:
        # The job doesn't exist, so create it
        job_uri = "s3://caption-by-committee/temp.wav"
        transcribe_client.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={"MediaFileUri": job_uri},
            MediaFormat="wav",
            LanguageCode="en-US",
            Subtitles={"Formats": ["vtt"], "OutputStartIndex": 1},
        )
        logging.debug("Waiting for transcription to complete...")
        transcribe_client.get_transcription_job(TranscriptionJobName=job_name)


def whisper_transcribe(path: Union[str, bytes], model: str = "large-v2") -> Tuple[str, str]:
    logging.debug("Loading audio...")

    if isinstance(path, str):
        inputs = load_audio_to_wav_bytes(path)
        if inputs is None:
            return "", NO_AUDIO_VTT
        data, _ = sf.read(io.BytesIO(inputs))
    else:
        data, _ = sf.read(io.BytesIO(path))

    data = data.mean(axis=-1) if len(data.shape) == 2 else data  # noqa: PLR2004
    data = data.astype(np.float32)

    logging.debug("Loading model...")

    model = whisper.load_model(model)

    logging.debug("Transcribing audio...")

    result = whisper.transcribe(audio=data, model=model, language="en")

    vtt = ["WEBVTT\n"]
    for segment in result["segments"]:
        vtt.append(f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n")
        vtt[-1] = f"{vtt[-1]}{segment['text'].strip().replace('-->', '->')}\n"

    transcript = result["text"]
    vttText = "\n".join(vtt)

    logging.debug(f"Transcription complete: {transcript}")

    return transcript, vttText


def load_audio_to_wav_bytes(path: str) -> Optional[bytes]:
    # Use FFMPG and pipe the output to stdout, then read it
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-i", path, "-f", "wav", "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = ffmpeg.communicate()
    return None if ffmpeg.returncode != 0 else stdout
