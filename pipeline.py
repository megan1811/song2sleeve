import sys
import time
from faster_whisper import WhisperModel
import torchaudio
import torch
from pathlib import Path
from demucs.pretrained import get_model
from demucs.apply import apply_model
from torchaudio.transforms import Resample
import shutil
import boto3
import json


import librosa
import numpy as np
import boto3
import json
from pathlib import Path
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import librosa
import requests
import csv

import boto3
import json
import base64
from PIL import Image
from io import BytesIO

yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

def get_yamnet_class_names():
    """
    Downloads and parses the YAMNet class label CSV file.

    This function fetches the official YAMNet class map from the TensorFlow GitHub repository
    and extracts the display names of all 527 AudioSet sound event classes.

    Returns:
        list[str]: A list of sound class names used by YAMNet.
    """
    url = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
    response = requests.get(url)
    lines = response.text.splitlines()
    reader = csv.reader(lines)
    next(reader)  # Skip header
    return [row[2] for row in reader]

def extract_instrument_tags(audio_path: Path, verbose: bool = False) -> str:
    """Extract instrument tags from instrumental audio, using yamnet.

    Args:
        audio_path (Path): path to instrimental audio.
        verbose (bool, optional): Logs metrics if set to True. Defaults to False.

    Returns:
        str: list of predicted tags along with their confidence scores.
    """
    # mono signal needed for yamnet
    waveform, _ = librosa.load(audio_path, sr=16000, mono=True)

    # Run inference
    scores, _, _ = yamnet_model(waveform)
    class_names = get_yamnet_class_names()
    # classes to skip if present in final prediction
    skip_class_names = ["Silence", "Music", "Musical instrument", "Jingle Bell", "Speech"]

    # Max scores across time and get top predictions
    start = time.time()
    max_scores = tf.reduce_max(scores, axis=0).numpy()
    inference_time = time.time() - start 
    
    top_indices = np.argsort(max_scores)[::-1][:10]
    tags = [(class_names[i], max_scores[i])  for i in top_indices if class_names[i] not in skip_class_names and max_scores[i] > 0.3]
    
    if verbose:
        print("##INSTRUMENTAL TAGS")
        print(f"tags: {tags}")  
        print(f"Inference time: {inference_time:.2f}s")

    return ", ".join([f"{cl}: {conf}"for cl, conf in tags])

def extract_audio_features(audio_path: Path, verbose: bool = False) -> tuple[float, float]:
    """Extract audio features from instrimental audio file. Use librosa library.

    Args:
        audio_path (Path): path to instrimental audio.
        verbose (bool, optional): Logs metrics if set to True. Defaults to False.

    Returns:
        tuple[float, float]: tempo, spectral centroid mean
    """
    y, sr = librosa.load(audio_path, sr=None)

    # Tempo (BPM)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    # Spectral centroid (average over time)
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_centroid_mean = np.mean(spectral_centroids)

    if verbose:
        print("##EXTRACTED AUDIO FEATURES")
        print(f"spectral_centroid_mean: {spectral_centroid_mean}")  
        print(f"tempo: {tempo[0]}\n")  
          
    return round(tempo[0], 2), round(spectral_centroid_mean, 2)
    


def vocals_to_transcript(vocals_path: Path, verbose: bool = False) -> str:
    """Function that takes path to vocal audio, and transcribes the lyrics using 
    faster whisper, which is a reimplementation of OpenAI transcript 
    https://github.com/SYSTRAN/faster-whisper.

    Args:
        vocals_path (Path): Path to split vocal audio.
        verbose (bool, optional): Logs metrics if set to True. Defaults to False.

    Returns:
        str: transcribed lyrics.
    """
    # tiny model size for faster inference
    #TODO might not be on cuda later
    model = WhisperModel("medium", device="cpu", compute_type="int8")
    
    start = time.time()
    segments, _ = model.transcribe(vocals_path)
    inference_time = time.time() - start
    transcript = " ".join([seg.text for seg in segments])
    
    if verbose:
        print("##FASTER WHISPER")
        print(f"Inference time: {inference_time:.2f}s")
        print(f"predicted transcript {transcript}\n")
    
    return transcript

def split_audio_2stems(audio_path: str, output_dir: Path, verbose: bool = False) -> tuple[Path, Path]:
    """Separates a stereo audio file into vocals and instrumental using Demucs.
    Saves the results as WAV files to the given output directory.

    Args:
        audio_path (str): Local path to audio file.
        output_dir (Path): tmp output folder.
        verbose (bool, optional): Logs metrics if set to True. Defaults to False.

    Returns:
        tuple(Path, Path): path to split vocal and instrumental .wav files.
    """
    # FB htdemucs for splitting audio into intruments / vocals
    model = get_model(name="htdemucs")

    # Load audio
    waveform, sr = torchaudio.load(audio_path)

    # Ensure stereo & resample if needed
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)

    if sr != model.samplerate:
        resampler = Resample(orig_freq=sr, new_freq=model.samplerate)
        waveform = resampler(waveform)
        sr = model.samplerate

    # 1 min of waveform for testing purposes
    # TODO: rm line in production
    waveform = waveform[:, sr*30:sr*60]
    
    # Add batch dimension
    waveform = waveform.unsqueeze(0)  # (1, 2, time)

    start = time.time()
    with torch.no_grad():
        sources = apply_model(model, waveform, split=True, overlap=0.25)[0]
    inference_time = time.time() - start
    
    # sources = [drums, bass, other, vocals]
    vocals = sources[3]
    instrumental = sum([sources[0], sources[1], sources[2]])

    # Save files
    separated_dir = output_dir / "separated"
    separated_dir.mkdir(parents=True, exist_ok=True)
    vocals_path = separated_dir / "vocals.wav"
    instrumental_path = separated_dir / "instrumental.wav"
    torchaudio.save(str(vocals_path), vocals, sr)
    torchaudio.save(str(instrumental_path), instrumental, sr)

    if verbose: 
        print("##DEMUCS SEPERATOR")
        print(f"Inference time: {inference_time:.2f}s")
        print(f"Saved vocals to {vocals_path}")
        print(f"Saved instrumental to {instrumental_path}\n")
        

    return vocals_path, instrumental_path



def generate_album_cover_prompt(lyrics, tempo_bpm, spectral_centroid, instrument_tags, verbose=False):
    """
    Generates a short, image-generation prompt for an album cover using Claude Sonnet,
    based on lyrics and audio features.

    Args:
        lyrics (str): Extracted lyrics from the track.
        tempo_bpm (float): Estimated tempo in BPM.
        spectral_centroid (float): Average spectral centroid (Hz).
        instrument_tags (list): Detected instrument tags (e.g., from YAMNet).
        verbose (bool, optional): If True, prints the Claude API response. Defaults to True.

    Returns:
        str: text-to-image generation prompt for album cover.
    """

    prompt = f"""
    From the extracted lyrics and audio features under, generate a text-to-image prompt for an AI image generator. 
    The goal is to create a visually expressive album cover that reflects the song's emotional tone, atmosphere, and underlying themes.
    If lyrics are sparse or unclear, prioritize the music.
    Respond with only the final prompt, no preamble, and preferably in 2-3 sentences max.

    Lyrics:
    \"\"\"{lyrics}\"\"\"

    Audio Features:
    - Tempo: {tempo_bpm} BPM
    - Timbre (Spectral Centroid): {spectral_centroid} Hz
    - Instrument Tags: {instrument_tags}
    """

    bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

    body = {
        "messages": [{"role": "user", "content": prompt}],
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 100,
    }

    start = time.time()
    response = bedrock.invoke_model(
        modelId="anthropic.claude-3-sonnet-20240229-v1:0",
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body)
    )
    inference_time = time.time() - start
    response_body = json.loads(response["body"].read())

    prompt = response_body["content"][0]["text"]
    if verbose:
        print("##IMAGE PROMPT GENERATION")
        print(f"Inference time: {inference_time:.2f}s")
        print(f"Generated prompt {prompt}\n")
        
    return prompt

def generate_album_cover(prompt: str, output_dir: Path, verbose: bool = False) -> Image:
    """
    Generates an album cover image from a text prompt using AWS Bedrock's Stable Diffusion XL.

    Sends the prompt to the image generation model and saves the resulting image
    to the specified output directory. Optionally displays inference time and the image.

    Args:
        prompt (str): The text prompt describing the album cover.
        output_dir (Path): Directory where the generated image will be saved.
        verbose (bool, optional): If True, prints generation details and shows the image. Defaults to False.

    Returns:
        PIL.Image.Image: The generated album cover image.
    """
    
    bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
    image_gen_body = {
        "text_prompts": [{"text": prompt}],
        "cfg_scale": 8,  # Prompt adherence (7–10 is good)
        "steps": 20,     # More = better detail, slower
    }

    start = time.time()
    response = bedrock.invoke_model(
        modelId="stability.stable-diffusion-xl-v1",
        contentType="application/json",
        accept="application/json",
        body=json.dumps(image_gen_body)
    )
    inference_time = time.time() - start


    result = json.loads(response["body"].read())
    image_bytes = result['artifacts'][0]['base64']  

    # Decode and display
    image = Image.open(BytesIO(base64.b64decode(image_bytes)))
    # image.show()

    # Or save to disk
    image.save(output_dir / "generated_img.png")
    if verbose:
        print("##IMAGE GENERATION")
        print(f"Inference time: {inference_time:.2f}s")
        image.show()
        
    return image

def run(audio_path: str, output_dir: Path):
    start_total = time.time()

    vocals_path, instrumental_path = split_audio_2stems(audio_path, output_dir)
    lyrics = vocals_to_transcript(vocals_path)
    tempo, spectral_centroid_mean = extract_audio_features(instrumental_path)
    instrument_tags = extract_instrument_tags(instrumental_path)

    prompt = generate_album_cover_prompt(lyrics, tempo, spectral_centroid_mean, instrument_tags)
    image = generate_album_cover(prompt, output_dir)

    return {
        "lyrics": lyrics,
        "tempo": tempo,
        "spectral_centroid": spectral_centroid_mean,
        "tags": instrument_tags,
        "image": image,
        "prompt": prompt,
        "inference_time": round(time.time() - start_total, 2)
    }
    
    
def main(audio_path: str):
    output_dir = Path("tmp")
    # Remove the folder if it exists
    if output_dir.exists() and output_dir.is_dir():
        shutil.rmtree(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    out = run(audio_path, output_dir)
    print(out["lyrics"])
    
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py path_to_audio.mp3")
        sys.exit(1)
    main(sys.argv[1])    