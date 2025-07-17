import librosa
import numpy as np
from pathlib import Path
from typing import List, Tuple

def extract_instrumental_features(audio_path: Path, verbose: bool = False) -> tuple[float, float]:
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


def structure_claude_prompt(lyrics: str, tempo_bpm: float, spectral_centroid: float, instrument_tags: List[Tuple[str, float]]) -> str:
    """Build Claude prompt using extracted features.

    Args:
        lyrics (str): Extracted lyrics from the track.
        tempo_bpm (float): Estimated tempo in BPM.
        spectral_centroid (float): Average spectral centroid (Hz).
        instrument_tags (list): Detected instrument tags (e.g., from YAMNet).

    Returns:
        str: Claude prompt.
    """

    prompt = f"""
        Write a clear, actionable instruction for an AI image generator to create a professional‑grade album cover.
        The instruction must begin with: "Generate a professional‑grade album cover containing..."  
        Base the instruction on the mood and themes suggested by the lyrics and audio features below.  
        If the lyrics are minimal or unclear, emphasize the sound, genre, and instrumentation instead.  
        Include specific visual elements such as setting, objects, color palette, lighting, composition, and artistic style (e.g., surrealism, digital painting, retro vinyl aesthetic).  
        Do not include any song titles, artist names, or extra commentary.  
        Return only the final instruction, in under 5 sentences.

        Lyrics: {lyrics}

        Audio Features:
        - Tempo: {tempo_bpm} BPM
        - Timbre (Spectral Centroid): {spectral_centroid} Hz
        - Instrument Tags: {", ".join([cl for cl, _ in instrument_tags])}
        """
    
    return prompt