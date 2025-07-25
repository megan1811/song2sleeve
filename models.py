import csv
import time
import requests
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Union, List

import librosa
import tensorflow as tf
import tensorflow_hub as hub

from faster_whisper import WhisperModel

class LyricsTranscriber():
    """
    Transcribes vocals audio into text using a Faster Whisper model.
    
    Attributes:
        model: The Faster Whisper model loaded once at class level.
    """
    # TODO: not only on cpu or size base
    model = WhisperModel("base", device="cpu", compute_type="int8")

    def __init__(self, verbose: bool):
        """
        Initialize the LyricsTranscriber.

        Args:
            verbose (bool): Whether to print detailed progress information.
        """
        self.verbose = verbose
        self._metrics = {}
        
    @property
    def metrics(self) -> Dict[str, float]:
        """
        Metrics from the most recent transcription run.

        Returns:
            Dict[str, float]: A dictionary of metrics (e.g., inference time in seconds).
        """
        return self._metrics

    def _record_metrics(self, metrics):
        """
        Record metrics from a transcription run.

        Args:
            metrics (Dict[str, float]): Metrics to store, typically inference time.
        """
        self._metrics = metrics
        
    def infer(self, vocals_path: Union[str, Path]) -> str:
        """
        Transcribe the given audio file (vocals) into text.

        Args:
            vocals_path (Union[str, Path]): Path to the audio file containing vocals.

        Returns:
            str: The transcribed lyrics as a single concatenated string.
        """
        start = time.time()
        segments, _ = self.model.transcribe(vocals_path)
        segments = list(segments) 
        inference_time = time.time() - start
        
        # Combine transcribed segments into a single string
        lyrics_pred = " ".join([seg.text for seg in segments])

        if self.verbose:
            print("## LyricsTranscriber")
            print(f"Inference time: {inference_time:.2f}s")
            print(f"predicted lyrics {lyrics_pred}\n")

        self._record_metrics({"inference": inference_time, "segments": len(segments)})
        
        return lyrics_pred
    
    
class InstrumentalTagger():
    """
    Extracts sound event or instrument tags from audio using the YAMNet model.
    
    Attributes:
        model: The TensorFlow Hub YAMNet model loaded once at class level.
        class_names: List of class names fetched from YAMNet's official CSV.
    """
    @staticmethod
    def _fetch_yamnet_class_names() -> List[str]:
        """
        Download and parse the YAMNet class label CSV.

        Returns:
            List[str]: A list of all AudioSet sound class names used by YAMNet.
        """
        url = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        reader = csv.reader(response.text.splitlines())
        next(reader)
        return [row[2] for row in reader]

    # Class-level resources
    model = hub.load('https://tfhub.dev/google/yamnet/1')
    class_names = _fetch_yamnet_class_names()
    
    def __init__(self, verbose: bool):
        """
        Initialize the InstrumentalTagger.

        Args:
            verbose (bool): If True, print debug information and metrics.
        """
        self.exclude_class_names = ["Silence", "Music", "Musical instrument", "Jingle Bell", "Speech"]
        self.verbose = verbose
        self._metrics = {}
        return
    
    @property
    def metrics(self) -> Dict[str, float]:
        """
        Metrics from the most recent inference.

        Returns:
            Dict[str, float]: Dictionary of metrics, e.g. {"inference": <seconds>}.
        """
        return self._metrics

    def _record_metrics(self, metrics):
        """
        Record metrics from the most recent inference.

        Args:
            metrics (Dict[str, float]): Metrics to store.
        """
        self._metrics = metrics
        
    def infer(
        self,
        audio_path: Union[str, Path],
        top_n: int = 10,
        conf_thresh: float = 0.3
    ) -> List[Tuple[str, float]]:
        """
        Run YAMNet inference on an audio file and return top tags.

        Args:
            audio_path (Union[str, Path]): Path to the audio file (mono is expected).
            top_n (int, optional): Maximum number of tags to return. Defaults to 10.
            conf_thresh (float, optional): Confidence threshold to filter tags. Defaults to 0.3.

        Returns:
            List[Tuple[str, float]]: A list of (class_name, confidence_score) tuples
            sorted by confidence in descending order.
        """
        # mono signal needed for yamnet
        waveform, _ = librosa.load(audio_path, sr=16000, mono=True)
            
        # run inference
        start = time.time()
        scores, _, _ = self.model(waveform)
        inference_time = time.time() - start 

        # Max scores across time and get top predictions as opposed to mean
        # over time
        max_scores = tf.reduce_max(scores, axis=0).numpy()
        
        top_indices = np.argsort(max_scores)[::-1][:top_n]
        tags = [(self.class_names[i], max_scores[i])  for i in top_indices if self.class_names[i] not in self.exclude_class_names and max_scores[i] > conf_thresh]

        if self.verbose:
            print("## InstrumentalTagger")
            print(f"tags: {tags}")  
            print(f"Inference time: {inference_time:.2f}s")
        
        self._record_metrics({"inference" :inference_time})
        
        return tags
    



    
    
