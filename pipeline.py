from pathlib import Path
import shutil
import time

from models import LyricsTranscriber, InstrumentalTagger
from utils import extract_instrumental_features, structure_claude_prompt
from client import BedrockClient

class Pipeline():
    """
    A high-level audio-to-image pipeline that:
      1. Transcribes lyrics from the vocal track.
      2. Extracts tempo and timbre features from the instrumental track.
      3. Tags instruments present in the audio.
      4. Generates an image prompt using a language model (Claude Sonnet).
      5. Generates a final image using Stable Diffusion.
    """
    def __init__(self, output_path: str = "tmp", verbose: bool = False):
        """
        Initialize Pipeline.

        Args:
            max_tokens (str, optional): Path to the output directory. Defaults to tmp.
            verbose (bool, optional): Whether to print progress/debug information. Defaults to False.
        """
        self.verbose = verbose
        self.output_dir = Path(output_path)
        
        # Remove existing output directory if present
        if self.output_dir.exists() and self.output_dir.is_dir():
            shutil.rmtree(self.output_dir)
    
        self.output_dir.mkdir(parents=True, exist_ok=True) 
        
        # Initiate models needed for pipeline
        self.transcriber = LyricsTranscriber(self.verbose)
        self.tagger = InstrumentalTagger(self.verbose)
        self.client = BedrockClient(self.verbose)

        return

    
    def run(self, audio_path: Path) -> dict:
        """
        Execute the full pipeline:
          - Transcribe lyrics
          - Extract instrumental features
          - Tag instruments
          - Generate prompt & image
        
        Args:
            audio_path (Path): Path to input audio file.
        Returns:
            dict: Dictionary with pipeline results.
        """
        start = time.time()
        
        lyrics = self.transcriber.infer(audio_path)
        tempo, spectral_centroid_mean = extract_instrumental_features(audio_path, self.verbose)
        tags = self.tagger.infer(audio_path)
        
        claude_prompt = structure_claude_prompt(lyrics, tempo, spectral_centroid_mean, tags)
        sd_prompt = self.client.infer_claude_sonnet(claude_prompt)
        image = self.client.infer_stable_diffusion(sd_prompt, self.output_dir, steps=20)
        
        return {
            "lyrics": lyrics,
            "tempo": tempo,
            "spectral_centroid": spectral_centroid_mean,
            "tags": tags,
            "image": image,
            "claude_prompt": claude_prompt,
            "stable_diffusion_prompt": sd_prompt,
            "pipeline_time": round(time.time() - start, 2) 
        }
        
        
# if __name__ == "__main__":
    # audio_path = "data/techno.wav"
    # pipe = Pipeline(verbose=True)
    # pipe.run(audio_path)
    
