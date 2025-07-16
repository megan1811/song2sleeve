import boto3
import time
import json
from PIL import Image
from io import BytesIO
import base64
from pathlib import Path


class BedrockClient:
    """
    Client for interacting with AWS Bedrock services such as Claude Sonnet and Stable Diffusion XL.
    """
    def __init__(self, verbose: bool = False):
        """
        Initialize the Bedrock client.
        
        Args:
            verbose (bool): If True, print detailed logs and timing information. Defaults to True.
        """
        self.client = boto3.client("bedrock-runtime", region_name="us-east-1")
        self.verbose = verbose
        
    def infer_claude_sonnet(self, prompt: str, max_tokens: int = 100) -> str:
        """
        Generates a short, image-generation prompt for an album cover using Claude Sonnet,
        based on lyrics and audio features.

        Args:
            prompt (str): Claude Sonnet prompt.
            max_tokens (int): Maximum number of tokens the model can return.

        Returns:
            str: text-to-image generation prompt for album cover.
        """
        body = {
            "messages": [{"role": "user", "content": prompt}],
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
        }

        start = time.time()
        response = self.client.invoke_model(
            modelId="anthropic.claude-3-sonnet-20240229-v1:0",
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body)
        )
        inference_time = time.time() - start
        response_body = json.loads(response["body"].read())

        response = response_body["content"][0]["text"].strip()
        if self.verbose:
            print("##IMAGE PROMPT GENERATION")
            print(f"Inference time: {inference_time:.2f}s")
            print(f"Generated prompt {response}\n")
            

        return response
    
    def infer_stable_diffusion(self, prompt: str, output_dir: Path, cfg_scale: int = 8, steps: int = 50) -> Image:
        """
        Generates an album cover image from a text prompt using AWS Bedrock's Stable Diffusion XL.

        Sends the prompt to the image generation model and saves the resulting image
        to the specified output directory. Optionally displays inference time and the image.

        Args:
            prompt (str): The text prompt describing the album cover.
            output_dir (Path): Directory where the generated image will be saved.
            cfg_scale (int, optional): Prompt adherence (between 0 and 10). Defaults to 8.
            steps (int, optional): number of iterations the generator should complete. Defaults to 50.

        Returns:
            PIL.Image.Image: The generated album cover image.
        """
        body = {
            "text_prompts": [{"text": prompt}],
            "cfg_scale": cfg_scale,  # Prompt adherence (7–10 is good)
            "steps": steps,     # More = better detail, slower
        }

        start = time.time()
        response = self.client.invoke_model(
            modelId="stability.stable-diffusion-xl-v1",
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body)
        )
        inference_time = time.time() - start


        result = json.loads(response["body"].read())
        image_bytes = result['artifacts'][0]['base64']  

        # Decode and display
        image = Image.open(BytesIO(base64.b64decode(image_bytes)))
        image.save(output_dir / "generated_img.png")
        if self.verbose:
            print("##IMAGE GENERATION")
            print(f"Inference time: {inference_time:.2f}s")
            image.show()

        return image