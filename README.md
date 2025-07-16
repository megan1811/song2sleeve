# Song 2 Sleeve

Transform any song into a custom AI-generated album cover.

Song2Sleeve is a full-stack Streamlit application that analyzes any .wav track and generates a unique album cover inspired by both its lyrics and audio characteristics.

Built for the GenAI Hackathon hosted by AWS and Impetus, this project fuses audio analysis with generative AI to bring music to life visually.

## AI Pipeline Architecture

The Song2Sleeve pipeline transforms a raw .wav music file into a fully generated album cover image by integrating signal processing, machine listening, large language modeling, and diffusion-based image generation. The pipeline was designed to be modular, interpretable, and optimized for both vocal and instrumental audio. Here’s a deep dive into each component:

### 1. Audio Input

- **User Input:** The user uploads a .wav file via the Streamlit frontend.

### 2. Vocal–Instrumental Separation with Demucs

- **Tool:** Facebook AI’s Demucs, a state-of-the-art deep neural network for music source separation.
- **Function:** Separates the uploaded audio into two stems: `vocals.wav` for lyric transcription & `instrumental.wav` for musical analysis (timbre, tempo, instrument tags)
- `Note:` This improves downstream performance by isolating clean vocal and instrumental signals.

### 3. Lyrics Transcription

- **Model:** faster-whisper, a highly optimized and quantized implementation of OpenAI’s Whisper ASR.
- **Purpose:** Converts vocals.wav into a raw lyric transcript, enabling a semantic interpretation of the track’s narrative and themes.

### 4. Instrumental Analysis

Two complementary methods are applied to instrumental.wav:

#### A. Feature Extraction (Librosa)

Metrics Extracted:

- **Tempo (BPM):** Beat estimation using beat tracking.
- **Spectral Centroid (Hz):** A proxy for timbre or brightness.

#### B. Instrument Tagging (YAMNet)

- **Model:** YAMNet — A convolutional neural network trained on over 500 AudioSet labels.
- **Purpose:** Predicts the dominant instruments and sound textures in the audio.

### 5. Prompt Engineering with Claude Sonnet

- **LLM:** Anthropic Claude 3 Sonnet, used for interpreting poetic text, accessed via **AWS Bedrock**.
- **Goal:** To synthesize a compelling and stylistically consistent text-to-image prompt that encompasses the outputs from **3. Lyrics Transcripton** and **4. Instrumental Analysis**.

### 6. Image Generation with SDXL

- **Model**: Stable Diffusion XL (SDXL) via AWS Bedrock (stability.stable-diffusion-xl-v1)
- Here we generate the final image using the prompt generated in the previous step.

## Testing & Evaluation

We validated the pipeline across a variety of genres, using royalty-free tracks from Pixabay Music. Results showed:

- Album covers with a similar aesthetic to the imput music
- Distinct mood capture in the final image
- Pertinent album covers even for songs with minimal lyrics

## Installation & Deployment

We provide two deployment pathways:

1. Local Testing

```
    poetry install
    streamlit run app.py
```

2. AWS Production Deployment

Hosted via EC2 t3.medium with Amazon Linux, publically available:
