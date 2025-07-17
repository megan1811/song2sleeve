# 🎵 Song2Sleeve

**Song2Sleeve** is a Streamlit app that transforms a song into a custom, professional‑grade album cover.  
Upload a `.wav` file, and the app will analyze its lyrics, tempo, timbre, and instrumentation to generate an imaginative visual prompt and render an album cover image using AWS Bedrock (Claude Sonnet + Stable Diffusion XL).

---

## ✨ Inspiration

Album art often reflects the mood and essence of a song.  
Song2Sleeve explores how AI can bridge audio analysis and creative image generation to quickly produce unique, genre‑aware cover art.

---

## 🚀 What It Does

1. **Upload** a `.wav` audio file.
2. **Analyze** the song:
   - Transcribe lyrics with [Faster‑Whisper](https://github.com/guillaumekln/faster-whisper),
   - Extract tempo & spectral centroid (timbre) with [librosa](https://librosa.org),
   - Tag instruments with [YAMNet](https://tfhub.dev/google/yamnet/1).
3. **Generate a prompt** with [Claude 3 Sonnet](https://aws.amazon.com/bedrock/).
4. **Create a cover** with [Stable Diffusion XL](https://aws.amazon.com/bedrock/).

**Output:**  
🖼️ A unique album cover image and a summary of the analyzed features.

---

## 🏗️ How We Built It

- **Frontend:** [Streamlit](https://streamlit.io) for the UI and workflow.
- **Audio Analysis:** `faster-whisper`, `librosa`, `tensorflow-hub` (YAMNet).
- **Prompt Generation:** Claude Sonnet via AWS Bedrock.
- **Image Generation:** Stable Diffusion XL via AWS Bedrock.
- **Pipeline Orchestration:** Custom Python classes (`Pipeline`, `LyricsTranscriber`, `InstrumentalTagger`).

---

## 📦 Local Installation

> ⚠️ AWS Bedrock access is required for Claude & Stable Diffusion.  
> Set your AWS credentials (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`) in your environment.

### 1️⃣ Clone the repository

```bash
git clone https://github.com/megan1811/song2sleeve.git
cd song2sleeve
```

### 2️⃣ Install dependencies

Using Poetry:

```bash
poetry install
poetry shell
```

Or using pip:

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the app

```bash
streamlit run app.py
```

Open your browser at http://localhost:8501.

---

## 📂 Project Structure

```bash
song2sleeve/
│
├── app.py            # Streamlit frontend
├── pipeline.py       # High-level audio-to-image pipeline
├── client.py         # AWS Bedrock client (Claude & SDXL)
├── models.py         # Transcription (Whisper) & tagging (YAMNet)
├── utils.py          # Feature extraction & prompt structuring
├── poetry.toml       # Poetry dependency configuration
└── README.md         # Project documentation
```
