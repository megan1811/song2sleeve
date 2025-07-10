import streamlit as st
from pathlib import Path
import tempfile
from pipeline import *
import time
import random

## STYLE

GRADIENT_THEMES = [
    ("#e0f7fa", "#80deea"),  # light blue
    ("#fff3e0", "#ffb74d"),  # orange-peach
    ("#e8f5e9", "#81c784"),  # green
    ("#fce4ec", "#f06292"),  # pink
    ("#ede7f6", "#9575cd"),  # purple
    ("#f3e5f5", "#ce93d8"),  # lavender
]

# Generate new gradient color per session
if "bg_gradient" not in st.session_state:
    st.session_state.bg_gradient = random.choice(GRADIENT_THEMES)

end_color, start_color = st.session_state.bg_gradient
st.markdown(f"""
    <style>
    body {{
        background: linear-gradient(135deg, {start_color}, {end_color});
        background-size: 600% 600%;
        background-position: 0% 0%;

        animation: gradientFlow 10s linear infinite;

        background-attachment: fixed;
    }}
    @keyframes gradientFlow {{
        0% {{
            background-position: 0% 0%;
        }}
        
        50% {{
            background-position: 100% 100%;
        }}
        
        100% {{
            background-position: 0% 0%;
        }}
    }}
    .stApp {{
        background: transparent;
    }}
    .st-key-container {{
        background-color: white;
        padding: 2rem;
        border-radius: 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        margin-top: 2rem;
        margin-bottom: 2rem;
    }}

    </style>
""", unsafe_allow_html=True)


## STREAMLIT APP

st.set_page_config(page_title="Song2Sleeve", layout="centered")

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "home"
if "result" not in st.session_state:
    st.session_state.result = None

# Home Page
def show_home():
    container = st.container(key="container")
    container.title("🎵 Song2Sleeve")
    container.markdown("Upload a `.wav` song and get a custom album cover based on the lyrics + musical features.")

    # media uploader
    uploaded_file = container.file_uploader("Upload your `.wav` file", type=["wav"])

    if uploaded_file is not None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / "input.wav"
            with open(tmp_path, "wb") as f:
                # save file locally
                f.write(uploaded_file.read())
            
            with container.status("Analyzing track...", expanded=True) as status:
                start = time.time()
                
                status.update(label="🎙️ Separating vocals and instrumental... [1/5]")
                vocals_path, instrumental_path = split_audio_2stems(tmp_path, Path(tmpdir))
                
                status.update(label="📝 Transcribing lyrics... [2/5]")
                lyrics = vocals_to_transcript(vocals_path)

                status.update(label="🎛️ Extracting audio features... [3/5]")
                tempo, spectral_centroid = extract_audio_features(instrumental_path)
                instrument_tags = extract_instrument_tags(instrumental_path)

                status.update(label="🧠 Generating album prompt... [4/5]")
                prompt = generate_album_cover_prompt(lyrics, tempo, spectral_centroid, instrument_tags)

                status.update(label="🎨 Generating album cover image... [5/5]")
                image = generate_album_cover(prompt, Path(tmpdir))
                
                result = {
                    "lyrics": lyrics,
                    "tempo": tempo,
                    "spectral_centroid": spectral_centroid,
                    "tags": instrument_tags,
                    "image": image,
                    "prompt": prompt,
                    "inference_time": round(time.time() - start, 2)
                }

                status.update(label="✅ Done!", state="complete")
            
            # Send result dictionary to show_result page
            st.session_state.result = result
            st.session_state.page = "result"
            st.rerun()


# Show Result Page
def show_result():
    # locally save result sent from home page
    result = st.session_state.result
    
    container = st.container(key="container")
    container.title("🎵 Song2Sleeve")
    
    container.image(result["image"], caption="🎨 Generated Album Cover")

    container.markdown("### Analysis Summary")
    container.markdown(f"**📝 Lyrics (excerpt):** {result['lyrics'][:300]}...")
    container.markdown(f"**🎶 Tempo:** {result['tempo']} BPM")
    container.markdown(f"**🎛️ Timbre:** {result['spectral_centroid']} Hz")
    container.markdown(f"**🔖 Instrument Tags:** {result['tags']}")
    container.markdown(f"**🎯 Prompt:** {result['prompt']}")
    container.markdown(f"**⏱️ Total Inference Time:** {result['inference_time']}s")

    # navigate back to home page
    st.button("🔄 Start Over", on_click=lambda: st.session_state.update({"page": "home", "result": None, "processing": False}))



# Main Router
if st.session_state.page == "home":
    show_home()
elif st.session_state.page == "result":
    show_result()