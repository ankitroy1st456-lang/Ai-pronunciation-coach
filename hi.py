import streamlit as st
import torch
from transformers import pipeline
import soundfile as sf
from io import BytesIO
import string
import difflib
from gtts import gTTS
import random

# Set up page layout
st.set_page_config(page_title="AI Pronunciation Coach", layout="wide")
st.title("AI Pronunciation Coach")

# Load lightweight speech recognition model
@st.cache_resource
def load_model():
    return pipeline("automatic-speech-recognition", model="openai/whisper-tiny.en", device="cpu")

model = load_model()

# Practice sentences grouped by level
POOLS = {
    1: [
        "The cat sat on the mat.",
        "Dogs love to run.",
        "A blue bird sings sweet songs.",
        "We eat fresh apples every day.",
        "The sun shines bright in the sky.",
        "They walk to school together.",
        "Water is good for your health."
    ],
    2: [
        "Reading books expands your vocabulary.",
        "The unexpected rain ruined our plans.",
        "The quick brown fox jumps over the lazy dog.",
        "She traveled to ancient ruins during her summer vacation.",
        "Learning a new language requires patience and consistent practice.",
        "The local bakery makes fresh bread every morning.",
        "Technology plays a massive role in modern communication."
    ],
    3: [
        "Environmental preservation efforts are gaining momentum.",
        "Symmetrical architectural blueprints demand precision.",
        "A pessimistic perspective consistently complicates simple corporate resolutions.",
        "Linguistic structural algorithms require extensive background computational resources.",
        "Anonymity and institutionalized vulnerability create unique psychological phenomena.",
        "Superficial characteristics unnecessarily obfuscate deep architectural flaws.",
        "The international space station orbits the earth at an incredible velocity."
    ]
}

# Initialize session state variables
if "input_key" not in st.session_state:
    st.session_state.input_key = 0
if "level" not in st.session_state:
    st.session_state.level = 1
if "text" not in st.session_state:
    st.session_state.text = random.choice(POOLS[st.session_state.level])
if "results" not in st.session_state:
    st.session_state.results = None
if "history" not in st.session_state:
    st.session_state.history = []
if "mode" not in st.session_state:
    st.session_state.mode = "Challenge Pool"

# Toggle between challenge sentences and custom entry
st.session_state.mode = st.radio("Choose Practice Mode:", ["Challenge Pool", "Custom Text"], horizontal=True)

# Set the active practice text based on mode
if st.session_state.mode == "Challenge Pool":
    st.write(f"**Current Level:** {st.session_state.level}")
    st.info(st.session_state.text)
else:
    custom_text = st.text_input("Type your own sentence here:", "").strip()
    st.session_state.text = custom_text

# Process audio input if text is present
if st.session_state.text:
    # Generate and play reference audio guide
    tts = gTTS(text=st.session_state.text, lang="en")
    tts_bytes = BytesIO()
    tts.write_to_fp(tts_bytes)
    tts_bytes.seek(0)
    st.audio(tts_bytes.read(), format="audio/mp3")

    # Audio recorder linked to session key for dynamic resets
    audio = st.audio_input("Record your voice", key=f"audio_recorder_{st.session_state.input_key}")

    # Remove capitalization and punctuation for scoring
    def clean(word):
        return word.lower().translate(str.maketrans("", "", string.punctuation))

    if audio is not None:
        if st.button("Analyze"):
            # Measure recording duration
            raw_audio_bytes = audio.read()
            audio_bytes = BytesIO(raw_audio_bytes)
            data, rate = sf.read(audio_bytes)
            audio_duration = len(data) / rate
            
            # Reject files longer than 10 seconds
            if audio_duration > 10.0:
                st.error(f"⚠️ Recording too long ({audio_duration:.1f}s). Please keep it under 10 seconds.")
                st.session_state.results = None
                st.session_state.input_key += 1
                st.button("🔄 Try Again")
            else:
                # Transcribe spoken audio
                with st.spinner("Processing..."):
                    result = model({"raw": data, "sampling_rate": rate})
                    spoken = result["text"].strip()
                
                # Tokenize strings into word arrays
                target_words = st.session_state.text.split()
                spoken_words = spoken.split()
                
                clean_target = [clean(w) for w in target_words]
                clean_spoken = [clean(w) for w in spoken_words]
                
                # Compare arrays word by word
                matcher = difflib.SequenceMatcher(None, clean_target, clean_spoken)
                output = [""] * len(target_words)
                correct = 0
                
                # Apply visual styling flags based on matching results
                for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                    for idx in range(i1, i2):
                        word = target_words[idx]
                        if tag == 'equal':
                            output[idx] = f"<span style='color:green; font-weight:bold;'>{word}</span>"
                            correct += 1
                        else:
                            output[idx] = f"<span style='color:red; text-decoration:line-through;'>{word}</span>"
                
                # Calculate accuracy score
                score = (correct / len(target_words)) * 100
                st.session_state.history.append(score)
                
                # Set evaluation state results (includes raw transcript and raw audio bytes)
                st.session_state.results = {
                    "score": score,
                    "spoken": spoken,
                    "html": " ".join(output),
                    "user_audio": raw_audio_bytes,
                    "level_up": score >= 85.0 and st.session_state.level < 3 and st.session_state.mode == "Challenge Pool",
                    "level_down": score < 60.0 and st.session_state.level > 1 and st.session_state.mode == "Challenge Pool"
                }
                
                # Change user level criteria dynamically
                if st.session_state.results["level_up"]:
                    st.session_state.level += 1
                elif st.session_state.results["level_down"]:
                    st.session_state.level -= 1

                # Advance key to force audio clearing and refresh layout
                st.session_state.input_key += 1
                st.rerun()
            
else:
    st.session_state.results = None
    st.warning("Please type a valid phrase in the custom text input box before recording.")

# Render layout panels side-by-side
if st.session_state.results is not None or st.session_state.history:
    st.write("---")
    main_col1, main_col2 = st.columns(2, gap="large")
    
    # Left column: Evaluation text feedback elements
    with main_col1:
        if st.session_state.results is not None:
            res = st.session_state.results
            st.subheader("📊 Performance Analysis")
            st.metric(label="Accuracy", value=f"{res['score']:.1f}%")
            
            # Displays the exact text generated by the STT model
            st.write(f"**What the AI Heard:** \"{res['spoken']}\"")
            st.markdown(f"**Word Comparison:** {res['html']}", unsafe_allow_html=True)
            
            # Adds playback option to hear what you recorded
            st.write("**Listen to your recording:**")
            st.audio(res["user_audio"], format="audio/wav")
            
            if res["level_up"]:
                st.success("Level up! Great work.")
            elif res["level_down"]:
                st.error("Lowering difficulty to practice basic phrases.")

    # Right column: Score trend tracking graph
    with main_col2:
        if st.session_state.history:
            st.subheader("📈 Progress History")
            st.line_chart(st.session_state.history)

# Load a new sentence challenge
def next_phrase():
    st.session_state.text = random.choice(POOLS[st.session_state.level])
    st.session_state.results = None

# Clear previous metrics for another attempt
def retry_phrase():
    st.session_state.results = None

# Render navigation buttons based on current mode
if st.session_state.mode == "Challenge Pool":
    nav_col1, nav_col2 = st.columns(2)
    with nav_col1:
        st.button("🔄 Try Same Sentence Again", on_click=retry_phrase)
    with nav_col2:
        st.button("➡️ Move to Next Challenge", on_click=next_phrase)
else:
    st.button("🔄 Reset This Attempt", on_click=retry_phrase)
