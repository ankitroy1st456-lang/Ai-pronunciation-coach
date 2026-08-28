import streamlit as st
import torch
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
import soundfile as sf
from io import BytesIO
import string
import difflib
from gtts import gTTS
import random
from PIL import Image

# Set up the website page layout
st.set_page_config(page_title="AI Pronunciation Coach", layout="wide")
st.title("AI Pronunciation Coach")

# Load the speech-to-text AI model (Whisper)
@st.cache_resource
def get_speech_model():
    return pipeline("automatic-speech-recognition", model="openai/whisper-tiny.en", device="cpu")

speech_ai = get_speech_model()

# Load the image captioning AI model (BLIP)
@st.cache_resource
def get_image_ai():
    model_name = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    return processor, model

image_processor, image_ai = get_image_ai()
# Sentences for users to practice, split by difficulty levels
SENTENCE_LEVELS = {
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

# Create memory variables so the app remembers your data when the page refreshes
if "recorder_id" not in st.session_state:
    st.session_state.recorder_id = 0
if "user_level" not in st.session_state:
    st.session_state.user_level = 1
if "current_sentence" not in st.session_state:
    st.session_state.current_sentence = random.choice(SENTENCE_LEVELS[st.session_state.user_level])
if "score_results" not in st.session_state:
    st.session_state.score_results = None
if "score_history" not in st.session_state:
    st.session_state.score_history = []
if "practice_mode" not in st.session_state:
    st.session_state.practice_mode = "Challenge Pool"
if "saved_custom_text" not in st.session_state:
    st.session_state.saved_custom_text = ""
if "saved_image_bytes" not in st.session_state:
    st.session_state.saved_image_bytes = None

# What happens when a user switches practice modes
def handle_mode_change():
    if st.session_state.practice_mode == "Custom Text":
        st.session_state.current_sentence = st.session_state.saved_custom_text
    elif st.session_state.practice_mode == "Challenge Pool":
        st.session_state.current_sentence = random.choice(SENTENCE_LEVELS[st.session_state.user_level])
    elif st.session_state.practice_mode == "Image Caption":
        st.session_state.current_sentence = ""
        st.session_state.saved_image_bytes = None # Clear old image
    st.session_state.score_results = None

# What happens when a user types in their own text
def handle_custom_text_change():
    st.session_state.current_sentence = st.session_state.saved_custom_text
    st.session_state.score_results = None

# Show the practice mode choice buttons
st.radio(
    "Choose Practice Mode:",
    ["Challenge Pool", "Custom Text", "Image Caption"],
    horizontal=True,
    key="practice_mode",
    on_change=handle_mode_change,
)

# Render the layout based on the chosen mode
if st.session_state.practice_mode == "Challenge Pool":
    st.write(f"**Current Level:** {st.session_state.user_level}")
    st.info(st.session_state.current_sentence)

elif st.session_state.practice_mode == "Custom Text":
    st.text_input(
        "Type your own sentence here and press enter :",
        key="saved_custom_text",
        on_change=handle_custom_text_change,
    )

elif st.session_state.practice_mode == "Image Caption":
    uploaded_file = st.file_uploader("Upload an image to generate a practice sentence:", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        st.session_state.saved_image_bytes = uploaded_file.getvalue()

    if st.session_state.saved_image_bytes is not None:
        pil_image = Image.open(BytesIO(st.session_state.saved_image_bytes)).convert("RGB")
        st.image(pil_image, caption="Uploaded Image", width=350)

        if st.button("Generate Caption"):
            with st.spinner("Creating a sentence from your image..."):
                inputs = image_processor(images=pil_image, return_tensors="pt")
                with torch.no_grad():
                    generated_ids = image_ai.generate(**inputs, max_new_tokens=30)
                caption_text = image_processor.decode(generated_ids[0], skip_special_tokens=True)
                
                # Make it look like a neat sentence
                caption_text = caption_text.strip()
                if caption_text:
                    caption_text = caption_text.capitalize()
                    if not caption_text.endswith((".", "!", "?")):
                        caption_text += "."

            if caption_text:
                st.session_state.current_sentence = caption_text
                st.session_state.score_results = None
                st.success(f"**Generated Caption:** {st.session_state.current_sentence}")
                st.rerun()
            else:
                st.error("❌ Could not generate a caption for this image.")
    else:
        st.session_state.current_sentence = ""
        st.warning("Please upload an image file to begin.")
# Only show the audio tools if there is a sentence ready to practice
if st.session_state.current_sentence.strip():
    # Make a computer-generated voice example for the user to listen to
    text_to_speech = gTTS(text=st.session_state.current_sentence, lang="en")
    speech_bytes = BytesIO()
    text_to_speech.write_to_fp(speech_bytes)
    speech_bytes.seek(0)
    st.audio(speech_bytes.read(), format="audio/mp3")

    # The audio recorder input box (uses a dynamic ID to clear out old recordings)
    recorded_audio = st.audio_input("Record your voice", key=f"mic_{st.session_state.recorder_id}")

    # Helper function to remove punctuation and uppercase letters for fair grading
    def clean_word(text):
        return text.lower().translate(str.maketrans("", "", string.punctuation))

    if recorded_audio is not None:
        if st.button("Analyze", type="primary"):
            # Check how long the audio clip is
            raw_bytes = recorded_audio.read()
            audio_stream = BytesIO(raw_bytes)
            audio_data, sample_rate = sf.read(audio_stream)
            seconds = len(audio_data) / sample_rate
            
            # Stop if the user recorded for more than 10 seconds
            if seconds > 10.0:
                st.error(f"⚠️ Recording too long ({seconds:.1f}s). Please keep it under 10 seconds.")
                st.session_state.score_results = None
                st.session_state.recorder_id += 1
                st.rerun()
            else:
                # Turn the user's spoken audio into text strings
                with st.spinner("Listening to your voice..."):
                    ai_transcription = speech_ai({"raw": audio_data, "sampling_rate": sample_rate})
                    heard_text = ai_transcription["text"].strip()
                
                # Split sentences into lists of single words
                target_word_list = st.session_state.current_sentence.split()
                heard_word_list = heard_text.split()
                
                clean_targets = [clean_word(w) for w in target_word_list]
                clean_heard = [clean_word(w) for w in heard_word_list]
                
                # Match the words to see which ones are correct
                word_matcher = difflib.SequenceMatcher(None, clean_targets, clean_heard)
                styled_output_list = [""] * len(target_word_list)
                correct_count = 0
                
                # Highlight correct words in green and wrong words in crossed-out red
                for tag, target_start, target_end, heard_start, heard_end in word_matcher.get_opcodes():
                    for index in range(target_start, target_end):
                        word = target_word_list[index]
                        if tag == 'equal':
                            styled_output_list[index] = f"<span style='color:green; font-weight:bold;'>{word}</span>"
                            correct_count += 1
                        else:
                            styled_output_list[index] = f"<span style='color:red; text-decoration:line-through;'>{word}</span>"
                
                # Calculate the percentage accuracy score
                final_score = (correct_count / len(target_word_list)) * 100
                st.session_state.score_history.append(final_score)
                
                # Check if the user qualifies to level up or down
                should_level_up = final_score >= 85.0 and st.session_state.user_level < 3 and st.session_state.practice_mode == "Challenge Pool"
                should_level_down = final_score < 60.0 and st.session_state.user_level > 1 and st.session_state.practice_mode == "Challenge Pool"
                
                # Pack the calculations together into session memory
                st.session_state.score_results = {
                    "percentage": final_score,
                    "transcript": heard_text,
                    "html_markup": " ".join(styled_output_list),
                    "playback_bytes": raw_bytes,
                    "level_up": should_level_up,
                    "level_down": should_level_down
                }
                
                # Change levels based on the results flags
                if should_level_up:
                    st.session_state.user_level += 1
                elif should_level_down:
                    st.session_state.user_level -= 1

                # Update recorder ID to clear out the microphone slot, then refresh page layout
                st.session_state.recorder_id += 1
                st.rerun()
            
else:
    st.session_state.score_results = None


# Show performance results panel if data exists
if st.session_state.score_results is not None or st.session_state.score_history:
    st.write("---")
    left_column, right_column = st.columns(2, gap="large")
    
    # Left side panel: Word scoring analysis
    with left_column:
        if st.session_state.score_results is not None:
            results_data = st.session_state.score_results
            st.subheader("📊 Performance Analysis")
            st.metric(label="Accuracy", value=f"{results_data['percentage']:.1f}%")
            
            st.write(f"**What the AI Heard:** \"{results_data['transcript']}\"")
            st.markdown(f"**Word Comparison:** {results_data['html_markup']}", unsafe_allow_html=True)
            
            st.write("**Listen to your recording:**")
            st.audio(results_data["playback_bytes"], format="audio/wav")
            
            if results_data["level_up"]:
                st.success("Level up! Great work.")
            elif results_data["level_down"]:
                st.error("Lowering difficulty to practice basic phrases.")

    # Right side panel: History progress tracking graph
    with right_column:
        if st.session_state.score_history:
            st.subheader("📈 Progress History")
            st.line_chart(st.session_state.score_history)

# Button functions to handle skipping or retrying items
def load_next_sentence():
    st.session_state.current_sentence = random.choice(SENTENCE_LEVELS[st.session_state.user_level])
    st.session_state.score_results = None

def clear_current_metrics():
    st.session_state.score_results = None

# Show action buttons based on the user's active screen
if st.session_state.practice_mode == "Challenge Pool":
    button_col1, button_col2 = st.columns(2)
    with button_col1:
        st.button("🔄 Try Same Sentence Again", on_click=clear_current_metrics, use_container_width=True)
    with button_col2:
        st.button("➡️ Move to Next Challenge", on_click=load_next_sentence, use_container_width=True)
else:
    st.button("🔄 Reset This Attempt", on_click=clear_current_metrics, use_container_width=True)
