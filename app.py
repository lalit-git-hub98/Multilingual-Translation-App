import streamlit as st
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment
import os
import tempfile

# Load Translation Model
@st.cache_resource
def load_m2m_model():
    model_name = "facebook/m2m100_418M"
    model = M2M100ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = M2M100Tokenizer.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_m2m_model()

# Language Mapping
language_codes = {
    "English": "en",
    "Hindi": "hi",
    "Marathi": "mr",
    "Chinese": "zh",
    "Japanese": "ja",
    "Spanish": "es",
    "Korean": "ko"
}

# Streamlit App
st.markdown("<h1 style='text-align: center;'>🔄 PolyglotPal 🔄</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center;'>Your friendly multilingual assistant</h5>", unsafe_allow_html=True)
st.sidebar.header("Language Translation Options")

# Input Language
input_language = st.sidebar.selectbox("Select Input Language", list(language_codes.keys()))

# Target Language
target_language = st.sidebar.selectbox(
    "Select Target Language",
    [lang for lang in language_codes.keys() if lang != input_language]
)

# Input Method
input_method = st.radio("Select Input Method", ["Text Input", "Speech Input"])

if input_method == "Speech Input":
    st.subheader("Upload Audio File for Translation")
    audio_file = st.file_uploader("Upload an MP3 file", type=["mp3"])

# Text Input or Speech-to-Text
input_text = ""
if input_method == "Text Input":
    st.subheader(f"Input Text ({input_language})")
    input_text = st.text_area("Enter text to translate:", height=150)
elif input_method == "Speech Input" and audio_file:
    # Process MP3 to WAV
    try:
        # Convert MP3 to WAV using pydub
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav_file:
            audio = AudioSegment.from_file(audio_file, format="mp3")
            audio.export(tmp_wav_file.name, format="wav")
        
        # Perform Speech-to-Text
        recognizer = sr.Recognizer()
        with sr.AudioFile(tmp_wav_file.name) as source:
            audio_data = recognizer.record(source)
            input_text = recognizer.recognize_google(audio_data, language=language_codes[input_language])
            st.success(f"Recognized Speech: {input_text}")
    except Exception as e:
        st.error(f"Error processing audio: {e}")

if st.button("Translate"):
    if input_text.strip():
        try:
            # Set source and target languages
            tokenizer.src_lang = language_codes[input_language]
            encoded_input = tokenizer(input_text, return_tensors="pt")
            
            # Generate translation
            generated_tokens = model.generate(
                **encoded_input,
                forced_bos_token_id=tokenizer.lang_code_to_id[language_codes[target_language]]
            )
            translated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
            
            # Display Translation
            st.subheader(f"Translated Text ({target_language})")
            st.text_area("Translation Result:", value=translated_text, height=150, disabled=True)
            
            # Text-to-Speech
            st.subheader("Text-to-Speech Output")
            tts = gTTS(translated_text, lang=language_codes[target_language])
            tts.save("translated_audio.mp3")
            st.audio("translated_audio.mp3", format="audio/mp3")
        except Exception as e:
            st.error(f"Translation error: {e}")
    else:
        st.error("Please provide text or speech input for translation.")
