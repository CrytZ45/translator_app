import os
import pickle
import numpy as np
from flask import Flask, request, render_template
from gtts import gTTS

app = Flask(__name__)

# Ensure the models folder exists
MODEL_DIR = os.path.join(os.getcwd(), "models")
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Load Pickle Files Safely
def load_pickle(file_name):
    file_path = os.path.join(MODEL_DIR, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing file: {file_path}")
    with open(file_path, "rb") as f:
        return pickle.load(f)

# Load Tokenizers
ka_tokenizer = load_pickle("ka_tokenizer.pkl")
en_tokenizer = load_pickle("en_tokenizer.pkl")
ka_tokenizer_reversed = load_pickle("ka_tokenizer_reversed.pkl")
en_tokenizer_reversed = load_pickle("en_tokenizer_reversed.pkl")

# Load Translation Models from Pickle Files
kamayo_to_eng_model = load_pickle("translator_model.h5")  # Kamayo → English
eng_to_kamayo_model = load_pickle("translator_model_reverse.h5")  # English → Kamayo

# Translation Function
def translate_with_model(model, tokenizer_input, tokenizer_output, text, max_len=20):
    seq = tokenizer_input.texts_to_sequences([text])
    seq_padded = np.zeros((1, max_len))
    seq_padded[0, :len(seq[0])] = seq[0]

    prediction = model.predict(seq_padded)
    predicted_seq = np.argmax(prediction, axis=-1)

    return " ".join(tokenizer_output.index_word[i] for i in predicted_seq[0] if i > 0)

# Convert Text to Speech
def save_audio(text):
    tts = gTTS(text=text, lang="en")
    audio_path = os.path.join("static", "output.mp3")
    tts.save(audio_path)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        input_text = request.form.get("input_text")
        translation_direction = request.form.get("translation_direction")

        if translation_direction == "kamayo_to_english":
            translated_text = translate_with_model(kamayo_to_eng_model, ka_tokenizer, en_tokenizer, input_text)
        else:
            translated_text = translate_with_model(eng_to_kamayo_model, en_tokenizer_reversed, ka_tokenizer_reversed, input_text)

        save_audio(translated_text)  # Generate speech file

        return render_template("index.html", input_text=input_text, translated_text=translated_text, audio_file="static/output.mp3")

    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
