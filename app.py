from flask import Flask, request, render_template, jsonify
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import pyttsx3

app = Flask(__name__)

# Load the translation models
kamayo_to_eng_model = load_model("translator_model.h5")  # Kamayo → English
eng_to_kamayo_model = load_model("translator_model_reverse.h5")  # English → Kamayo

# Load the tokenizers
with open("ka_tokenizer.pkl", "rb") as f:
    ka_tokenizer = pickle.load(f)
with open("en_tokenizer.pkl", "rb") as f:
    en_tokenizer = pickle.load(f)

with open("ka_tokenizer_reversed.pkl", "rb") as f:
    ka_tokenizer_reversed = pickle.load(f)
with open("en_tokenizer_reversed.pkl", "rb") as f:
    en_tokenizer_reversed = pickle.load(f)

max_len = 20  # Maximum length for input/output sequences

# Initialize Text-to-Speech
engine = pyttsx3.init()

# Translation Functions
def translate_kamayo_to_english(text):
    seq = ka_tokenizer.texts_to_sequences([text])
    seq_padded = np.zeros((1, max_len))
    seq_padded[0, :len(seq[0])] = seq[0]
    prediction = kamayo_to_eng_model.predict(seq_padded)
    predicted_seq = np.argmax(prediction, axis=-1)
    english_text = " ".join(
        en_tokenizer.index_word[i] for i in predicted_seq[0] if i > 0
    )
    return english_text

def translate_english_to_kamayo(text):
    seq = en_tokenizer_reversed.texts_to_sequences([text])
    seq_padded = np.zeros((1, max_len))
    seq_padded[0, :len(seq[0])] = seq[0]
    prediction = eng_to_kamayo_model.predict(seq_padded)
    predicted_seq = np.argmax(prediction, axis=-1)
    kamayo_text = " ".join(
        ka_tokenizer_reversed.index_word[i] for i in predicted_seq[0] if i > 0
    )
    return kamayo_text

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        input_text = request.form.get("input_text")
        translation_direction = request.form.get("translation_direction")

        if translation_direction == "kamayo_to_english":
            translated_text = translate_kamayo_to_english(input_text)
        else:
            translated_text = translate_english_to_kamayo(input_text)

        # Text-to-Speech
        engine.save_to_file(translated_text, "static/output.mp3")
        engine.runAndWait()

        return render_template(
            "index.html",
            input_text=input_text,
            translated_text=translated_text,
            translation_direction=translation_direction,
            audio_file="static/output.mp3",
        )
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
