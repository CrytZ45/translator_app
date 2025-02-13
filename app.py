from flask import Flask, request, render_template
import numpy as np
import pickle
import tensorflow.lite as tflite
from gtts import gTTS
import os

app = Flask(__name__)

# Load Tokenizers
with open("ka_tokenizer.pkl", "rb") as f:
    ka_tokenizer = pickle.load(f)
with open("en_tokenizer.pkl", "rb") as f:
    en_tokenizer = pickle.load(f)

with open("ka_tokenizer_reversed.pkl", "rb") as f:
    ka_tokenizer_reversed = pickle.load(f)
with open("en_tokenizer_reversed.pkl", "rb") as f:
    en_tokenizer_reversed = pickle.load(f)

# Load TFLite models
def load_tflite_model(model_path):
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

kamayo_to_eng_model = load_tflite_model("translator_model.tflite")
eng_to_kamayo_model = load_tflite_model("translator_model_reverse.tflite")

# Translation Functions
def translate_with_tflite(interpreter, tokenizer_input, tokenizer_output, text, max_len=20):
    seq = tokenizer_input.texts_to_sequences([text])
    seq_padded = np.zeros((1, max_len))
    seq_padded[0, :len(seq[0])] = seq[0]

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], seq_padded.astype(np.float32))
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    predicted_seq = np.argmax(prediction, axis=-1)
    return " ".join(tokenizer_output.index_word[i] for i in predicted_seq[0] if i > 0)

# Replace pyttsx3 with gTTS
def save_audio(text):
    tts = gTTS(text=text, lang="en")
    tts.save("static/output.mp3")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        input_text = request.form.get("input_text")
        translation_direction = request.form.get("translation_direction")

        if translation_direction == "kamayo_to_english":
            translated_text = translate_with_tflite(kamayo_to_eng_model, ka_tokenizer, en_tokenizer, input_text)
        else:
            translated_text = translate_with_tflite(eng_to_kamayo_model, en_tokenizer_reversed, ka_tokenizer_reversed, input_text)

        save_audio(translated_text)  # Generate speech file

        return render_template("index.html", input_text=input_text, translated_text=translated_text, audio_file="static/output.mp3")

    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
