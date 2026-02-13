import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ------------------------------
# Load saved files
# ------------------------------


@st.cache_resource
def load_resources():
    model = load_model("lstm_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("max_len.pkl", "rb") as f:
        max_len = pickle.load(f)
    return model, tokenizer, max_len


model, tokenizer, max_len = load_resources()

# Create reverse word index (faster lookup)
index_word = {index: word for word, index in tokenizer.word_index.items()}

# ------------------------------
# Multi-word prediction function
# ------------------------------


def predict_next_words(text, n_words):
    current_text = text

    for _ in range(n_words):
        sequence = tokenizer.texts_to_sequences([current_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_len-1, padding='pre')

        preds = model.predict(sequence, verbose=0)
        predicted_index = np.argmax(preds)

        next_word = index_word.get(predicted_index, "")

        if next_word == "":
            break

        current_text += " " + next_word

    return current_text


# ------------------------------
# Streamlit UI
# ------------------------------

st.set_page_config(page_title="Next Word Prediction", layout="centered")

st.title("🧠 LSTM Text Completion Engine")
st.write("Enter a sentence and choose how many words to generate.")

with st.form(key="prediction_form", clear_on_submit=False):
    user_input = st.text_input(
        "✍️ Enter text:", placeholder="Type a sentence here..."
    )

    num_words = st.slider(
        "🔢 Number of words to generate:",
        min_value=1,
        max_value=20,
        value=5
    )

    submit_button = st.form_submit_button("Generate")

if submit_button:
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        generated_text = predict_next_words(user_input, num_words)
        st.success(f"**Generated Text:** {generated_text}")

# ------------------------------
# Footer
# ------------------------------

st.markdown("---")
st.caption("LSTM-based Text Completion using Streamlit")
