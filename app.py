import streamlit as st
import pickle
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

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

# ------------------------------
# Prediction function
# ------------------------------


def predict_next_word(text):
    sequence = tokenizer.texts_to_sequences([text])[0]
    sequence = pad_sequences([sequence], maxlen=max_len-1, padding='pre')

    preds = model.predict(sequence, verbose=0)
    predicted_index = np.argmax(preds)

    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word
    return ""


# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Next Word Prediction", layout="centered")

st.title("🧠 Next Word Prediction (LSTM)")
st.write("Enter a sentence and the model will predict the **next word**.")

with st.form(key="prediction_form", clear_on_submit=False):
    user_input = st.text_input(
        "✍️ Enter text:", placeholder="Type a sentence here...")
    submit_button = st.form_submit_button("Predict Next Word")

if submit_button:
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        next_word = predict_next_word(user_input)
        st.success(f"**Predicted Next Word:** {next_word}")

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.caption("LSTM-based Next Word Prediction using Streamlit")
