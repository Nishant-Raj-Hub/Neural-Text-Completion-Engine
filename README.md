# 🚀 Neural Text Completion Engine (LSTM-based Language Model)

## 📌 Overview

This project implements a Neural Text Completion Engine using an LSTM-based deep learning architecture.  
The model generates sentence-level completions by iteratively predicting the next word in a sequence, learning contextual dependencies from text data.

It demonstrates a complete NLP pipeline — from raw text preprocessing to training a neural language model.

---

## 🔥 Features

- End-to-end NLP pipeline from raw text to trained model
- Custom vocabulary construction using Keras Tokenizer
- Sliding window sequence generation for supervised learning
- Embedding + LSTM architecture for contextual modeling
- Sentence-level completions via iterative next-word generation
- Memory-efficient training using sparse categorical crossentropy
- Softmax-based probability distribution over 10,000-word vocabulary

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Pandas

---

## 📂 Project Workflow

### 1️⃣ Data Preprocessing

- Loaded dataset containing textual quotes
- Removed punctuation using string translation
- Tokenized text and limited vocabulary to top 10,000 words
- Converted sentences into integer sequences

### 2️⃣ Training Data Generation

- Generated input-output pairs using sliding window technique
- Created 85k+ supervised training samples
- Applied pre-padding to ensure uniform sequence length

### 3️⃣ Model Architecture

```
Input (Token IDs)
        ↓
  Embedding Layer
        ↓
    LSTM Layer
        ↓
Dense Layer + Softmax
        ↓
Sentence Completion (iterative next-word prediction)
```

---

## 🧠 Model Configuration

- **Vocabulary Size:** 10,000
- **Embedding Dimension:** 50
- **LSTM Units:** 128
- **Loss Function:** sparse_categorical_crossentropy
- **Optimizer:** Adam
- **Epochs:** 100
- **Batch Size:** 128
- **Validation Split:** 10%

---

## 📊 Input & Output Shapes

- Input Shape: `(batch_size, max_len)`
- After Embedding: `(batch_size, max_len, embedding_dim)`
- After LSTM: `(batch_size, rnn_units)`
- Final Output: `(batch_size, vocab_size)`

---

## 🎯 Objective

The primary goal of this project is to understand and implement neural sequence modeling for natural language processing tasks, including multi-word sentence completion.

This project showcases:

- Text preprocessing for NLP
- Sequence modeling using LSTM
- Neural language modeling fundamentals
- Efficient classification using sparse loss



