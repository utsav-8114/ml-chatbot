import tensorflow as tf
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split

# Load your dataset
df = pd.read_csv('medquad.csv')  # Replace with your actual dataset path

# Data Cleaning: Handle missing values and remove duplicates
df.dropna(subset=['question', 'answer'], inplace=True)  # Remove rows with missing values in 'question' or 'answer'
df.drop_duplicates(subset=['question', 'answer'], inplace=True)  # Remove duplicates

# Preprocessing function for text data
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Remove special characters and numbers
    return text

# Apply text preprocessing
df['question'] = df['question'].apply(preprocess_text)
df['answer'] = df['answer'].apply(preprocess_text)

# Tokenizing the 'question' column
tokenizer_question = tf.keras.preprocessing.text.Tokenizer()
tokenizer_question.fit_on_texts(df['question'])
X = tokenizer_question.texts_to_sequences(df['question'])
X = tf.keras.preprocessing.sequence.pad_sequences(X, padding='post', maxlen=26)  # Set maxlen=26

# Tokenizing the 'answer' column
tokenizer_answer = tf.keras.preprocessing.text.Tokenizer()
tokenizer_answer.fit_on_texts(df['answer'])
y = tokenizer_answer.texts_to_sequences(df['answer'])
y = tf.keras.preprocessing.sequence.pad_sequences(y, padding='post', maxlen=26)  # Set maxlen=26

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model architecture
vocab_size_question = len(tokenizer_question.word_index) + 1  # +1 for padding
vocab_size_answer = len(tokenizer_answer.word_index) + 1  # +1 for padding

model = tf.keras.Sequential()

# Embedding layer for the 'question' input
model.add(tf.keras.layers.Embedding(input_dim=vocab_size_question, output_dim=100, input_length=X.shape[1]))

# LSTM layer
model.add(tf.keras.layers.LSTM(512, return_sequences=False))  # Return only the last output
model.add(tf.keras.layers.Dropout(0.5))

# Output layer to predict the next token in the sequence
model.add(tf.keras.layers.Dense(vocab_size_answer, activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train[:, 0], epochs=200 , batch_size=64, validation_data=(X_test, y_test[:, 0]))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test[:, 0])
print(f"Test Accuracy: {accuracy:.2f}")



