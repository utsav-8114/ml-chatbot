import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report

# Load and preprocess data
df = pd.read_csv('medquad.csv')
df = df.dropna()
df['question'] = df['question'].str.lower()

# Keep only meaningful question words
question_words = ['what', 'how', 'when', 'where', 'why', 'who', 'which']
df = df[df['question'].str.split().str[0].isin(question_words)]
df = df.drop_duplicates().reset_index(drop=True)

# Clean text
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\(.*?\)", "", text)  # Remove text in brackets
    text = re.sub(r'\s+', ' ', text.strip().lower())  # Normalize spaces
    return text

df['question'] = df['question'].apply(clean_text)

# Replace `answer` with a categorized target variable
df['intent'] = df['answer'].factorize()[0]  # Convert text answers into categories

# Prepare X and y
X = df['question'].values
y = pd.get_dummies(df['intent']).values  # Use intent categories

# Tokenize and pad sequences
tokenizer = Tokenizer(oov_token='<OOV>')  # Handle unseen words
tokenizer.fit_on_texts(X)
X = tokenizer.texts_to_sequences(X)

max_len = 100
X = pad_sequences(X, maxlen=max_len, padding='post')

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Build improved model
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=max_len))

# Bidirectional LSTM to improve learning
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Dropout(0.5))

model.add(Bidirectional(LSTM(128)))
model.add(BatchNormalization())

# Dense layers for better classification
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(y_train.shape[1], activation='softmax'))

# Compile and train
model.compile(optimizer=Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

model.save("model.py")

