import pandas as pd
import matplotlib.pyplot as plt
import re
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset with specified dtypes
dtype_dict = {str(i): 'float' for i in range(1, 785)}  # 784 features + 1 label column
dtype_dict['0'] = 'str'  # Label column as string
df_alphabets = pd.read_csv('alphabets_28x28.csv', dtype=dtype_dict)

# Convert all columns to numeric, forcing errors to NaN
df_alphabets = df_alphabets.apply(pd.to_numeric, errors='coerce')

# Optionally, drop rows with NaN values if they exist
df_alphabets = df_alphabets.dropna()

# Ensure the columns are in the correct numeric format
df_alphabets = df_alphabets.astype(float)

# Visualize a few samples, ensuring index is within bounds
num_samples = min(5, len(df_alphabets))
for i in range(num_samples):
    plt.imshow(df_alphabets.iloc[i, 1:].values.reshape(28, 28), cmap='gray')
    plt.title(f"Label: {df_alphabets.iloc[i, 0]}")
    plt.show()

# Load and clean sentiment analysis dataset
df_sentiment = pd.read_csv('sentiment_analysis_dataset.csv')
print(df_sentiment.head())

# Clean the text data
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

df_sentiment['cleaned_text'] = df_sentiment['line'].apply(clean_text)

# Prepare CNN model for alphabet classification
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(26, activation='softmax')  # Assuming 26 classes for A-Z alphabets
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Assuming X_train and y_train are prepared from the dataset
# Placeholder for the alphabet data preparation
# X_train, y_train, X_val, y_val = prepare_alphabet_data(df_alphabets)

# model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Train Naive Bayes model for sentiment analysis
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df_sentiment['cleaned_text'])
y = df_sentiment['sentiment']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model_nb = MultinomialNB()
model_nb.fit(X_train, y_train)

y_pred = model_nb.predict(X_val)
print(f"Accuracy: {accuracy_score(y_val, y_pred)}")
