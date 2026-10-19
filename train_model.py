import pandas as pd
import numpy as np
import re
import pickle
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from datasets import load_dataset

print("1. Fetching dataset automatically via code...")
# This automatically downloads 16,000 labeled emotion sentences directly into memory
dataset = load_dataset("dair-ai/emotion", "split", trust_remote_code=True)
df = dataset['train'].to_pandas()

# The dataset uses numbers (0-5) for moods. We map them back to strings.
label_map = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
df['mood'] = df['label'].map(label_map)

# Tokenize the text (convert to lowercase and extract words)
def tokenize(text):
    return re.findall(r'\b\w+\b', str(text).lower())

print("2. Tokenizing 16,000 sentences...")
df['tokens'] = df['text'].apply(tokenize)

print("3. Training Word2Vec Skip-gram model on new vocabulary...")
# sg=1 enables Skip-gram architecture
word2vec_model = Word2Vec(sentences=df['tokens'], vector_size=100, window=5, min_count=1, sg=1)

# Save the expanded Word2Vec model
word2vec_model.save("./word2vec_skipgram.model")

# Convert sentences to average vectors
def get_sentence_vector(tokens, model):
    valid_vectors = [model.wv[word] for word in tokens if word in model.wv]
    if not valid_vectors:
        return np.zeros(model.vector_size)
    return np.mean(valid_vectors, axis=0)

print("4. Converting text to vectors for the classifier...")
X = np.array([get_sentence_vector(tokens, word2vec_model) for tokens in df['tokens']])
y = df['mood']

print("5. Training Mood Classifier...")
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X, y)

# Save the trained classifier
with open("mood_classifier.pkl", "wb") as f:
    pickle.dump(classifier, f)

print("Training complete! our model now understands words like 'boring'.")