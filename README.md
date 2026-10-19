🎭 Mood Predictor

 A lightweight, single-page Natural Language Processing (NLP) application that predicts the emotional mood of a given text.

📖 Project Scope & Description

This project is an educational NLP demonstration, not a production-grade machine learning application.

The primary goal of this project is to implement and demonstrate Word Embedding techniques from scratch using the Word2Vec Skip-gram architecture. Because the model is trained on a limited custom vocabulary (rather than massive pre-trained transformer weights), it may struggle with Out-of-Vocabulary (OOV) words. The focus here is strictly on the mechanics of the embedding pipeline, data transformation, and integrating a custom NLP model into a modern web backend.

🚀 Tech Stack

 • Backend API: FastAPI (Python)
 • NLP / Embeddings: Gensim (Word2Vec, Skip-gram architecture)
 • Classification: Scikit-Learn (LogisticRegression)
 • Frontend UI: Vanilla JavaScript, HTML5, and Tailwind CSS (Served dynamically via FastAPI)

🧠 How the NLP Pipeline Works

 • Tokenization: The user's input text is cleaned, lowercased, and split into individual word tokens using regular expressions.
 • Word Embeddings (Skip-gram): The custom-trained Word2Vec model checks its vocabulary for the tokens and converts known words into        100-dimensional numerical vectors.
 • Sentence Vectorization: The application calculates the average of all valid word vectors in the input to create a single, unified mathematical representation of the entire sentence.
 • Classification: A Logistic Regression model evaluates the sentence vector and calculates probability scores across different emotional classes (e.g., Joy, Sadness, Anger, Fear).

🛠️ Local Setup & Installation

1. Clone the repository and navigate to the folder:

 • git clone <your-repo-url>
 • cd MoodPredictor

2. Install dependencies:
 • pip install fastapi uvicorn pydantic scikit-learn gensim numpy datasets

3. Train the models:
 • Generate the .model and .pkl files by running the training script. This downloads the training dataset and builds the Skip-gram and Logistic Regression models.
 • python train_model.py

4. Start the Application:
 • Run the FastAPI server to serve both the API and the user interface.
 • python -m uvicorn app:app --reload


5. View the App:
 • Open your browser and navigate to http://localhost:8000.

📂 Project Structure

 • app.py: The main FastAPI server, API routing, and embedded HTML/JS frontend.
 • train_model.py: The data ingestion and NLP model training script.
 • word2vec_skipgram.model: The saved Gensim vocabulary and embedding weights.
 • mood_classifier.pkl: The saved Logistic Regression classification model.

Author
Deep Solanki — Founder & Lead Full-Stack Developer
portfolio: https://deepsolanki.onrender.com
