from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import numpy as np
import re
import pickle
from gensim.models import Word2Vec

# Initialize the API
app = FastAPI(title="Mood Predictor API")
    
# 1. Load Models (Global Scope)
try:
    word2vec_model = Word2Vec.load("word2vec_skipgram.model")
    with open("mood_classifier.pkl", "rb") as f:
        classifier = pickle.load(f)
except Exception as e:
    print(f"Failed to load models: {e}")

# 2. Define Data Structures
class TextInput(BaseModel):
    text: str

def get_sentence_vector(text, model):
    tokens = re.findall(r'\b\w+\b', str(text).lower())
    valid_vectors = [model.wv[word] for word in tokens if word in model.wv]
    if not valid_vectors:
        return np.zeros(model.vector_size)
    return np.mean(valid_vectors, axis=0)

# 3. Create the Prediction Endpoint
@app.post("/api/predict")
def predict_mood(input_data: TextInput):
    if not input_data.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    vectorized_input = get_sentence_vector(input_data.text, word2vec_model).reshape(1, -1)
    
    prediction = classifier.predict(vectorized_input)[0]
    probabilities = classifier.predict_proba(vectorized_input)[0]
    
    # Package the scores cleanly
    confidence_scores = [
        {"mood": mood, "score": round(float(prob) * 100, 1)}
        for mood, prob in zip(classifier.classes_, probabilities)
    ]

    # Sort so the highest score is first
    confidence_scores = sorted(confidence_scores, key=lambda x: x['score'], reverse=True)

    return {
        "prediction": prediction.upper(),
        "confidence": confidence_scores
    }

# 4. Serve the Frontend (Single Page UI)
@app.get("/", response_class=HTMLResponse)
def serve_ui():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Mood Predictor</title>
        <!-- Use Tailwind CSS via CDN for instant modern styling -->
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-gray-50 min-h-screen flex items-center justify-center p-4 font-sans">

        <div class="max-w-md w-full bg-white rounded-2xl shadow-xl p-8 border border-gray-100">
            <div class="text-center mb-6">
                <div class="text-4xl mb-3">🎭</div>
                <h1 class="text-2xl font-bold text-gray-800 mb-2">Mood Predictor</h1>
                <p class="text-gray-500 text-sm">Enter a situation or feeling, and the AI will analyze your mood.</p>
            </div>

            <textarea 
                id="userInput" 
                class="w-full h-32 p-4 border border-gray-200 rounded-xl focus:ring-2 focus:ring-indigo-500 focus:border-transparent outline-none resize-none mb-4 transition-all"
                placeholder="e.g., I feel like I have to make the suffering stop..."></textarea>

            <p id="errorMessage" class="text-red-500 text-sm mb-4 hidden"></p>

            <button 
                id="predictBtn" 
                onclick="makePrediction()"
                class="w-full bg-indigo-600 text-white font-semibold py-3 rounded-xl hover:bg-indigo-700 transition-colors flex justify-center items-center gap-2">
                <span>Predict Mood</span>
                <svg id="loadingSpinner" class="animate-spin h-5 w-5 hidden" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                    <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
            </button>

            <!-- Results Section (Hidden by default) -->
            <div id="resultsArea" class="mt-8 pt-6 border-t border-gray-100 hidden">
                <div class="text-center mb-6">
                    <p class="text-sm text-gray-500 uppercase tracking-wide font-semibold mb-1">Predicted Mood</p>
                    <h2 id="predictionText" class="text-3xl font-bold text-indigo-600"></h2>
                </div>
                
                <div id="confidenceBars" class="space-y-4">
                    <!-- Bars injected via JavaScript -->
                </div>
            </div>
        </div>

        <script>
            async function makePrediction() {
                const textInput = document.getElementById('userInput').value;
                const btn = document.getElementById('predictBtn');
                const spinner = document.getElementById('loadingSpinner');
                const errorMsg = document.getElementById('errorMessage');
                const resultsArea = document.getElementById('resultsArea');
                const barsContainer = document.getElementById('confidenceBars');
                
                // Reset states
                errorMsg.classList.add('hidden');
                resultsArea.classList.add('hidden');
                barsContainer.innerHTML = '';

                if (!textInput.trim()) {
                    errorMsg.textContent = "Please enter some text first.";
                    errorMsg.classList.remove('hidden');
                    return;
                }

                // Show loading state
                btn.disabled = true;
                btn.classList.add('opacity-75', 'cursor-not-allowed');
                spinner.classList.remove('hidden');

                try {
                    const response = await fetch('/api/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ text: textInput })
                    });

                    if (!response.ok) throw new Error("API failed");

                    const data = await response.json();
                    
                    // Display results
                    document.getElementById('predictionText').textContent = data.prediction;
                    
                    data.confidence.forEach(item => {
                        const barHtml = `
                            <div>
                                <div class="flex justify-between text-sm font-medium mb-1 text-gray-700">
                                    <span class="capitalize">${item.mood.toLowerCase()}</span>
                                    <span>${item.score}%</span>
                                </div>
                                <div class="w-full bg-gray-100 rounded-full h-2">
                                    <div class="bg-indigo-500 h-2 rounded-full transition-all duration-1000" style="width: 0%"></div>
                                </div>
                            </div>
                        `;
                        barsContainer.insertAdjacentHTML('beforeend', barHtml);
                    });

                    resultsArea.classList.remove('hidden');
                    
                    // Animate bars after a tiny delay for smooth UI
                    setTimeout(() => {
                        const bars = barsContainer.querySelectorAll('.bg-indigo-500');
                        bars.forEach((bar, index) => {
                            bar.style.width = `${data.confidence[index].score}%`;
                        });
                    }, 50);

                } catch (err) {
                    errorMsg.textContent = "Failed to connect to the prediction server.";
                    errorMsg.classList.remove('hidden');
                } finally {
                    // Remove loading state
                    btn.disabled = false;
                    btn.classList.remove('opacity-75', 'cursor-not-allowed');
                    spinner.classList.add('hidden');
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)