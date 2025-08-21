from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
import os

# --- IMPORTANT: How to create the 'model.pkl' file ---
# You need to run this part once to create the file before running the Flask app.
# Make sure you have scikit-learn and numpy installed: pip install scikit-learn numpy
# After running, a file named 'model.pkl' will be in the same directory.
"""
from sklearn.linear_model import LinearRegression
import numpy as np
import pickle

# Create a dummy dataset for demonstration
# X = Years of Experience, y = Salary
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y = np.array([30000, 35000, 45000, 50000, 60000, 65000, 75000, 80000, 90000, 95000])

# Create and train the linear regression model
model = LinearRegression()
model.fit(X, y)

# Save the trained model to a file using pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("model.pkl created successfully!")
"""

# --- Flask App Code ---
app = Flask(__name__)

# Check if the model file exists
if not os.path.exists('model.pkl'):
    print("Error: 'model.pkl' not found. Please run the code above to create it first.")
    # Exit or handle the error gracefully
    # For this example, we'll just print a message and the app will likely fail to load the model.

# Load the trained model from the file
try:
    with open('SLR_Model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully.")
except FileNotFoundError:
    model = None
    print("Model file not found. The app will not be able to make predictions.")


# Route to serve the HTML page
@app.route('/')
def home():
    # render_template looks for the file in a 'templates' folder by default.
    # For simplicity, we can just return the raw HTML string.
    # A better approach for larger apps is to put index.html in a 'templates' folder.
    # However, since you're providing both codes, we'll assume they're in the same folder.
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Salary Predictor</title>
        <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap">
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            body {
                font-family: 'Inter', sans-serif;
                background-color: #f3f4f6;
            }
        </style>
    </head>
    <body class="bg-gray-100 flex items-center justify-center min-h-screen p-4">

        <div class="bg-white p-8 rounded-2xl shadow-xl w-full max-w-md border border-gray-200">
            <h1 class="text-3xl font-bold text-gray-800 text-center mb-6">Salary Predictor</h1>
            <p class="text-center text-gray-600 mb-8">
                Enter the years of experience to predict the salary.
            </p>
            <form id="prediction-form" class="space-y-6">
                <div>
                    <label for="experience" class="block text-sm font-medium text-gray-700 mb-1">
                        Years of Experience:
                    </label>
                    <input type="number" id="experience" name="experience" min="0" step="0.1" required
                           class="mt-1 block w-full px-4 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm transition duration-150 ease-in-out">
                </div>
                <button type="submit"
                        class="w-full flex justify-center py-3 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition duration-150 ease-in-out">
                    Predict Salary
                </button>
            </form>
            <div id="result-container" class="mt-8 pt-6 border-t border-gray-200 hidden">
                <h2 class="text-xl font-semibold text-gray-700 mb-3">Predicted Salary:</h2>
                <p id="prediction-result" class="text-2xl font-bold text-green-600 text-center">
                </p>
            </div>
            <div id="message-box" class="mt-4 hidden p-4 rounded-md text-sm" role="alert">
                <p id="message-text"></p>
            </div>
        </div>
        <script>
            const form = document.getElementById('prediction-form');
            const resultContainer = document.getElementById('result-container');
            const predictionResult = document.getElementById('prediction-result');
            const messageBox = document.getElementById('message-box');
            const messageText = document.getElementById('message-text');

            function showMessage(text, type = 'info') {
                messageBox.classList.remove('hidden', 'bg-red-100', 'text-red-700', 'bg-green-100', 'text-green-700', 'bg-blue-100', 'text-blue-700');
                messageBox.classList.add('block');
                messageText.textContent = text;
                if (type === 'error') {
                    messageBox.classList.add('bg-red-100', 'text-red-700');
                } else if (type === 'success') {
                    messageBox.classList.add('bg-green-100', 'text-green-700');
                } else {
                    messageBox.classList.add('bg-blue-100', 'text-blue-700');
                }
            }
            function hideMessage() {
                messageBox.classList.add('hidden');
            }

            form.addEventListener('submit', async function(event) {
                event.preventDefault();
                hideMessage();
                resultContainer.classList.add('hidden');

                const experience = document.getElementById('experience').value;

                if (!experience) {
                    showMessage("Please enter a value for years of experience.", "error");
                    return;
                }
                showMessage("Predicting...", "info");

                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ experience: parseFloat(experience) })
                    });
                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                    const data = await response.json();
                    predictionResult.textContent = `$${Math.round(data.prediction).toLocaleString()}`;
                    resultContainer.classList.remove('hidden');
                    hideMessage();
                } catch (error) {
                    console.error('Error:', error);
                    showMessage("Prediction failed. Please try again.", "error");
                }
            });
        </script>
    </body>
    </html>
    """
    return html_content


# Route to handle prediction requests from the HTML form
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Check server logs.'}), 500

    # Get the JSON data sent from the HTML page
    data = request.get_json(force=True)

    # Extract the 'experience' value
    years_of_experience = data['experience']

    # The model expects a 2D array, so we reshape the input
    input_data = np.array([[years_of_experience]])

    # Make the prediction
    prediction = model.predict(input_data)

    # Convert the prediction to a standard Python float before sending it back
    predicted_salary = float(prediction[0])

    # Return the prediction as a JSON response
    return jsonify(prediction=predicted_salary)


# Main entry point to run the app
if __name__ == '__main__':
    # Use 0.0.0.0 to make the server accessible from outside the local machine
    # during development. In a live environment, this might be handled by a proxy.
    app.run(host='0.0.0.0', port=5000)

