# Symptom Checker Chatbot - Web Interface

A web-based medical symptom checker chatbot built with Flask and scikit-learn.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have `dataset.csv` in the project directory.

## Running the Web Interface

Start the Flask server:
```bash
python app.py
```

The web interface will be available at: `http://localhost:5000`

Open your browser and navigate to that URL to use the chatbot.

## Features

- **Modern Web UI**: Beautiful, responsive interface
- **Symptom Analysis**: Enter symptoms in natural language
- **Disease Prediction**: Get possible diagnoses with confidence scores
- **Feature Extraction**: See what information was extracted from your input

## API Endpoints

- `GET /` - Main web interface
- `POST /api/predict` - Submit symptoms and get diagnosis
- `GET /api/health` - Health check endpoint

## Usage

1. Enter your symptoms in the text area (e.g., "I'm a 25 year old male with a high fever, bad cough, and I'm very tired.")
2. Click "Get Diagnosis" or press Ctrl+Enter (Cmd+Enter on Mac)
3. View the results including the most likely condition and other possibilities
4. Click "Start New Diagnosis" to analyze another set of symptoms

## Important Disclaimer

This tool is for educational purposes only and is NOT medical advice. Always consult a qualified healthcare professional for real medical concerns.

