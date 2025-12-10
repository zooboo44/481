from flask import Flask, render_template, request, jsonify
import os
from chatbot import (
    load_data,
    train_model,
    parse_free_text_description,
    predict_diagnosis,
    DATA_PATH
)

app = Flask(__name__)

# Global variables to cache the model and dataset
_model = None
_dataset = None


def get_model():
    """Load and cache the model and dataset."""
    global _model, _dataset
    if _model is None or _dataset is None:
        _dataset = load_data()
        _model = train_model(_dataset)
    return _model, _dataset


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for making predictions."""
    try:
        data = request.get_json()
        user_text = data.get('text', '').strip()
        
        if not user_text:
            return jsonify({'error': 'Please provide a description of your symptoms.'}), 400
        
        # Get model and dataset
        model, df = get_model()
        
        # Parse the user's text
        info = parse_free_text_description(user_text)
        
        # Make prediction
        result = predict_diagnosis(model, info, df)
        
        return jsonify({
            'success': True,
            'best_disease': result['best_disease'],
            'best_prob': result['best_prob'],
            'top_suggestions': result['top_suggestions'],
            'used_features': result['used_features']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    try:
        model, df = get_model()
        return jsonify({
            'status': 'healthy',
            'model_loaded': model is not None,
            'dataset_size': len(df) if df is not None else 0
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500


if __name__ == '__main__':
    # Pre-load the model on startup
    print("Loading dataset and training model...")
    try:
        get_model()
        print("Model ready!")
    except Exception as e:
        print(f"Error loading model: {e}")
    
    # Use debug mode only if FLASK_ENV is not set to production
    debug_mode = os.getenv('FLASK_ENV', 'development') != 'production'
    port = int(os.getenv('PORT', 8085))  # Allow port to be configured via environment variable
    app.run(debug=debug_mode, host='0.0.0.0', port=port)

