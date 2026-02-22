"""
Flask API for SmartCulinary ML Inference
Serves predictions for Flutter web app
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for Flutter web app

# Load the trained model
MODEL_PATH = 'checkpoints/best_model.h5'
model = None

# Class names (81 food categories)
CLASS_NAMES = [
    'Apple', 'Apricot', 'Avocado', 'Banana', 'Beetroot', 'Blueberry', 'Cactus fruit',
    'Cantaloupe', 'Carambola', 'Cauliflower', 'Cherry', 'Chestnut', 'Clementine',
    'Cocos', 'Corn', 'Cucumber', 'Dates', 'Eggplant', 'Ginger Root', 'Granadilla',
    'Grape', 'Grapefruit', 'Guava', 'Hazelnut', 'Huckleberry', 'Kaki', 'Kiwi',
    'Kohlrabi', 'Kumquats', 'Lemon', 'Limes', 'Lychee', 'Mandarine', 'Mango',
    'Mangostan', 'Maracuja', 'Melon Piel de Sapo', 'Mulberry', 'Nectarine', 'Nut',
    'Onion', 'Orange', 'Papaya', 'Passion Fruit', 'Peach', 'Pear', 'Pepper',
    'Physalis', 'Pineapple', 'Pitahaya Red', 'Plum', 'Pomegranate', 'Pomelo',
    'Potato', 'Quince', 'Rambutan', 'Raspberry', 'Redcurrant', 'Salak', 'Strawberry',
    'Tamarillo', 'Tangelo', 'Tomato', 'Walnut', 'Watermelon'
]

def load_model():
    """Load the Keras model"""
    global model
    if model is None:
        print(f"Loading model from {MODEL_PATH}...")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully!")
    return model

def preprocess_image(image_bytes):
    """Preprocess image for model inference"""
    # Load image from bytes
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to model input size (100x100)
    image = image.resize((100, 100))
    
    # Convert to numpy array and normalize
    img_array = np.array(image, dtype=np.float32)
    img_array = img_array / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint
    Expects: multipart/form-data with 'image' file
    Returns: JSON with top-5 predictions
    """
    try:
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        # Read image bytes
        image_bytes = file.read()
        
        # Preprocess image
        img_array = preprocess_image(image_bytes)
        
        # Load model if not loaded
        model = load_model()
        
        # Run inference
        predictions = model.predict(img_array, verbose=0)[0]
        
        # Get top 5 predictions
        top_5_indices = np.argsort(predictions)[-5:][::-1]
        
        results = []
        for idx in top_5_indices:
            results.append({
                'label': CLASS_NAMES[idx],
                'confidence': float(predictions[idx])
            })
        
        return jsonify({
            'success': True,
            'predictions': results,
            'top_label': results[0]['label'],
            'top_confidence': results[0]['confidence']
        })
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/classes', methods=['GET'])
def get_classes():
    """Return list of all supported classes"""
    return jsonify({
        'classes': CLASS_NAMES,
        'count': len(CLASS_NAMES)
    })

if __name__ == '__main__':
    # Load model on startup
    load_model()
    
    # Run Flask app
    print("\n🚀 SmartCulinary API Server")
    print("=" * 50)
    print("Endpoints:")
    print("  GET  /health   - Health check")
    print("  POST /predict  - Image prediction")
    print("  GET  /classes  - List all classes")
    print("=" * 50)
    print("\nStarting server on http://localhost:5000\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
