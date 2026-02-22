# SmartCulinary AI - Backend API Setup

## Overview

Instead of converting the model to TensorFlow.js (which has compatibility issues), we're using a **Flask API backend** to serve ML predictions. This approach is:

✅ **More reliable** - No browser compatibility issues  
✅ **Better performance** - Server-side GPU acceleration  
✅ **Production-ready** - Easy to deploy and scale  
✅ **Simpler** - No model conversion needed  

## Architecture

```
Flutter Web App → HTTP Request → Flask API → TensorFlow Model → Predictions
```

## Setup Instructions

### 1. Install Dependencies

```powershell
# In your virtual environment
pip install flask flask-cors pillow
```

### 2. Start the API Server

```powershell
# Run the Flask API
python api_server.py
```

The API will start on `http://localhost:5000`

### 3. Test the API

```powershell
# Health check
curl http://localhost:5000/health

# Test prediction (with an image file)
curl -X POST -F "image=@path/to/image.jpg" http://localhost:5000/predict
```

### 4. Run Flutter Web App

```powershell
cd smartculinary_app
flutter run -d chrome
```

The web app will automatically connect to the API and use real ML predictions!

## API Endpoints

### GET /health
Check if the API and model are loaded

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### POST /predict
Classify a food image

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: `image` file

**Response:**
```json
{
  "success": true,
  "top_label": "avocado",
  "top_confidence": 0.87,
  "predictions": [
    {"label": "avocado", "confidence": 0.87},
    {"label": "tomato", "confidence": 0.45},
    ...
  ]
}
```

## Deployment Options

### Option 1: Local Development
- Run API on localhost:5000
- Web app connects to localhost

### Option 2: Cloud Deployment
Deploy the Flask API to:
- **Heroku** (free tier available)
- **Google Cloud Run** (serverless)
- **AWS Lambda** (with API Gateway)
- **Azure App Service**

Update `ml_service_web.dart` with your deployed API URL:
```dart
static const String apiUrl = 'https://your-api.herokuapp.com';
```

## Advantages Over TensorFlow.js

| Feature | TensorFlow.js | Flask API |
|---------|---------------|-----------|
| Model Conversion | Required (complex) | Not needed ✅ |
| Browser Compatibility | Limited | Universal ✅ |
| Model Size | 10-15 MB download | Server-side ✅ |
| Performance | Browser-dependent | Consistent ✅ |
| GPU Acceleration | WebGL only | Full CUDA support ✅ |
| Deployment | Complex | Simple ✅ |

## Fallback Behavior

The web app has built-in fallback:
1. Try to connect to API
2. If API unavailable → use mock predictions
3. User can still test UI without backend

This makes development easier and provides graceful degradation.
