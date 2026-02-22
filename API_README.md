# SmartCulinary Backend API

Flask API server for ML inference on web platform.

## Setup

1. **Install dependencies**:
```powershell
pip install -r api_requirements.txt
```

2. **Run the server**:
```powershell
python api_server.py
```

The server will start on `http://localhost:5000`

## API Endpoints

### Health Check
```
GET /health
```
Returns server and model status.

### Predict
```
POST /predict
Content-Type: multipart/form-data
```
Upload an image file and get top-5 predictions.

**Request**:
- `image`: Image file (JPEG, PNG)

**Response**:
```json
{
  "success": true,
  "predictions": [
    {"label": "Apple", "confidence": 0.95},
    {"label": "Pear", "confidence": 0.03},
    ...
  ],
  "top_label": "Apple",
  "top_confidence": 0.95
}
```

### Get Classes
```
GET /classes
```
Returns list of all 81 supported food categories.

## Testing

Test the API with curl:
```powershell
# Health check
curl http://localhost:5000/health

# Prediction (replace with your image path)
curl -X POST -F "image=@path/to/image.jpg" http://localhost:5000/predict
```

## Flutter Web Integration

The Flutter web app (`ml_service_web.dart`) automatically connects to this API at `http://localhost:5000`.

For production, update the API URL in `ml_service_web.dart`:
```dart
static const String _apiUrl = 'https://your-production-api.com';
```

## Deployment

For production deployment:
1. Use a production WSGI server (Gunicorn, uWSGI)
2. Set up HTTPS
3. Configure CORS for your domain
4. Use environment variables for configuration
