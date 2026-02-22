# Converting TensorFlow Model to TensorFlow.js for Web

## Why TensorFlow.js?

The `.tflite` model works great on mobile (Android/iOS) but doesn't work on web because it uses `dart:ffi`. However, we can convert our trained Keras model (`.h5`) to TensorFlow.js format to enable **real ML inference on web**!

## Conversion Steps

### 1. Install TensorFlow.js Converter

```powershell
# Activate Python environment
.\smartculinary_env\Scripts\activate

# Install tensorflowjs
pip install tensorflowjs
```

### 2. Convert Keras Model to TensorFlow.js

```powershell
# Convert the best model
tensorflowjs_converter `
  --input_format=keras `
  --output_format=tfjs_graph_model `
  checkpoints/best_model.h5 `
  smartculinary_app/web/models/tfjs_model

# This creates:
# - model.json (model architecture)
# - group1-shard1of1.bin (weights)
```

### 3. Update Web Assets

The converted model files need to be accessible from the web app:

```
smartculinary_app/
├── web/
│   └── models/
│       └── tfjs_model/
│           ├── model.json
│           └── group1-shard1of1.bin
```

### 4. Load Model in Flutter Web

Update `ml_service_web.dart` to load the TensorFlow.js model:

```dart
import 'package:tensorflow_lite_web/tensorflow_lite_web.dart';

Future<void> loadModel() async {
  // Load TensorFlow.js model
  _model = await tfl.loadGraphModel('models/tfjs_model/model.json');
  _isLoaded = true;
}

Future<Prediction> classifyImage(String imagePath) async {
  // Preprocess image to 224x224x3
  final tensor = await preprocessImage(imagePath);
  
  // Run inference
  final predictions = await _model.predict(tensor);
  
  // Get top 5 results
  return parseTopPredictions(predictions);
}
```

## Benefits

✅ **Real ML inference on web** - No more mock predictions  
✅ **Same model everywhere** - Consistent results across platforms  
✅ **No backend needed** - Client-side inference  
✅ **Fast performance** - GPU acceleration in browser

## Model Size Comparison

- **TFLite (mobile)**: 3.17 MB ✅
- **TensorFlow.js (web)**: ~10-15 MB (includes metadata)
- **Original Keras**: 32.46 MB

## Performance

- **Mobile (TFLite)**: < 200ms target
- **Web (TensorFlow.js)**: 500-1000ms (browser-dependent)
- **Web with WebGL**: 200-400ms (GPU acceleration)

## Next Steps

1. Run the conversion command above
2. Test the web model in browser
3. Update `ml_service_web.dart` with real inference
4. Deploy web app with embedded model

## Alternative: Backend API

If the web model is too large, you can also:
1. Keep mock predictions on web
2. Deploy a Flask/FastAPI backend with the `.h5` model
3. Call the API from the Flutter web app

This trades model size for network latency.
