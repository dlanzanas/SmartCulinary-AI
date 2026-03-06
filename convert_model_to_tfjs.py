"""
Convert Keras model to TensorFlow.js format for web deployment
"""
import os
import sys

def convert_to_tfjs():
    """Convert the trained Keras model to TensorFlow.js format"""
    
    # Import tensorflowjs converter
    import tensorflowjs as tfjs
    print(f"TensorFlow.js version: {tfjs.__version__}")
    
    # Paths
    keras_model_path = "checkpoints/best_model.h5"
    tfjs_output_path = "smartculinary_app/web/models/tfjs_model"
    
    # Check if input model exists
    if not os.path.exists(keras_model_path):
        print(f"Error: Model not found at {keras_model_path}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(tfjs_output_path, exist_ok=True)
    
    print(f"Converting {keras_model_path} to TensorFlow.js format...")
    print(f"Output directory: {tfjs_output_path}")
    
    # Convert using the Python API
    tfjs.converters.convert_tf_keras_model(
        keras_model_path,
        tfjs_output_path,
        quantization_dtype_map=None,  # No quantization for now
        skip_op_check=False,
        strip_debug_ops=True
    )
    
    print("\n✅ Conversion successful!")
    print(f"\nGenerated files in {tfjs_output_path}:")
    for file in os.listdir(tfjs_output_path):
        file_path = os.path.join(tfjs_output_path, file)
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"  - {file} ({size_mb:.2f} MB)")
    
    print("\n📝 Next steps:")
    print("1. Update ml_service_web.dart to load the model")
    print("2. Test the web app: flutter run -d chrome")

if __name__ == "__main__":
    convert_to_tfjs()
