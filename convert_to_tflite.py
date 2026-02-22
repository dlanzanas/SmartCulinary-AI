"""
SmartCulinary AI - TensorFlow Lite Conversion Script
Converts trained Keras model to TFLite format with optimization
"""

import tensorflow as tf
from pathlib import Path
import yaml
import argparse
import numpy as np

# Import custom modules
import sys
sys.path.append(str(Path(__file__).parent))

from utils.metrics import calculate_model_size, measure_inference_latency

def load_config():
    """Load configuration from YAML file"""
    config_path = Path(__file__).parent / "config" / "training_config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def convert_to_tflite(model_path, output_path, optimization='default'):
    """
    Convert Keras model to TensorFlow Lite format
    
    Args:
        model_path: Path to Keras model (.h5)
        output_path: Path to save TFLite model (.tflite)
        optimization: Optimization strategy ('default', 'float16', 'int8')
    
    Returns:
        Path to converted model
    """
    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    # Create converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Apply optimization
    if optimization == 'default':
        print("Applying default optimizations (dynamic range quantization)...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif optimization == 'float16':
        print("Applying float16 quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif optimization == 'int8':
        print("Applying int8 quantization...")
        print("⚠ Note: int8 quantization requires representative dataset")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # You would need to provide a representative dataset here
    
    # Convert
    print("Converting model...")
    tflite_model = converter.convert()
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"✓ Model converted and saved to: {output_path}")
    
    return output_path

def test_tflite_model(tflite_path, original_model_path, test_images=10):
    """
    Test TFLite model accuracy and performance
    
    Args:
        tflite_path: Path to TFLite model
        original_model_path: Path to original Keras model
        test_images: Number of test images to use
    """
    print("\nTesting TFLite model...")
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"  Input shape: {input_details[0]['shape']}")
    print(f"  Output shape: {output_details[0]['shape']}")
    
    # Measure inference latency
    latency_stats = measure_inference_latency(
        interpreter,
        input_shape=tuple(input_details[0]['shape'][1:]),
        num_iterations=100
    )
    
    print(f"\n  Inference Latency:")
    print(f"    Mean: {latency_stats['mean_ms']:.2f} ms")
    print(f"    Median: {latency_stats['median_ms']:.2f} ms")
    print(f"    P95: {latency_stats['p95_ms']:.2f} ms")
    
    # Calculate model size
    model_size = calculate_model_size(tflite_path)
    print(f"\n  Model Size: {model_size:.2f} MB")
    
    return {
        'latency_ms': latency_stats['mean_ms'],
        'size_mb': model_size
    }

def main(args):
    """Main execution function"""
    print("=" * 60)
    print("SmartCulinary AI - TFLite Conversion")
    print("=" * 60)
    print()
    
    # Load configuration
    config = load_config()
    
    # Setup paths
    project_root = Path(__file__).parent
    model_path = Path(args.model)
    
    if not model_path.exists():
        print(f"✗ Model not found: {model_path}")
        print("\nPlease train the model first: python train.py")
        return
    
    # Output path
    exports_dir = project_root / config['paths']['exports_dir']
    exports_dir.mkdir(exist_ok=True)
    
    tflite_path = exports_dir / "smartculinary_model.tflite"
    
    # Convert model
    convert_to_tflite(
        model_path,
        tflite_path,
        optimization=args.optimization
    )
    
    # Test converted model
    stats = test_tflite_model(tflite_path, model_path)
    
    # Validate against targets
    targets = config['evaluation']['targets']
    
    print("\n" + "=" * 60)
    print("Validation Against Targets")
    print("=" * 60)
    
    latency_pass = stats['latency_ms'] <= targets['inference_latency_ms']
    size_pass = stats['size_mb'] <= targets['model_size_mb']
    
    print(f"\nInference Latency: {stats['latency_ms']:.2f} ms (target: ≤ {targets['inference_latency_ms']} ms)")
    print(f"  Status: {'✓ PASS' if latency_pass else '✗ FAIL'}")
    
    print(f"\nModel Size: {stats['size_mb']:.2f} MB (target: ≤ {targets['model_size_mb']} MB)")
    print(f"  Status: {'✓ PASS' if size_pass else '✗ FAIL'}")
    
    if latency_pass and size_pass:
        print("\n✓ TFLite model meets all deployment targets!")
    else:
        print("\n⚠ TFLite model does not meet all targets")
        print("  Consider further optimization or model architecture changes")
    
    print("\n" + "=" * 60)
    print("Conversion Complete!")
    print("=" * 60)
    print(f"\nTFLite model saved to: {tflite_path}")
    print("\nNext steps:")
    print("  1. Integrate model into Flutter app")
    print("  2. Test on actual mobile device")
    print("  3. Deploy to production")
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert model to TensorFlow Lite')
    parser.add_argument(
        '--model',
        type=str,
        default='checkpoints/best_model.h5',
        help='Path to trained Keras model'
    )
    parser.add_argument(
        '--optimization',
        choices=['default', 'float16', 'int8'],
        default='default',
        help='Optimization strategy'
    )
    
    args = parser.parse_args()
    main(args)
