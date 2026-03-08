"""
SmartCulinary AI - Custom Metrics and Evaluation Functions
"""

import tensorflow as tf
import numpy as np
import time
from pathlib import Path

def top_k_accuracy(y_true, y_pred, k=5):
    """
    Calculate Top-K accuracy
    
    Args:
        y_true: True labels (one-hot encoded)
        y_pred: Predicted probabilities
        k: Number of top predictions to consider
    
    Returns:
        Top-K accuracy score
    """
    # Convert one-hot to class indices
    if len(y_true.shape) > 1:
        y_true_indices = np.argmax(y_true, axis=1)
    else:
        y_true_indices = y_true
    
    # Get top k predictions
    top_k_preds = np.argsort(y_pred, axis=1)[:, -k:]
    
    # Check if true label is in top k
    correct = np.array([y_true_indices[i] in top_k_preds[i] for i in range(len(y_true_indices))])
    
    return np.mean(correct)

def measure_inference_latency(model, input_shape=(224, 224, 3), num_iterations=100, warmup=10):
    """
    Measure model inference latency
    
    Args:
        model: Keras model or TFLite interpreter
        input_shape: Input image shape
        num_iterations: Number of inference iterations
        warmup: Number of warmup iterations
    
    Returns:
        Dictionary with latency statistics
    """
    # Create dummy input
    dummy_input = np.random.rand(1, *input_shape).astype(np.float32)
    
    # Warmup
    for _ in range(warmup):
        if isinstance(model, tf.keras.Model):
            _ = model.predict(dummy_input, verbose=0)
        else:
            # TFLite interpreter
            model.set_tensor(model.get_input_details()[0]['index'], dummy_input)
            model.invoke()
    
    # Measure latency
    latencies = []
    for _ in range(num_iterations):
        start_time = time.perf_counter()
        
        if isinstance(model, tf.keras.Model):
            _ = model.predict(dummy_input, verbose=0)
        else:
            # TFLite interpreter
            model.set_tensor(model.get_input_details()[0]['index'], dummy_input)
            model.invoke()
        
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000)  # Convert to ms
    
    return {
        'mean_ms': np.mean(latencies),
        'std_ms': np.std(latencies),
        'min_ms': np.min(latencies),
        'max_ms': np.max(latencies),
        'median_ms': np.median(latencies),
        'p95_ms': np.percentile(latencies, 95),
        'p99_ms': np.percentile(latencies, 99)
    }

def calculate_model_size(model_path):
    """
    Calculate model file size
    
    Args:
        model_path: Path to model file
    
    Returns:
        Size in MB
    """
    size_bytes = Path(model_path).stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    return size_mb

def validate_success_criteria(metrics, config):
    """
    Validate model against success criteria
    
    Args:
        metrics: Dictionary of evaluation metrics
        config: Configuration dictionary
    
    Returns:
        Dictionary with validation results
    """
    targets = config['evaluation']['targets']
    
    results = {
        'top1_accuracy': {
            'value': metrics.get('top1_accuracy', 0),
            'target': targets['top1_accuracy'],
            'passed': metrics.get('top1_accuracy', 0) >= targets['top1_accuracy']
        },
        'top5_accuracy': {
            'value': metrics.get('top5_accuracy', 0),
            'target': targets['top5_accuracy'],
            'passed': metrics.get('top5_accuracy', 0) >= targets['top5_accuracy']
        },
        'inference_latency': {
            'value': metrics.get('inference_latency_ms', float('inf')),
            'target': targets['inference_latency_ms'],
            'passed': metrics.get('inference_latency_ms', float('inf')) <= targets['inference_latency_ms']
        },
        'model_size': {
            'value': metrics.get('model_size_mb', float('inf')),
            'target': targets['model_size_mb'],
            'passed': metrics.get('model_size_mb', float('inf')) <= targets['model_size_mb']
        }
    }
    
    # Overall pass/fail
    results['all_passed'] = all(r['passed'] for r in results.values() if isinstance(r, dict))
    
    return results

def print_validation_results(validation_results):
    """Print validation results in a formatted table"""
    print("\n" + "=" * 60)
    print("Success Criteria Validation")
    print("=" * 60)
    print()
    
    print(f"{'Metric':<25} {'Value':<15} {'Target':<15} {'Status':<10}")
    print("-" * 60)
    
    for metric_name, result in validation_results.items():
        if metric_name == 'all_passed':
            continue
        
        value = result['value']
        target = result['target']
        passed = result['passed']
        
        # Format value based on metric type
        if 'accuracy' in metric_name:
            value_str = f"{value:.2%}"
            target_str = f"≥ {target:.2%}"
        elif 'latency' in metric_name:
            value_str = f"{value:.2f} ms"
            target_str = f"≤ {target} ms"
        elif 'size' in metric_name:
            value_str = f"{value:.2f} MB"
            target_str = f"≤ {target} MB"
        else:
            value_str = f"{value}"
            target_str = f"{target}"
        
        status = "✓ PASS" if passed else "✗ FAIL"
        status_color = "\033[92m" if passed else "\033[91m"
        reset_color = "\033[0m"
        
        print(f"{metric_name.replace('_', ' ').title():<25} {value_str:<15} {target_str:<15} {status_color}{status}{reset_color}")
    
    print("-" * 60)
    
    if validation_results['all_passed']:
        print("\n✓ All success criteria met!")
    else:
        print("\n✗ Some success criteria not met")
    
    print("=" * 60)

if __name__ == "__main__":
    # Test metrics
    print("Testing custom metrics...")
    
    # Test top-k accuracy
    y_true = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    y_pred = np.array([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1], [0.1, 0.2, 0.7]])
    
    top1 = top_k_accuracy(y_true, y_pred, k=1)
    top5 = top_k_accuracy(y_true, y_pred, k=3)
    
    print(f"Top-1 Accuracy: {top1:.2%}")
    print(f"Top-3 Accuracy: {top5:.2%}")
    
    print("\n✓ Metrics test successful!")
