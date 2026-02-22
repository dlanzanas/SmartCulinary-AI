"""
SmartCulinary AI - Model Evaluation Script
Evaluates trained model on test set and validates against success criteria
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from pathlib import Path
import yaml
import argparse
from sklearn.metrics import confusion_matrix, classification_report
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Import custom modules
import sys
sys.path.append(str(Path(__file__).parent))

from data.data_preprocessing import create_datasets
from utils.metrics import (
    top_k_accuracy, 
    measure_inference_latency, 
    calculate_model_size,
    validate_success_criteria,
    print_validation_results
)
from utils.visualization import (
    plot_confusion_matrix,
    plot_per_class_accuracy,
    plot_prediction_samples
)

def load_config():
    """Load configuration from YAML file"""
    config_path = Path(__file__).parent / "config" / "training_config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def evaluate_model(model, test_ds, class_names):
    """
    Evaluate model on test dataset
    
    Returns:
        Dictionary with evaluation metrics
    """
    print("Evaluating model on test set...")
    
    # Get predictions
    y_true = []
    y_pred = []
    y_pred_probs = []
    
    for images, labels in test_ds:
        predictions = model.predict(images, verbose=0)
        y_pred_probs.extend(predictions)
        y_pred.extend(np.argmax(predictions, axis=1))
        y_true.extend(np.argmax(labels.numpy(), axis=1))
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_probs = np.array(y_pred_probs)
    
    # Calculate metrics
    top1_acc = np.mean(y_true == y_pred)
    top5_acc = top_k_accuracy(
        tf.keras.utils.to_categorical(y_true, len(class_names)),
        y_pred_probs,
        k=5
    )
    
    print(f"✓ Evaluation complete")
    print(f"  Top-1 Accuracy: {top1_acc:.4f} ({top1_acc*100:.2f}%)")
    print(f"  Top-5 Accuracy: {top5_acc:.4f} ({top5_acc*100:.2f}%)")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Per-class accuracy
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    
    return {
        'top1_accuracy': top1_acc,
        'top5_accuracy': top5_acc,
        'confusion_matrix': cm,
        'class_accuracies': class_accuracies,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_pred_probs': y_pred_probs
    }

def generate_classification_report(y_true, y_pred, class_names, save_path=None):
    """Generate and save classification report"""
    report = classification_report(
        y_true, 
        y_pred, 
        target_names=class_names,
        digits=4
    )
    
    print("\nClassification Report:")
    print("=" * 60)
    print(report)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write("Classification Report\n")
            f.write("=" * 60 + "\n\n")
            f.write(report)
        print(f"✓ Classification report saved to: {save_path}")
    
    return report

def plot_training_history(checkpoints_dir, save_dir):
    """
    Load and plot training history from JSON files
    
    Args:
        checkpoints_dir: Directory containing training_history_*.json files
        save_dir: Directory to save plots
    """
    print("\nGenerating training history visualizations...")
    
    checkpoints_path = Path(checkpoints_dir)
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Find history files
    phase1_file = checkpoints_path / "training_history_phase1.json"
    phase2_file = checkpoints_path / "training_history_phase2.json"
    
    if not phase1_file.exists() and not phase2_file.exists():
        print("  ⚠ No training history files found")
        return
    
    # Load histories
    histories = {}
    if phase1_file.exists():
        with open(phase1_file, 'r') as f:
            histories['Phase 1'] = json.load(f)
    if phase2_file.exists():
        with open(phase2_file, 'r') as f:
            histories['Phase 2'] = json.load(f)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training History', fontsize=16, fontweight='bold')
    
    colors = {'Phase 1': '#1f77b4', 'Phase 2': '#ff7f0e'}
    
    # Plot 1: Training & Validation Loss
    ax = axes[0, 0]
    for phase_name, history in histories.items():
        epochs = range(1, len(history['loss']) + 1)
        ax.plot(epochs, history['loss'], label=f'{phase_name} Train', 
                color=colors[phase_name], linestyle='-', linewidth=2)
        ax.plot(epochs, history['val_loss'], label=f'{phase_name} Val', 
                color=colors[phase_name], linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Loss', fontweight='bold')
    ax.set_title('Loss Curves', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Training & Validation Accuracy
    ax = axes[0, 1]
    for phase_name, history in histories.items():
        epochs = range(1, len(history['accuracy']) + 1)
        ax.plot(epochs, [acc * 100 for acc in history['accuracy']], 
                label=f'{phase_name} Train', color=colors[phase_name], 
                linestyle='-', linewidth=2)
        ax.plot(epochs, [acc * 100 for acc in history['val_accuracy']], 
                label=f'{phase_name} Val', color=colors[phase_name], 
                linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Accuracy Curves', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Top-5 Accuracy
    ax = axes[1, 0]
    for phase_name, history in histories.items():
        if 'top_5_accuracy' in history:
            epochs = range(1, len(history['top_5_accuracy']) + 1)
            ax.plot(epochs, [acc * 100 for acc in history['top_5_accuracy']], 
                    label=f'{phase_name} Train', color=colors[phase_name], 
                    linestyle='-', linewidth=2)
            ax.plot(epochs, [acc * 100 for acc in history['val_top_5_accuracy']], 
                    label=f'{phase_name} Val', color=colors[phase_name], 
                    linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Top-5 Accuracy (%)', fontweight='bold')
    ax.set_title('Top-5 Accuracy Curves', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Learning Rate
    ax = axes[1, 1]
    for phase_name, history in histories.items():
        if 'lr' in history:
            epochs = range(1, len(history['lr']) + 1)
            ax.plot(epochs, history['lr'], label=phase_name, 
                    color=colors[phase_name], linewidth=2, marker='o', markersize=3)
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Learning Rate', fontweight='bold')
    ax.set_title('Learning Rate Schedule', fontweight='bold')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = save_path / "training_history.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Training history plot saved to: {plot_path}")

def main(args):
    """Main evaluation function"""
    print("=" * 60)
    print("SmartCulinary AI - Model Evaluation")
    print("=" * 60)
    print()
    
    # Load configuration
    config = load_config()
    
    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"✗ Model not found: {model_path}")
        print("\nPlease train the model first: python train.py")
        return
    
    print(f"Loading model from: {model_path}")
    model = keras.models.load_model(model_path)
    print("✓ Model loaded successfully")
    print()
    
    # Load test dataset
    print("Loading test dataset...")
    _, _, test_ds, class_names = create_datasets()
    print()
    
    # Evaluate model
    eval_results = evaluate_model(model, test_ds, class_names)
    
    # Measure inference latency
    print("\nMeasuring inference latency...")
    latency_stats = measure_inference_latency(
        model,
        input_shape=tuple(config['dataset']['image_size']) + (3,),
        num_iterations=100
    )
    
    print(f"✓ Latency measurement complete")
    print(f"  Mean: {latency_stats['mean_ms']:.2f} ms")
    print(f"  Median: {latency_stats['median_ms']:.2f} ms")
    print(f"  P95: {latency_stats['p95_ms']:.2f} ms")
    print(f"  P99: {latency_stats['p99_ms']:.2f} ms")
    
    # Calculate model size
    model_size = calculate_model_size(model_path)
    print(f"\nModel size: {model_size:.2f} MB")
    
    # Validate against success criteria
    metrics = {
        'top1_accuracy': eval_results['top1_accuracy'],
        'top5_accuracy': eval_results['top5_accuracy'],
        'inference_latency_ms': latency_stats['mean_ms'],
        'model_size_mb': model_size
    }
    
    validation_results = validate_success_criteria(metrics, config)
    print_validation_results(validation_results)
    
    # Generate classification report
    project_root = Path(__file__).parent
    reports_dir = project_root / "reports"
    reports_dir.mkdir(exist_ok=True)
    
    generate_classification_report(
        eval_results['y_true'],
        eval_results['y_pred'],
        class_names,
        save_path=reports_dir / "classification_report.txt"
    )
    
    # Save evaluation results - convert all numpy types to native Python types
    def convert_to_native(obj):
        """Recursively convert numpy types to native Python types"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_native(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        else:
            return obj
    
    results_dict = {
        'top1_accuracy': float(eval_results['top1_accuracy']),
        'top5_accuracy': float(eval_results['top5_accuracy']),
        'inference_latency_ms': float(latency_stats['mean_ms']),
        'model_size_mb': float(model_size),
        'validation_results': convert_to_native(validation_results),
        'per_class_accuracy': {
            class_names[i]: float(acc) 
            for i, acc in enumerate(eval_results['class_accuracies'])
        }
    }
    
    results_path = reports_dir / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"\n✓ Evaluation results saved to: {results_path}")
    
    # Generate visualizations
    if args.visualize:
        print("\nGenerating visualizations...")
        viz_dir = project_root / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # Confusion matrix
        plot_confusion_matrix(
            eval_results['confusion_matrix'],
            class_names,
            save_path=viz_dir / "confusion_matrix.png",
            normalize=True
        )
        
        # Per-class accuracy
        plot_per_class_accuracy(
            eval_results['class_accuracies'],
            class_names,
            save_path=viz_dir / "per_class_accuracy.png"
        )
        
        # Training history plots
        plot_training_history(
            project_root / "checkpoints",
            viz_dir
        )
        
        print(f"✓ Visualizations saved to: {viz_dir}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)
    print(f"\nResults saved to: {reports_dir}")
    
    if validation_results['all_passed']:
        print("\n✓ Model meets all success criteria!")
        print("\nNext step: Convert to TFLite")
        print("  python convert_to_tflite.py")
    else:
        print("\n⚠ Model does not meet all success criteria")
        print("  Consider retraining with different hyperparameters")
    
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate SmartCulinary AI model')
    parser.add_argument(
        '--model',
        type=str,
        default='checkpoints/best_model.h5',
        help='Path to trained model'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualization plots'
    )
    
    args = parser.parse_args()
    main(args)
