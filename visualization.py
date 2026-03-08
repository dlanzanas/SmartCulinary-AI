"""
SmartCulinary AI - Visualization Utilities
Generates plots for training history, confusion matrix, and predictions
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import json

def plot_training_history(history_path, save_path=None):
    """
    Plot training history (loss and accuracy curves)
    
    Args:
        history_path: Path to training history JSON file
        save_path: Path to save the plot (optional)
    """
    # Load history
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Loss
    axes[0].plot(history['loss'], label='Training Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Plot 2: Accuracy
    axes[1].plot(history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[1].plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # Plot 3: Top-5 Accuracy
    if 'top_5_accuracy' in history:
        axes[2].plot(history['top_5_accuracy'], label='Training Top-5', linewidth=2)
        axes[2].plot(history['val_top_5_accuracy'], label='Validation Top-5', linewidth=2)
        axes[2].set_xlabel('Epoch', fontsize=12)
        axes[2].set_ylabel('Top-5 Accuracy', fontsize=12)
        axes[2].set_title('Top-5 Accuracy', fontsize=14, fontweight='bold')
        axes[2].legend()
        axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training history plot saved to: {save_path}")
    
    plt.show()

def plot_confusion_matrix(cm, class_names, save_path=None, normalize=False):
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix (numpy array)
        class_names: List of class names
        save_path: Path to save the plot (optional)
        normalize: Whether to normalize the confusion matrix
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    plt.figure(figsize=(20, 18))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
    )
    
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to: {save_path}")
    
    plt.show()

def plot_per_class_accuracy(class_accuracies, class_names, save_path=None):
    """
    Plot per-class accuracy as a bar chart
    
    Args:
        class_accuracies: Dictionary or array of per-class accuracies
        class_names: List of class names
        save_path: Path to save the plot (optional)
    """
    if isinstance(class_accuracies, dict):
        accuracies = [class_accuracies[name] for name in class_names]
    else:
        accuracies = class_accuracies
    
    # Sort by accuracy
    sorted_indices = np.argsort(accuracies)
    sorted_names = [class_names[i] for i in sorted_indices]
    sorted_accuracies = [accuracies[i] for i in sorted_indices]
    
    # Create bar plot
    plt.figure(figsize=(12, 10))
    colors = ['red' if acc < 0.7 else 'orange' if acc < 0.85 else 'green' for acc in sorted_accuracies]
    
    plt.barh(sorted_names, sorted_accuracies, color=colors, alpha=0.7)
    plt.xlabel('Accuracy', fontsize=12)
    plt.ylabel('Class', fontsize=12)
    plt.title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    plt.xlim(0, 1.0)
    plt.grid(axis='x', alpha=0.3)
    
    # Add accuracy values
    for i, (name, acc) in enumerate(zip(sorted_names, sorted_accuracies)):
        plt.text(acc + 0.01, i, f'{acc:.2%}', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Per-class accuracy plot saved to: {save_path}")
    
    plt.show()

def plot_prediction_samples(images, true_labels, pred_labels, class_names, 
                           confidences=None, num_samples=10, save_path=None):
    """
    Visualize sample predictions
    
    Args:
        images: Array of images
        true_labels: True class indices
        pred_labels: Predicted class indices
        class_names: List of class names
        confidences: Prediction confidences (optional)
        num_samples: Number of samples to display
        save_path: Path to save the plot (optional)
    """
    num_samples = min(num_samples, len(images))
    
    # Select random samples
    indices = np.random.choice(len(images), num_samples, replace=False)
    
    # Create grid
    cols = 5
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    for i, idx in enumerate(indices):
        if i >= len(axes):
            break
        
        # Display image
        axes[i].imshow(images[idx])
        axes[i].axis('off')
        
        # Create title
        true_name = class_names[true_labels[idx]]
        pred_name = class_names[pred_labels[idx]]
        
        if confidences is not None:
            conf = confidences[idx]
            title = f"True: {true_name}\nPred: {pred_name}\nConf: {conf:.2%}"
        else:
            title = f"True: {true_name}\nPred: {pred_name}"
        
        # Color based on correctness
        color = 'green' if true_labels[idx] == pred_labels[idx] else 'red'
        axes[i].set_title(title, fontsize=9, color=color, fontweight='bold')
    
    # Hide unused subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Sample Predictions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Prediction samples saved to: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    print("Visualization utilities module loaded successfully!")
