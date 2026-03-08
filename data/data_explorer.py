"""
SmartCulinary AI - Dataset Explorer
Visualizes and analyzes the dataset
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import yaml
from collections import Counter
import tensorflow as tf

def load_config():
    """Load configuration from YAML file"""
    config_path = Path(__file__).parent.parent / "config" / "training_config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def count_images_per_class(dataset_dir):
    """Count images in each class for each split"""
    splits = ['train', 'val', 'test']
    class_counts = {}
    
    for split in splits:
        split_dir = Path(dataset_dir) / split
        if not split_dir.exists():
            continue
        
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            if class_name not in class_counts:
                class_counts[class_name] = {'train': 0, 'val': 0, 'test': 0}
            
            image_count = len(list(class_dir.glob('*.jpg'))) + len(list(class_dir.glob('*.png')))
            class_counts[class_name][split] = image_count
    
    return class_counts

def plot_class_distribution(class_counts, save_path=None):
    """Plot class distribution across splits"""
    classes = sorted(class_counts.keys())
    train_counts = [class_counts[c]['train'] for c in classes]
    val_counts = [class_counts[c]['val'] for c in classes]
    test_counts = [class_counts[c]['test'] for c in classes]
    
    x = np.arange(len(classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.bar(x - width, train_counts, width, label='Train', alpha=0.8)
    ax.bar(x, val_counts, width, label='Validation', alpha=0.8)
    ax.bar(x + width, test_counts, width, label='Test', alpha=0.8)
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Number of Images', fontsize=12)
    ax.set_title('Dataset Distribution Across Classes', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Class distribution plot saved to: {save_path}")
    
    plt.show()

def visualize_sample_images(dataset_dir, num_samples=5, save_path=None):
    """Visualize sample images from each class"""
    train_dir = Path(dataset_dir) / 'train'
    
    if not train_dir.exists():
        print(f"✗ Training directory not found: {train_dir}")
        return
    
    class_dirs = sorted([d for d in train_dir.iterdir() if d.is_dir()])
    
    # Limit to first 10 classes for visualization
    class_dirs = class_dirs[:10]
    
    fig, axes = plt.subplots(len(class_dirs), num_samples, figsize=(15, 3 * len(class_dirs)))
    
    if len(class_dirs) == 1:
        axes = axes.reshape(1, -1)
    
    for i, class_dir in enumerate(class_dirs):
        class_name = class_dir.name
        image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
        
        # Sample random images
        sample_images = np.random.choice(image_files, min(num_samples, len(image_files)), replace=False)
        
        for j, img_path in enumerate(sample_images):
            img = plt.imread(img_path)
            axes[i, j].imshow(img)
            axes[i, j].axis('off')
            
            if j == 0:
                axes[i, j].set_title(f"{class_name}\n({len(image_files)} images)", 
                                    fontsize=10, fontweight='bold')
    
    plt.suptitle('Sample Images from Dataset (First 10 Classes)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Sample images plot saved to: {save_path}")
    
    plt.show()

def visualize_augmentation(dataset_dir, save_path=None):
    """Visualize data augmentation effects"""
    from data_preprocessing import DataPreprocessor, load_config
    
    config = load_config()
    preprocessor = DataPreprocessor(config)
    
    # Load a single image
    train_dir = Path(dataset_dir) / 'train'
    class_dirs = [d for d in train_dir.iterdir() if d.is_dir()]
    
    if not class_dirs:
        print("✗ No class directories found")
        return
    
    # Get first image from first class
    first_class = class_dirs[0]
    image_files = list(first_class.glob('*.jpg')) + list(first_class.glob('*.png'))
    
    if not image_files:
        print("✗ No images found")
        return
    
    img_path = image_files[0]
    img = tf.keras.utils.load_img(img_path, target_size=tuple(config['dataset']['image_size']))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    
    # Create augmentation layer
    augmentation = preprocessor.create_augmentation_layer()
    
    # Generate augmented versions
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    axes = axes.flatten()
    
    # Original image
    axes[0].imshow(img)
    axes[0].set_title('Original', fontweight='bold')
    axes[0].axis('off')
    
    # Augmented versions
    for i in range(1, 8):
        augmented = augmentation(img_array, training=True)
        axes[i].imshow(augmented[0].numpy().astype("uint8"))
        axes[i].set_title(f'Augmented {i}')
        axes[i].axis('off')
    
    plt.suptitle(f'Data Augmentation Examples\nClass: {first_class.name}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Augmentation visualization saved to: {save_path}")
    
    plt.show()

def print_dataset_summary(class_counts):
    """Print detailed dataset summary"""
    print("\n" + "=" * 60)
    print("Dataset Summary")
    print("=" * 60)
    
    total_train = sum(c['train'] for c in class_counts.values())
    total_val = sum(c['val'] for c in class_counts.values())
    total_test = sum(c['test'] for c in class_counts.values())
    total_images = total_train + total_val + total_test
    
    print(f"\nTotal Images: {total_images}")
    print(f"  Training:   {total_train} ({total_train/total_images*100:.1f}%)")
    print(f"  Validation: {total_val} ({total_val/total_images*100:.1f}%)")
    print(f"  Test:       {total_test} ({total_test/total_images*100:.1f}%)")
    print(f"\nTotal Classes: {len(class_counts)}")
    
    print("\nPer-Class Distribution:")
    print("-" * 60)
    print(f"{'Class':<20} {'Train':<10} {'Val':<10} {'Test':<10} {'Total':<10}")
    print("-" * 60)
    
    for class_name in sorted(class_counts.keys()):
        counts = class_counts[class_name]
        total = counts['train'] + counts['val'] + counts['test']
        print(f"{class_name:<20} {counts['train']:<10} {counts['val']:<10} "
              f"{counts['test']:<10} {total:<10}")
    
    print("=" * 60)

def main():
    """Main execution function"""
    print("=" * 60)
    print("SmartCulinary AI - Dataset Explorer")
    print("=" * 60)
    
    # Load configuration
    config = load_config()
    
    # Get dataset directory
    project_root = Path(__file__).parent.parent
    dataset_dir = project_root / config['paths']['dataset_dir']
    
    if not dataset_dir.exists():
        print(f"\n✗ Dataset not found at: {dataset_dir}")
        print("Please run: python data/download_dataset.py")
        return
    
    # Count images
    print("\nAnalyzing dataset...")
    class_counts = count_images_per_class(dataset_dir)
    
    # Print summary
    print_dataset_summary(class_counts)
    
    # Create visualizations directory
    viz_dir = project_root / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    print("1. Class distribution plot...")
    plot_class_distribution(
        class_counts, 
        save_path=viz_dir / "class_distribution.png"
    )
    
    print("2. Sample images...")
    visualize_sample_images(
        dataset_dir, 
        num_samples=5,
        save_path=viz_dir / "sample_images.png"
    )
    
    print("3. Augmentation examples...")
    visualize_augmentation(
        dataset_dir,
        save_path=viz_dir / "augmentation_examples.png"
    )
    
    print("\n" + "=" * 60)
    print("Dataset exploration complete!")
    print("=" * 60)
    print(f"\nVisualizations saved to: {viz_dir}")

if __name__ == "__main__":
    main()
