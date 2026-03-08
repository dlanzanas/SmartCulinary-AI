"""
SmartCulinary AI - Data Preprocessing Pipeline
Handles image preprocessing and augmentation for training
"""

import tensorflow as tf
from pathlib import Path
import yaml

def load_config():
    """Load configuration from YAML file"""
    config_path = Path(__file__).parent.parent / "config" / "training_config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class DataPreprocessor:
    """Handles all data preprocessing and augmentation"""
    
    def __init__(self, config):
        self.config = config
        self.image_size = tuple(config['dataset']['image_size'])
        self.num_classes = config['dataset']['num_classes']
        self.batch_size = config['training']['phase1']['batch_size']
        
    def create_augmentation_layer(self):
        """Create data augmentation layer"""
        aug_config = self.config['augmentation']
        
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(aug_config['rotation_range'] / 360.0),
            tf.keras.layers.RandomZoom(aug_config['zoom_range']),
            tf.keras.layers.RandomContrast(
                (aug_config['contrast_range'][1] - aug_config['contrast_range'][0]) / 2
            ),
        ], name="data_augmentation")
        
        return data_augmentation
    
    def preprocess_image(self, image, label):
        """Preprocess a single image"""
        # Resize to target size
        image = tf.image.resize(image, self.image_size)
        
        # Normalize to [0, 1]
        image = tf.cast(image, tf.float32) / 255.0
        
        return image, label
    
    def load_dataset(self, data_dir, split='train', shuffle=True, augment=True):
        """
        Load dataset from directory
        
        Args:
            data_dir: Path to dataset directory
            split: 'train', 'val', or 'test'
            shuffle: Whether to shuffle the dataset
            augment: Whether to apply data augmentation (only for training)
        
        Returns:
            tf.data.Dataset
        """
        split_dir = Path(data_dir) / split
        
        if not split_dir.exists():
            raise ValueError(f"Split directory not found: {split_dir}")
        
        # Load dataset using image_dataset_from_directory
        dataset = tf.keras.utils.image_dataset_from_directory(
            split_dir,
            image_size=self.image_size,
            batch_size=self.batch_size,
            shuffle=shuffle,
            seed=42,
            label_mode='categorical'
        )
        
        # Get class names
        class_names = dataset.class_names
        
        # Normalize images
        normalization_layer = tf.keras.layers.Rescaling(1./255)
        dataset = dataset.map(
            lambda x, y: (normalization_layer(x), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Apply augmentation only for training
        if split == 'train' and augment:
            augmentation_layer = self.create_augmentation_layer()
            dataset = dataset.map(
                lambda x, y: (augmentation_layer(x, training=True), y),
                num_parallel_calls=tf.data.AUTOTUNE
            )
        
        # Performance optimization
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        
        return dataset, class_names
    
    def get_dataset_info(self, dataset):
        """Get information about the dataset"""
        total_batches = tf.data.experimental.cardinality(dataset).numpy()
        total_samples = total_batches * self.batch_size
        
        return {
            'total_batches': total_batches,
            'estimated_samples': total_samples,
            'batch_size': self.batch_size
        }

def create_datasets(config_path=None):
    """
    Convenience function to create train, validation, and test datasets
    
    Returns:
        tuple: (train_ds, val_ds, test_ds, class_names)
    """
    if config_path is None:
        config = load_config()
    else:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    
    preprocessor = DataPreprocessor(config)
    
    # Get dataset directory
    project_root = Path(__file__).parent.parent
    dataset_dir = project_root / config['paths']['dataset_dir']
    
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_dir}\n"
            "Please run: python data/download_dataset.py"
        )
    
    print("Loading datasets...")
    
    # Load train dataset with augmentation
    train_ds, class_names = preprocessor.load_dataset(
        dataset_dir, 
        split='train', 
        shuffle=True, 
        augment=True
    )
    print(f"✓ Training dataset loaded")
    
    # Load validation dataset without augmentation
    val_ds, _ = preprocessor.load_dataset(
        dataset_dir, 
        split='val', 
        shuffle=False, 
        augment=False
    )
    print(f"✓ Validation dataset loaded")
    
    # Load test dataset without augmentation
    test_ds, _ = preprocessor.load_dataset(
        dataset_dir, 
        split='test', 
        shuffle=False, 
        augment=False
    )
    print(f"✓ Test dataset loaded")
    
    # Print dataset info
    train_info = preprocessor.get_dataset_info(train_ds)
    val_info = preprocessor.get_dataset_info(val_ds)
    test_info = preprocessor.get_dataset_info(test_ds)
    
    print(f"\nDataset Statistics:")
    print(f"  Classes: {len(class_names)}")
    print(f"  Training samples: ~{train_info['estimated_samples']}")
    print(f"  Validation samples: ~{val_info['estimated_samples']}")
    print(f"  Test samples: ~{test_info['estimated_samples']}")
    print(f"  Batch size: {preprocessor.batch_size}")
    
    return train_ds, val_ds, test_ds, class_names

def calculate_class_weights(dataset_dir, class_names):
    """
    Calculate class weights for handling imbalanced datasets
    
    Args:
        dataset_dir: Path to dataset directory
        class_names: List of class names
    
    Returns:
        dict: Class weights dictionary {class_index: weight}
    """
    import numpy as np
    from sklearn.utils.class_weight import compute_class_weight
    
    # Count samples per class in training set
    train_dir = Path(dataset_dir) / 'train'
    class_counts = []
    
    for class_name in class_names:
        class_dir = train_dir / class_name
        if class_dir.exists():
            num_samples = len(list(class_dir.glob('*.jpg')) + 
                            list(class_dir.glob('*.png')) + 
                            list(class_dir.glob('*.jpeg')))
            class_counts.append(num_samples)
        else:
            class_counts.append(0)
    
    # Compute balanced class weights
    class_indices = np.arange(len(class_names))
    
    # Create sample array for sklearn
    y_train = []
    for idx, count in enumerate(class_counts):
        y_train.extend([idx] * count)
    y_train = np.array(y_train)
    
    # Compute weights
    weights = compute_class_weight(
        class_weight='balanced',
        classes=class_indices,
        y=y_train
    )
    
    # Create weight dictionary
    class_weight_dict = {i: weights[i] for i in range(len(class_names))}
    
    # Print statistics
    print("\nClass Weight Statistics:")
    print(f"  Min weight: {weights.min():.3f}")
    print(f"  Max weight: {weights.max():.3f}")
    print(f"  Weight ratio: {weights.max() / weights.min():.2f}:1")
    
    # Show top 5 most weighted classes (minority classes)
    top_weighted_indices = np.argsort(weights)[-5:][::-1]
    print("\n  Top 5 weighted classes (minority):")
    for idx in top_weighted_indices:
        print(f"    {class_names[idx]}: {weights[idx]:.3f} (count: {class_counts[idx]})")
    
    return class_weight_dict

if __name__ == "__main__":
    # Test the preprocessing pipeline
    print("Testing data preprocessing pipeline...")
    print("=" * 60)
    
    try:
        train_ds, val_ds, test_ds, class_names = create_datasets()
        
        print("\nClass names:")
        for i, name in enumerate(class_names):
            print(f"  {i}: {name}")
        
        print("\n✓ Data preprocessing pipeline test successful!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
