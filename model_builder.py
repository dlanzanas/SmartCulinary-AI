"""
SmartCulinary AI - Model Builder
Implements transfer learning architecture using MobileNetV2
"""

import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import yaml

def load_config():
    """Load configuration from YAML file"""
    config_path = Path(__file__).parent.parent / "config" / "training_config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def build_model(num_classes=25, config=None):
    """
    Build transfer learning model using MobileNetV2
    
    Args:
        num_classes: Number of output classes
        config: Configuration dictionary (optional)
    
    Returns:
        Compiled Keras model
    """
    if config is None:
        config = load_config()
    
    model_config = config['model']
    input_shape = tuple(model_config['input_shape'])
    
    print("Building model architecture...")
    print(f"  Base model: {model_config['base_model']}")
    print(f"  Input shape: {input_shape}")
    print(f"  Number of classes: {num_classes}")
    
    # Load pre-trained MobileNetV2
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=model_config['weights'],
        pooling=model_config['pooling']
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    print(f"  Base model layers: {len(base_model.layers)}")
    
    # Build custom classification head
    inputs = keras.Input(shape=input_shape)
    
    # Data augmentation layer (will be applied during training)
    x = inputs
    
    # Base model
    x = base_model(x, training=False)
    
    # Custom classification head
    head_config = model_config['head']
    
    # Dense layers
    for units in head_config['dense_units']:
        x = keras.layers.Dense(
            units, 
            activation=head_config['activation'],
            kernel_regularizer=keras.regularizers.l2(0.01)
        )(x)
        x = keras.layers.Dropout(head_config['dropout_rate'])(x)
    
    # Output layer
    outputs = keras.layers.Dense(
        num_classes, 
        activation=head_config['final_activation'],
        name='predictions'
    )(x)
    
    # Create model
    model = keras.Model(inputs, outputs, name='SmartCulinary_MobileNetV2')
    
    print(f"✓ Model architecture built successfully")
    
    return model, base_model

def compile_model(model, config=None, phase='phase1'):
    """
    Compile model with optimizer and loss
    
    Args:
        model: Keras model
        config: Configuration dictionary
        phase: Training phase ('phase1' or 'phase2')
    
    Returns:
        Compiled model
    """
    if config is None:
        config = load_config()
    
    training_config = config['training'][phase]
    
    # Optimizer
    if training_config['optimizer'] == 'adam':
        optimizer = keras.optimizers.Adam(
            learning_rate=training_config['learning_rate']
        )
    elif training_config['optimizer'] == 'sgd':
        optimizer = keras.optimizers.SGD(
            learning_rate=training_config['learning_rate'],
            momentum=0.9
        )
    else:
        raise ValueError(f"Unknown optimizer: {training_config['optimizer']}")
    
    # Loss with optional label smoothing
    label_smoothing = config['training'].get('label_smoothing', 0.0)
    if label_smoothing > 0:
        loss = keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)
        print(f"  Using label smoothing: {label_smoothing}")
    else:
        loss = config['training']['loss']
    
    # Metrics
    metrics = [
        'accuracy',
        keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')
    ]
    
    # Compile
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    print(f"✓ Model compiled for {phase}")
    print(f"  Optimizer: {training_config['optimizer']}")
    print(f"  Learning rate: {training_config['learning_rate']}")
    
    return model

def unfreeze_base_model(base_model, unfreeze_from_layer=100):
    """
    Unfreeze base model layers for fine-tuning
    
    Args:
        base_model: Base model to unfreeze
        unfreeze_from_layer: Layer index to start unfreezing from
    """
    base_model.trainable = True
    
    # Freeze early layers, unfreeze later layers
    for layer in base_model.layers[:unfreeze_from_layer]:
        layer.trainable = False
    
    trainable_layers = sum([1 for layer in base_model.layers if layer.trainable])
    
    print(f"✓ Base model unfrozen")
    print(f"  Total layers: {len(base_model.layers)}")
    print(f"  Trainable layers: {trainable_layers}")
    print(f"  Frozen layers: {len(base_model.layers) - trainable_layers}")
    
    return base_model

def get_model_summary(model, save_path=None):
    """
    Get and optionally save model summary
    
    Args:
        model: Keras model
        save_path: Path to save summary (optional)
    
    Returns:
        Model summary string
    """
    # Print summary
    model.summary()
    
    # Save to file if requested
    if save_path:
        with open(save_path, 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        print(f"✓ Model summary saved to: {save_path}")
    
    # Calculate model size
    total_params = model.count_params()
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Non-trainable parameters: {total_params - trainable_params:,}")
    
    return model

if __name__ == "__main__":
    # Test model building
    print("=" * 60)
    print("SmartCulinary AI - Model Builder Test")
    print("=" * 60)
    print()
    
    try:
        # Load config
        config = load_config()
        
        # Build model
        model, base_model = build_model(
            num_classes=config['dataset']['num_classes'],
            config=config
        )
        
        # Compile for phase 1
        model = compile_model(model, config, phase='phase1')
        
        # Get summary
        project_root = Path(__file__).parent.parent
        summary_path = project_root / "model_summary.txt"
        get_model_summary(model, save_path=summary_path)
        
        print("\n✓ Model building test successful!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
