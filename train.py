"""
SmartCulinary AI - Main Training Script
Two-phase training: (1) Train head only, (2) Fine-tune entire model
"""

import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import yaml
import argparse
from datetime import datetime
import json

# Import custom modules
import sys
sys.path.append(str(Path(__file__).parent))

from data.data_preprocessing import create_datasets, calculate_class_weights
from models.model_builder import build_model, compile_model, unfreeze_base_model, get_model_summary

def load_config():
    """Load configuration from YAML file"""
    config_path = Path(__file__).parent / "config" / "training_config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_callbacks(config, phase='phase1'):
    """Create training callbacks"""
    project_root = Path(__file__).parent
    
    # Directories
    checkpoint_dir = project_root / config['paths']['checkpoints_dir']
    logs_dir = project_root / config['paths']['logs_dir']
    checkpoint_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    
    # Timestamp for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    callbacks = []
    
    # ModelCheckpoint
    checkpoint_path = checkpoint_dir / f"model_{phase}_{timestamp}.h5"
    callbacks.append(
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor=config['training']['callbacks']['model_checkpoint']['monitor'],
            save_best_only=config['training']['callbacks']['model_checkpoint']['save_best_only'],
            save_weights_only=config['training']['callbacks']['model_checkpoint']['save_weights_only'],
            verbose=1
        )
    )
    
    # EarlyStopping
    callbacks.append(
        keras.callbacks.EarlyStopping(
            monitor=config['training']['callbacks']['early_stopping']['monitor'],
            patience=config['training']['callbacks']['early_stopping']['patience'],
            restore_best_weights=config['training']['callbacks']['early_stopping']['restore_best_weights'],
            verbose=1
        )
    )
    
    # ReduceLROnPlateau
    callbacks.append(
        keras.callbacks.ReduceLROnPlateau(
            monitor=config['training']['callbacks']['reduce_lr']['monitor'],
            factor=config['training']['callbacks']['reduce_lr']['factor'],
            patience=config['training']['callbacks']['reduce_lr']['patience'],
            min_lr=config['training']['callbacks']['reduce_lr']['min_lr'],
            verbose=1
        )
    )
    
    # TensorBoard
    tensorboard_dir = logs_dir / f"{phase}_{timestamp}"
    callbacks.append(
        keras.callbacks.TensorBoard(
            log_dir=str(tensorboard_dir),
            histogram_freq=config['training']['callbacks']['tensorboard']['histogram_freq'],
            write_graph=config['training']['callbacks']['tensorboard']['write_graph']
        )
    )
    
    print(f"✓ Callbacks created for {phase}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  TensorBoard logs: {tensorboard_dir}")
    
    return callbacks, checkpoint_path, tensorboard_dir

def train_phase1(model, base_model, train_ds, val_ds, config, class_weights=None):
    """
    Phase 1: Train classification head only (base model frozen)
    """
    print("\n" + "=" * 60)
    print("PHASE 1: Training Classification Head")
    print("=" * 60)
    
    # Ensure base model is frozen
    base_model.trainable = False
    
    # Compile model for phase 1
    model = compile_model(model, config, phase='phase1')
    
    # Create callbacks
    callbacks, checkpoint_path, tensorboard_dir = create_callbacks(config, phase='phase1')
    
    # Training configuration
    phase1_config = config['training']['phase1']
    epochs = phase1_config['epochs']
    
    print(f"\nTraining configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {phase1_config['batch_size']}")
    print(f"  Learning rate: {phase1_config['learning_rate']}")
    print(f"  Base model frozen: {not base_model.trainable}")
    if class_weights:
        print(f"  Using class weights: Yes")
    
    # Train
    print("\nStarting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    print(f"\nSuccess: Phase 1 training complete!")
    print(f"  Best model saved to: {checkpoint_path}")
    print(f"  TensorBoard logs: {tensorboard_dir}")
    
    return model, history, checkpoint_path

def train_phase2(model, base_model, train_ds, val_ds, config, class_weights=None):
    """
    Phase 2: Fine-tune entire model (unfreeze base model)
    """
    print("\n" + "=" * 60)
    print("PHASE 2: Fine-Tuning Entire Model")
    print("=" * 60)
    
    # Unfreeze base model
    phase2_config = config['training']['phase2']
    base_model = unfreeze_base_model(
        base_model, 
        unfreeze_from_layer=phase2_config['unfreeze_from_layer']
    )
    
    # Recompile model for phase 2 with lower learning rate
    model = compile_model(model, config, phase='phase2')
    
    # Create callbacks
    callbacks, checkpoint_path, tensorboard_dir = create_callbacks(config, phase='phase2')
    
    # Training configuration
    epochs = phase2_config['epochs']
    
    print(f"\nTraining configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {phase2_config['batch_size']}")
    print(f"  Learning rate: {phase2_config['learning_rate']}")
    print(f"  Base model trainable: {base_model.trainable}")
    if class_weights:
        print(f"  Using class weights: Yes")
    
    # Train
    print("\nStarting fine-tuning...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    print(f"\nSuccess: Phase 2 training complete!")
    print(f"  Best model saved to: {checkpoint_path}")
    print(f"  TensorBoard logs: {tensorboard_dir}")
    
    return model, history, checkpoint_path

def save_training_history(history, phase, save_dir):
    """Save training history to JSON"""
    history_dict = history.history
    
    # Convert numpy arrays to lists for JSON serialization
    for key in history_dict:
        history_dict[key] = [float(x) for x in history_dict[key]]
    
    save_path = save_dir / f"training_history_{phase}.json"
    with open(save_path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    print(f"  Training history saved to: {save_path}")

def main(args):
    """Main training function"""
    print("=" * 60)
    print("SmartCulinary AI - Model Training")
    print("=" * 60)
    print()
    
    # Load configuration
    config = load_config()
    
    # Setup GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✓ GPU detected: {len(gpus)} device(s)")
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"  Warning: {e}")
    else:
        print("⚠ No GPU detected - training will use CPU")
    
    print()
    
    # Load datasets
    print("Loading datasets...")
    train_ds, val_ds, test_ds, class_names = create_datasets()
    
    print(f"\n✓ Datasets loaded")
    print(f"  Number of classes: {len(class_names)}")
    print()
    
    # Build model
    print("Building model...")
    model, base_model = build_model(
        num_classes=len(class_names),
        config=config
    )
    
    # Save model architecture
    project_root = Path(__file__).parent
    get_model_summary(model, save_path=project_root / "model_architecture.txt")
    print()
    
    # Calculate class weights if enabled
    class_weights = None
    if config['training'].get('use_class_weights', False):
        print("Calculating class weights for imbalanced dataset...")
        dataset_dir = project_root / config['paths']['dataset_dir']
        class_weights = calculate_class_weights(dataset_dir, class_names)
        print()
    
    # Phase 1: Train head only
    if not args.skip_phase1:
        model, history1, checkpoint1 = train_phase1(
            model, base_model, train_ds, val_ds, config, class_weights
        )
        
        # Save history
        save_training_history(history1, 'phase1', project_root / config['paths']['checkpoints_dir'])
    else:
        print("\n⚠ Skipping Phase 1 (--skip-phase1 flag set)")
        checkpoint1 = None
    
    # Phase 2: Fine-tune entire model
    if not args.skip_phase2:
        model, history2, checkpoint2 = train_phase2(
            model, base_model, train_ds, val_ds, config, class_weights
        )
        
        # Save history
        save_training_history(history2, 'phase2', project_root / config['paths']['checkpoints_dir'])
    else:
        print("\n⚠ Skipping Phase 2 (--skip-phase2 flag set)")
        checkpoint2 = None
    
    # Save final model
    final_model_path = project_root / config['paths']['checkpoints_dir'] / "best_model.h5"
    
    if checkpoint2:
        # Use phase 2 checkpoint as best model
        import shutil
        shutil.copy2(checkpoint2, final_model_path)
        print(f"\n✓ Best model (Phase 2) copied to: {final_model_path}")
    elif checkpoint1:
        # Use phase 1 checkpoint as best model
        import shutil
        shutil.copy2(checkpoint1, final_model_path)
        print(f"\n✓ Best model (Phase 1) copied to: {final_model_path}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nBest model saved to: {final_model_path}")
    print("\nNext steps:")
    print("  1. Evaluate model: python evaluate.py")
    print("  2. Convert to TFLite: python convert_to_tflite.py")
    print(f"  3. View training logs: tensorboard --logdir {project_root / config['paths']['logs_dir']}")
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train SmartCulinary AI model')
    parser.add_argument('--skip-phase1', action='store_true', 
                       help='Skip phase 1 training (train head only)')
    parser.add_argument('--skip-phase2', action='store_true',
                       help='Skip phase 2 training (fine-tuning)')
    
    args = parser.parse_args()
    
    main(args)
