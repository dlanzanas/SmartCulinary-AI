# SmartCulinary AI - ML Training Program

Complete machine learning training program for the SmartCulinary AI MVP, featuring multi-dataset support, transfer learning with MobileNetV2, and automated TFLite conversion.

## 🎯 Project Overview

**SmartCulinary AI** is a mobile application that uses computer vision to identify grocery ingredients and automatically generates recipe recommendations. This repository contains the complete ML training pipeline for the ingredient recognition model.

### Success Metrics (MVP Targets)
- ✅ **Top-1 Accuracy**: > 85%
- ✅ **Top-5 Accuracy**: > 95%
- ✅ **Inference Latency**: < 200ms (mid-range smartphone)
- ✅ **Model Size**: < 15 MB

### Dataset Coverage
- **Freiburg Groceries Dataset**: 25 classes (packaged goods: beans, pasta, coffee, etc.)
- **Grocery Store Dataset**: 81 classes (fruits, vegetables, carton items)
- **Merged Dataset**: ~90+ unique grocery categories

## 🚀 Quick Start

### 1. Environment Setup

```powershell
# Run automated setup (installs Python 3.9.13 + all dependencies)
.\setup_environment.ps1

# Activate virtual environment
.\smartculinary_env\Scripts\Activate.ps1
```

### 2. Download Datasets

```bash
# Download both datasets (requires Kaggle API for Grocery Store dataset)
python data/download_dataset.py --dataset all

# Or download individually
python data/download_dataset.py --dataset freiburg
python data/download_dataset.py --dataset grocery_store
```

**Note**: Grocery Store dataset requires Kaggle API credentials. See [Kaggle API Setup](#kaggle-api-setup) below.

### 3. Merge Datasets

```bash
# Merge datasets with intelligent class mapping
python data/merge_datasets.py
```

### 4. Explore Dataset

```bash
# Generate visualizations and statistics
python data/data_explorer.py
```

### 5. Train Model

```bash
# Two-phase training (head only → fine-tune entire model)
python train.py

# Monitor training with TensorBoard
tensorboard --logdir logs
```

### 6. Evaluate Model

```bash
# Comprehensive evaluation with visualizations
python evaluate.py --model checkpoints/best_model.h5 --visualize
```

### 7. Convert to TFLite

```bash
# Convert for mobile deployment
python convert_to_tflite.py --model checkpoints/best_model.h5
```

## 📁 Project Structure

```
Project/
├── setup_environment.ps1          # Automated environment setup
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── train.py                       # Main training script
├── evaluate.py                    # Model evaluation
├── convert_to_tflite.py          # TFLite conversion
│
├── config/
│   └── training_config.yaml      # All configuration parameters
│
├── data/
│   ├── download_dataset.py       # Multi-dataset downloader
│   ├── merge_datasets.py         # Dataset merger with class mapping
│   ├── data_preprocessing.py     # Preprocessing pipeline
│   └── data_explorer.py          # Dataset visualization
│
├── models/
│   ├── model_builder.py          # MobileNetV2 architecture
│   └── inference_test.py         # Inference testing
│
├── utils/
│   ├── metrics.py                # Custom metrics (Top-K accuracy, latency)
│   └── visualization.py          # Training plots and visualizations
│
├── datasets/                      # Downloaded datasets (auto-created)
│   ├── freiburg_groceries_raw/   # Raw Freiburg dataset
│   ├── grocery_store_raw/        # Raw Grocery Store dataset
│   └── merged_grocery_dataset/   # Merged and split dataset
│
├── checkpoints/                   # Model checkpoints (auto-created)
├── logs/                          # TensorBoard logs (auto-created)
├── exports/                       # TFLite models (auto-created)
├── reports/                       # Evaluation reports (auto-created)
└── visualizations/                # Generated plots (auto-created)
```

## 🔧 Configuration

All training parameters are centralized in `config/training_config.yaml`:

- **Dataset settings**: Image size, splits, class mapping
- **Data augmentation**: Rotation, flip, brightness, contrast
- **Model architecture**: MobileNetV2 base, custom head
- **Training**: Learning rates, batch sizes, epochs (2 phases)
- **Callbacks**: Early stopping, LR reduction, checkpointing
- **Success targets**: Accuracy, latency, model size

## 📊 Training Workflow

### Phase 1: Train Classification Head (20 epochs)
- Base MobileNetV2 frozen
- Learning rate: 0.001
- Batch size: 32

### Phase 2: Fine-Tune Entire Model (30 epochs)
- Unfreeze last 100 layers of base model
- Learning rate: 0.0001 (lower for stability)
- Batch size: 16

### Callbacks
- **ModelCheckpoint**: Save best model based on validation accuracy
- **EarlyStopping**: Stop if no improvement for 10 epochs
- **ReduceLROnPlateau**: Reduce LR by 0.5x if no improvement for 5 epochs
- **TensorBoard**: Log metrics and histograms

## 🔍 Evaluation Metrics

The evaluation script (`evaluate.py`) provides:

1. **Accuracy Metrics**
   - Top-1 accuracy (primary metric)
   - Top-5 accuracy (secondary metric)
   - Per-class accuracy breakdown

2. **Performance Metrics**
   - Inference latency (mean, median, P95, P99)
   - Model size (MB)

3. **Visualizations**
   - Confusion matrix
   - Per-class accuracy bar chart
   - Sample predictions with confidence scores

4. **Reports**
   - Classification report (precision, recall, F1-score)
   - JSON results for programmatic access

## 📱 Mobile Deployment

The TFLite conversion script optimizes the model for mobile deployment:

### Optimization Options
- **Default**: Dynamic range quantization (recommended)
- **Float16**: 16-bit floating point quantization
- **Int8**: 8-bit integer quantization (requires representative dataset)

### Deployment Checklist
- ✅ Model size < 15 MB
- ✅ Inference latency < 200ms
- ✅ Accuracy retention > 95% after conversion
- ✅ Compatible with TensorFlow Lite 2.10+

## 🔑 Kaggle API Setup

To download the Grocery Store Dataset, you need Kaggle API credentials:

1. Go to [https://www.kaggle.com/account](https://www.kaggle.com/account)
2. Scroll to "API" section
3. Click "Create New API Token"
4. This downloads `kaggle.json`
5. Place `kaggle.json` in:
   - Windows: `C:\Users\<YourUsername>\.kaggle\kaggle.json`
   - Linux/Mac: `~/.kaggle/kaggle.json`

Alternatively, you can train with only the Freiburg dataset by setting `grocery_store.enabled: false` in `config/training_config.yaml`.

## 🐛 Troubleshooting

### GPU Not Detected
- Ensure NVIDIA GPU drivers are installed
- Install CUDA Toolkit 11.2 and cuDNN 8.1
- Verify with: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`

### Out of Memory During Training
- Reduce batch size in `config/training_config.yaml`
- Enable mixed precision: `hardware.mixed_precision: true`
- Close other GPU-intensive applications

### Kaggle Dataset Download Fails
- Verify Kaggle API credentials are correctly placed
- Check internet connection
- Try downloading manually from [Kaggle](https://www.kaggle.com/datasets/marcusklasson/grocery-store-dataset)

### Model Accuracy Below Target
- Increase training epochs
- Adjust data augmentation parameters
- Try different learning rates
- Ensure dataset quality (check for corrupted images)

## 📈 Expected Results

Based on the MVP requirements and dataset characteristics:

- **Training Time**: ~2-4 hours (GPU) / ~12-20 hours (CPU)
- **Final Model Size**: ~10-12 MB (TFLite)
- **Expected Top-1 Accuracy**: 87-92%
- **Expected Top-5 Accuracy**: 96-98%
- **Inference Latency**: 80-150ms (mid-range smartphone)

## 🤝 Contributing

This is an MVP training program. Potential improvements:

- [ ] Add more datasets (e.g., Open Images, ImageNet subsets)
- [ ] Implement advanced augmentation (CutMix, MixUp)
- [ ] Try other architectures (EfficientNet, Vision Transformer)
- [ ] Add model pruning and quantization-aware training
- [ ] Implement active learning for continuous improvement

## 📄 License

This project is part of the SmartCulinary AI MVP. All rights reserved.

## 🙏 Acknowledgments

- **Freiburg Groceries Dataset**: [PhilJd/freiburg_groceries_dataset](https://github.com/PhilJd/freiburg_groceries_dataset)
- **Grocery Store Dataset**: [Marcus Klasson (Kaggle)](https://www.kaggle.com/datasets/marcusklasson/grocery-store-dataset)
- **MobileNetV2**: [Sandler et al., 2018](https://arxiv.org/abs/1801.04381)

---

**For questions or issues, please refer to the troubleshooting section or check the generated reports in the `reports/` directory.**
