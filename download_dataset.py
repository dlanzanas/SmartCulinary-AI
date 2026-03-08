"""
SmartCulinary AI - Multi-Dataset Downloader
Downloads Freiburg Groceries and Grocery Store datasets
"""

import os
import sys
import zipfile
import requests
import shutil
from pathlib import Path
from tqdm import tqdm
import yaml
import argparse
import subprocess

def load_config():
    """Load configuration from YAML file"""
    config_path = Path(__file__).parent.parent / "config" / "training_config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def download_file(url, destination):
    """Download file with progress bar"""
    print(f"Downloading from: {url}")
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f, tqdm(
        desc=destination.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)
    
    print(f"✓ Download complete: {destination}")

def extract_zip(zip_path, extract_to):
    """Extract ZIP file"""
    print(f"Extracting {zip_path.name}...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    print(f"✓ Extraction complete")

def check_kaggle_credentials():
    """Check if Kaggle API credentials are configured"""
    kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'
    return kaggle_json.exists()

def download_kaggle_dataset(dataset_name, download_path):
    """Download dataset from Kaggle using kaggle API"""
    print(f"Downloading Kaggle dataset: {dataset_name}")
    
    # Check if kaggle is installed
    try:
        import kaggle
    except ImportError:
        print("✗ Kaggle package not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle", "--quiet"])
        import kaggle
    
    # Check credentials
    if not check_kaggle_credentials():
        print("\n" + "=" * 60)
        print("Kaggle API Credentials Required")
        print("=" * 60)
        print("\nTo download the Grocery Store Dataset, you need Kaggle API credentials.")
        print("\nSteps:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Scroll to 'API' section and click 'Create New API Token'")
        print("3. This will download kaggle.json")
        print("4. Place kaggle.json in: " + str(Path.home() / '.kaggle'))
        print("\nAlternatively, you can skip this dataset and use only Freiburg Groceries.")
        print("=" * 60)
        return False
    
    # Download dataset
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        
        api.dataset_download_files(
            dataset_name,
            path=download_path,
            unzip=True,
            quiet=False
        )
        
        print(f"✓ Kaggle dataset downloaded successfully")
        return True
        
    except Exception as e:
        print(f"✗ Failed to download Kaggle dataset: {e}")
        return False

def organize_dataset(source_dir, target_dir, train_split=0.70, val_split=0.15):
    """
    Organize dataset into train/val/test splits
    Expected structure: source_dir/class_name/*.jpg
    """
    import random
    from collections import defaultdict
    
    print("Organizing dataset into train/val/test splits...")
    
    # Create split directories
    splits = ['train', 'val', 'test']
    for split in splits:
        split_dir = target_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all class directories
    class_dirs = [d for d in source_dir.iterdir() if d.is_dir()]
    
    if not class_dirs:
        print("✗ No class directories found!")
        return False
    
    print(f"Found {len(class_dirs)} classes")
    
    stats = defaultdict(lambda: {'train': 0, 'val': 0, 'test': 0})
    total_images = 0
    
    # Process each class
    for class_dir in class_dirs:
        class_name = class_dir.name
        
        # Skip hidden directories and non-class folders
        if class_name.startswith('.') or class_name.startswith('_'):
            continue
        
        # Get all images in this class
        image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpeg'))
        
        if not image_files:
            print(f"  ⚠ No images found in {class_name}, skipping")
            continue
        
        # Shuffle images
        random.seed(42)  # For reproducibility
        random.shuffle(image_files)
        
        # Calculate split indices
        n_images = len(image_files)
        n_train = int(n_images * train_split)
        n_val = int(n_images * val_split)
        
        train_files = image_files[:n_train]
        val_files = image_files[n_train:n_train + n_val]
        test_files = image_files[n_train + n_val:]
        
        # Create class directories in each split
        for split in splits:
            class_split_dir = target_dir / split / class_name
            class_split_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy files to respective splits
        for img in train_files:
            shutil.copy2(img, target_dir / 'train' / class_name / img.name)
            stats[class_name]['train'] += 1
        
        for img in val_files:
            shutil.copy2(img, target_dir / 'val' / class_name / img.name)
            stats[class_name]['val'] += 1
        
        for img in test_files:
            shutil.copy2(img, target_dir / 'test' / class_name / img.name)
            stats[class_name]['test'] += 1
        
        total_images += n_images
        print(f"  {class_name}: {n_train} train, {n_val} val, {len(test_files)} test")
    
    print(f"\n✓ Dataset organized successfully!")
    print(f"  Total images: {total_images}")
    print(f"  Total classes: {len(stats)}")
    
    return True

def download_freiburg_dataset(config, datasets_dir):
    """Download and organize Freiburg Groceries Dataset"""
    print("\n" + "=" * 60)
    print("Downloading Freiburg Groceries Dataset")
    print("=" * 60)
    
    dataset_dir = datasets_dir / "freiburg_groceries_raw"
    
    # Check if already exists
    if dataset_dir.exists() and any(dataset_dir.iterdir()):
        response = input(f"Freiburg dataset already exists. Re-download? (y/n): ")
        if response.lower() != 'y':
            print("Using existing dataset.")
            return True
        else:
            shutil.rmtree(dataset_dir)
    
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if manually downloaded dataset exists
    manual_path = Path.home() / "Downloads" / "freiburg_groceries_dataset" / "images"
    if manual_path.exists():
        print(f"\n✓ Found manually downloaded dataset at: {manual_path}")
        response = input("Use this dataset instead of downloading? (y/n): ")
        if response.lower() == 'y':
            print("Using manually downloaded dataset...")
            success = organize_dataset(
                manual_path,
                dataset_dir,
                train_split=config['dataset']['train_split'],
                val_split=config['dataset']['val_split']
            )
            return success
    
    # Download
    dataset_url = config['paths']['freiburg_url']
    zip_path = datasets_dir / "freiburg_groceries.zip"
    
    try:
        download_file(dataset_url, zip_path)
    except Exception as e:
        print(f"✗ Download failed: {e}")
        return False
    
    # Extract
    temp_extract_dir = datasets_dir / "temp_freiburg"
    temp_extract_dir.mkdir(exist_ok=True)
    
    try:
        extract_zip(zip_path, temp_extract_dir)
    except Exception as e:
        print(f"✗ Extraction failed: {e}")
        return False
    
    # Find images directory - look for the 'images' folder with class directories
    source_dir = None
    
    # Search for images folder
    for root, dirs, files in os.walk(temp_extract_dir):
        root_path = Path(root)
        if root_path.name == "images":
            # Check if this images folder contains class directories
            subdirs = [d for d in root_path.iterdir() if d.is_dir()]
            if subdirs:
                source_dir = root_path
                print(f"Found images directory: {source_dir}")
                break
    
    if not source_dir:
        print("✗ Could not find images directory in extracted files")
        shutil.rmtree(temp_extract_dir)
        zip_path.unlink()
        return False
    
    # Organize into train/val/test
    success = organize_dataset(
        source_dir,
        dataset_dir,
        train_split=config['dataset']['train_split'],
        val_split=config['dataset']['val_split']
    )
    
    # Cleanup
    shutil.rmtree(temp_extract_dir)
    zip_path.unlink()
    
    return success

def download_grocery_store_dataset(config, datasets_dir):
    """Download and organize Grocery Store Dataset from Kaggle or manual download"""
    print("\n" + "=" * 60)
    print("Downloading Grocery Store Dataset")
    print("=" * 60)
    
    dataset_dir = datasets_dir / "grocery_store_raw"
    
    # Check if already exists
    if dataset_dir.exists() and any(dataset_dir.iterdir()):
        response = input(f"Grocery Store dataset already exists. Re-download? (y/n): ")
        if response.lower() != 'y':
            print("Using existing dataset.")
            return True
        else:
            shutil.rmtree(dataset_dir)
    
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for manually downloaded/cloned dataset
    manual_paths = [
        Path.home() / "Downloads" / "GroceryStoreDataset" / "dataset",
        Path.home() / "Downloads" / "GroceryStoreDataset-master" / "dataset",
        Path.home() / "Downloads" / "grocery-store-dataset" / "dataset"
    ]
    
    manual_source = None
    for path in manual_paths:
        if path.exists():
            manual_source = path
            break
    
    if manual_source:
        print(f"\n✓ Found manually downloaded dataset at: {manual_source}")
        response = input("Use this dataset instead of downloading from Kaggle? (y/n): ")
        if response.lower() == 'y':
            print("Using manually downloaded dataset...")
            # Copy the dataset structure
            if (manual_source / "train").exists():
                print("Dataset already split, copying structure...")
                shutil.copytree(manual_source / "train", dataset_dir / "train")
                if (manual_source / "val").exists():
                    shutil.copytree(manual_source / "val", dataset_dir / "val")
                else:
                    (dataset_dir / "val").mkdir(exist_ok=True)
                shutil.copytree(manual_source / "test", dataset_dir / "test")
                return True
            else:
                # Organize from raw images
                success = organize_dataset(
                    manual_source,
                    dataset_dir,
                    train_split=config['dataset']['train_split'],
                    val_split=config['dataset']['val_split']
                )
                return success
    
    # Download from GitHub
    dataset_url = config['paths']['grocery_store_url']
    zip_path = datasets_dir / "grocery_store.zip"
    
    try:
        download_file(dataset_url, zip_path)
    except Exception as e:
        print(f"✗ Download failed: {e}")
        print("\nAlternatively, you can:")
        print("1. Manually clone: git clone https://github.com/marcusklasson/GroceryStoreDataset.git")
        print("2. Place in Downloads folder and re-run this script")
        return False
    
    # Extract
    temp_extract_dir = datasets_dir / "temp_grocery_store"
    temp_extract_dir.mkdir(exist_ok=True)
    
    try:
        extract_zip(zip_path, temp_extract_dir)
    except Exception as e:
        print(f"✗ Extraction failed: {e}")
        return False
    
    # Find dataset directory - look for the 'dataset' folder
    source_dir = None
    
    # Search for dataset folder
    for root, dirs, files in os.walk(temp_extract_dir):
        root_path = Path(root)
        if root_path.name == "dataset":
            # Check if this dataset folder contains train/test directories
            if (root_path / "train").exists() or (root_path / "test").exists():
                source_dir = root_path
                print(f"Found dataset directory: {source_dir}")
                break
    
    if not source_dir:
        print("✗ Could not find dataset directory in extracted files")
        shutil.rmtree(temp_extract_dir)
        zip_path.unlink()
        return False
    
    # Check if already split
    if (source_dir / "train").exists() and (source_dir / "test").exists():
        print("Dataset already split, copying structure...")
        shutil.copytree(source_dir / "train", dataset_dir / "train")
        if (source_dir / "val").exists():
            shutil.copytree(source_dir / "val", dataset_dir / "val")
        shutil.copytree(source_dir / "test", dataset_dir / "test")
        
        # If no val split, create one from train
        if not (dataset_dir / "val").exists():
            print("Creating validation split from training data...")
            # This is a simplified approach - you may want to implement proper splitting
            (dataset_dir / "val").mkdir(exist_ok=True)
        
        success = True
    else:
        # Organize if not split
        success = organize_dataset(
            source_dir,
            dataset_dir,
            train_split=config['dataset']['train_split'],
            val_split=config['dataset']['val_split']
        )
    
    # Cleanup
    shutil.rmtree(temp_extract_dir)
    zip_path.unlink()
    
    return success

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Download grocery datasets')
    parser.add_argument(
        '--dataset',
        choices=['all', 'freiburg', 'grocery_store'],
        default='all',
        help='Which dataset(s) to download'
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("SmartCulinary AI - Multi-Dataset Downloader")
    print("=" * 60)
    print()
    
    # Load configuration
    config = load_config()
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    datasets_dir = project_root / "datasets"
    datasets_dir.mkdir(exist_ok=True)
    
    # Download datasets
    freiburg_success = False
    grocery_store_success = False
    
    if args.dataset in ['all', 'freiburg']:
        freiburg_success = download_freiburg_dataset(config, datasets_dir)
    
    if args.dataset in ['all', 'grocery_store']:
        grocery_store_success = download_grocery_store_dataset(config, datasets_dir)
    
    # Summary
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    
    if args.dataset in ['all', 'freiburg']:
        status = "✓ Success" if freiburg_success else "✗ Failed"
        print(f"Freiburg Groceries: {status}")
    
    if args.dataset in ['all', 'grocery_store']:
        status = "✓ Success" if grocery_store_success else "✗ Failed/Skipped"
        print(f"Grocery Store: {status}")
    
    print("\nNext steps:")
    if freiburg_success and grocery_store_success:
        print("  1. Merge datasets: python data/merge_datasets.py")
        print("  2. Explore merged dataset: python data/data_explorer.py")
        print("  3. Start training: python train.py")
    elif freiburg_success:
        print("  ⚠ Only Freiburg dataset available")
        print("  You can either:")
        print("    - Download Grocery Store dataset: python data/download_dataset.py --dataset grocery_store")
        print("    - Or train with Freiburg only (update config to disable grocery_store)")
    else:
        print("  ✗ No datasets downloaded successfully")
        print("  Please check errors above and try again")

if __name__ == "__main__":
    main()
