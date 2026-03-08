"""
SmartCulinary AI - Grocery Store Dataset Flattener
Flattens the hierarchical structure (Category/Product/Variety) to flat product-level classes
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict
import yaml

def load_config():
    """Load configuration from YAML file"""
    config_path = Path(__file__).parent.parent / "config" / "training_config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Classes to exclude from dataset (poor performance or confusion)
EXCLUDED_CLASSES = {
    'red_grapefruit',  # 47% accuracy - too similar to orange
    'satsumas',        # 41% accuracy - confuses lemon/lime
}

def flatten_grocery_store_dataset(source_dir, output_dir):
    """
    Flatten the hierarchical Grocery Store dataset structure
    From: Category/Product/Variety -> Product
    
    Args:
        source_dir: Path to source dataset (hierarchical structure)
        output_dir: Path to output flattened dataset
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    print("\n" + "=" * 60)
    print("Flattening Grocery Store Dataset")
    print("=" * 60)
    print(f"\nSource: {source_path}")
    print(f"Output: {output_path}")
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    stats = defaultdict(lambda: {'train': 0, 'val': 0, 'test': 0})
    
    # Process each split (train, val, test)
    for split in ['train', 'val', 'test']:
        split_source = source_path / split
        split_output = output_path / split
        
        if not split_source.exists():
            print(f"  ⚠ {split} split not found, skipping")
            continue
        
        split_output.mkdir(parents=True, exist_ok=True)
        
        print(f"\nProcessing {split} split...")
        
        # Walk through category/product/variety structure
        for category_dir in split_source.iterdir():
            if not category_dir.is_dir():
                continue
            
            category_name = category_dir.name
            print(f"  Category: {category_name}")
            
            # Iterate through products (e.g., Apple, Banana, Asparagus)
            for product_dir in category_dir.iterdir():
                if not product_dir.is_dir():
                    continue
                
                # Normalize product name (lowercase, replace hyphens with underscores)
                product_name = product_dir.name.lower().replace('-', '_')
                
                # Fix mushroom naming consistency
                # Normalize all mushroom varieties to 'brown_cap_mushroom'
                if 'mushroom' in product_name:
                    product_name = 'brown_cap_mushroom'
                
                # Skip excluded classes
                if product_name in EXCLUDED_CLASSES:
                    print(f"  Skipping excluded class: {product_name}")
                    continue
                
                # Create product directory in output
                product_output_dir = split_output / product_name
                product_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Check if product has variety subdirectories or direct images
                subdirs = [item for item in product_dir.iterdir() if item.is_dir()]
                direct_images = list(product_dir.glob('*.jpg')) + \
                               list(product_dir.glob('*.png')) + \
                               list(product_dir.glob('*.jpeg'))
                
                if subdirs and not direct_images:
                    # Case 1: Product has variety subdirectories (e.g., Apple → Golden-Delicious)
                    for variety_dir in subdirs:
                        variety_name = variety_dir.name
                        
                        # Copy all images from this variety to the product directory
                        image_files = list(variety_dir.glob('*.jpg')) + \
                                     list(variety_dir.glob('*.png')) + \
                                     list(variety_dir.glob('*.jpeg'))
                        
                        for img_file in image_files:
                            # Keep original filename (already includes variety name)
                            dest_file = product_output_dir / img_file.name
                            shutil.copy2(img_file, dest_file)
                            stats[product_name][split] += 1
                
                elif direct_images:
                    # Case 2: Product has images directly (e.g., Asparagus → images)
                    for img_file in direct_images:
                        dest_file = product_output_dir / img_file.name
                        shutil.copy2(img_file, dest_file)
                        stats[product_name][split] += 1
    
    # Print summary
    print("\n" + "=" * 60)
    print("Flattening Summary")
    print("=" * 60)
    
    total_images = 0
    total_classes = len(stats)
    
    print(f"\nTotal classes: {total_classes}")
    print(f"\nPer-class breakdown:")
    
    for product_name in sorted(stats.keys()):
        train_count = stats[product_name]['train']
        val_count = stats[product_name]['val']
        test_count = stats[product_name]['test']
        total = train_count + val_count + test_count
        total_images += total
        
        print(f"  {product_name}: {train_count} train, {val_count} val, {test_count} test (total: {total})")
    
    print(f"\nTotal images: {total_images}")
    print("Success: Flattening complete!")
    
    return True

def main():
    """Main execution function"""
    print("=" * 60)
    print("SmartCulinary AI - Grocery Store Dataset Flattener")
    print("=" * 60)
    
    # Load configuration
    config = load_config()
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    datasets_dir = project_root / "datasets"
    
    source_dir = datasets_dir / "grocery_store_raw"
    output_dir = datasets_dir / "grocery_store_flat"
    
    # Check if source exists
    if not source_dir.exists():
        print(f"\n✗ Source directory not found: {source_dir}")
        print("Please run download_dataset.py first")
        return
    
    # Check if output already exists
    if output_dir.exists() and any(output_dir.iterdir()):
        response = input(f"\nFlattened dataset already exists. Re-flatten? (y/n): ")
        if response.lower() != 'y':
            print("Using existing flattened dataset.")
            return
        else:
            shutil.rmtree(output_dir)
    
    # Flatten the dataset
    success = flatten_grocery_store_dataset(source_dir, output_dir)
    
    if success:
        print("\n" + "=" * 60)
        print("Next Steps")
        print("=" * 60)
        print("\n1. Merge with Freiburg dataset:")
        print("   python data/merge_datasets.py")
        print("\n2. Explore merged dataset:")
        print("   python data/data_explorer.py")
        print("\n3. Start training:")
        print("   python train.py")
    else:
        print("\n✗ Flattening failed")

if __name__ == "__main__":
    main()
