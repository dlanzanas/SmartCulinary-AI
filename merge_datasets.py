"""
SmartCulinary AI - Multi-Dataset Merger
Merges Freiburg Groceries and Grocery Store datasets with intelligent class mapping
"""

import os
import sys
import shutil
from pathlib import Path
from collections import defaultdict, Counter
import yaml
import json
from tqdm import tqdm

def load_config():
    """Load configuration from YAML file"""
    config_path = Path(__file__).parent.parent / "config" / "training_config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class DatasetMerger:
    """Handles merging of multiple grocery datasets"""
    
    def __init__(self, config):
        self.config = config
        self.class_mapping = {}
        self.merged_classes = []
        self.dataset_stats = defaultdict(lambda: defaultdict(int))
    
    def normalize_class_name(self, class_name):
        """Normalize class names for comparison"""
        # Convert to lowercase, remove special characters, strip whitespace
        normalized = class_name.lower().strip()
        normalized = normalized.replace('-', '_').replace(' ', '_')
        # Remove common prefixes/suffixes
        normalized = normalized.replace('_package', '').replace('_carton', '')
        return normalized
    
    def create_class_mapping(self, freiburg_classes, grocery_store_classes):
        """
        Create intelligent class mapping between datasets
        
        Returns:
            dict: Mapping from original class names to unified class names
        """
        print("\nCreating class mapping...")
        
        # Manual mappings from config
        manual_mapping = self.config['dataset'].get('class_mapping', {})
        
        # Initialize mapping
        mapping = {}
        unified_classes = set()
        
        # First, add all Freiburg classes (base dataset)
        for cls in freiburg_classes:
            normalized = self.normalize_class_name(cls)
            mapping[f"freiburg:{cls}"] = normalized
            unified_classes.add(normalized)
        
        # Then, map Grocery Store classes
        for cls in grocery_store_classes:
            normalized = self.normalize_class_name(cls)
            
            # Check manual mapping first
            if cls in manual_mapping:
                target = self.normalize_class_name(manual_mapping[cls])
                mapping[f"grocery_store:{cls}"] = target
                unified_classes.add(target)
                print(f"  Manual mapping: {cls} -> {target}")
            # Check if similar class exists in Freiburg
            elif normalized in unified_classes:
                mapping[f"grocery_store:{cls}"] = normalized
                print(f"  Auto-merged: {cls} -> {normalized}")
            else:
                # Add as new class
                mapping[f"grocery_store:{cls}"] = normalized
                unified_classes.add(normalized)
        
        self.class_mapping = mapping
        self.merged_classes = sorted(list(unified_classes))
        
        print("\nSuccess: Class mapping created")
        print(f"  Freiburg classes: {len(freiburg_classes)}")
        print(f"  Grocery Store classes: {len(grocery_store_classes)}")
        print(f"  Merged unique classes: {len(self.merged_classes)}")
        
        return mapping
    
    def get_dataset_classes(self, dataset_dir):
        """Get list of classes from a dataset directory"""
        classes = []
        if dataset_dir.exists():
            # Look for train directory
            train_dir = dataset_dir / 'train'
            if not train_dir.exists():
                # Try to find classes in root
                train_dir = dataset_dir
            
            classes = [d.name for d in train_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        
        return sorted(classes)
    
    def merge_datasets(self, freiburg_dir, grocery_store_dir, output_dir):
        """
        Merge two datasets into a unified structure
        
        Args:
            freiburg_dir: Path to Freiburg dataset
            grocery_store_dir: Path to Grocery Store dataset
            output_dir: Path to output merged dataset
        """
        print("\n" + "=" * 60)
        print("Merging Datasets")
        print("=" * 60)
        
        # Get classes from each dataset
        freiburg_classes = self.get_dataset_classes(freiburg_dir)
        grocery_store_classes = self.get_dataset_classes(grocery_store_dir)
        
        if not freiburg_classes and not grocery_store_classes:
            raise ValueError("No classes found in either dataset!")
        
        # Create class mapping
        self.create_class_mapping(freiburg_classes, grocery_store_classes)
        
        # Create output directory structure
        for split in ['train', 'val', 'test']:
            for cls in self.merged_classes:
                (output_dir / split / cls).mkdir(parents=True, exist_ok=True)
        
        # Merge Freiburg dataset
        if freiburg_dir.exists() and freiburg_classes:
            print("\nMerging Freiburg Groceries Dataset...")
            self._merge_single_dataset(
                freiburg_dir, 
                output_dir, 
                dataset_name='freiburg',
                classes=freiburg_classes
            )
        
        # Merge Grocery Store dataset
        if grocery_store_dir.exists() and grocery_store_classes:
            print("\nMerging Grocery Store Dataset...")
            self._merge_single_dataset(
                grocery_store_dir,
                output_dir,
                dataset_name='grocery_store',
                classes=grocery_store_classes
            )
        
        # Save merged dataset info
        self._save_merge_info(output_dir)
        
        print("\nSuccess: Dataset merging complete!")
        self._print_merge_statistics()
    
    def _merge_single_dataset(self, source_dir, target_dir, dataset_name, classes):
        """Merge a single dataset into the unified structure"""
        
        for split in ['train', 'val', 'test']:
            split_dir = source_dir / split
            
            if not split_dir.exists():
                # Dataset might not be split yet
                print(f"  ⚠ {split} split not found in {dataset_name}, skipping")
                continue
            
            for original_class in tqdm(classes, desc=f"  {split}"):
                source_class_dir = split_dir / original_class
                
                if not source_class_dir.exists():
                    continue
                
                # Get mapped class name
                mapped_class = self.class_mapping.get(f"{dataset_name}:{original_class}")
                
                if not mapped_class:
                    print(f"    ⚠ No mapping for {dataset_name}:{original_class}, skipping")
                    continue
                
                target_class_dir = target_dir / split / mapped_class
                
                # Copy images
                image_files = list(source_class_dir.glob('*.jpg')) + \
                             list(source_class_dir.glob('*.png')) + \
                             list(source_class_dir.glob('*.jpeg'))
                
                for img_file in image_files:
                    # Create unique filename to avoid conflicts
                    new_name = f"{dataset_name}_{img_file.name}"
                    target_path = target_class_dir / new_name
                    
                    shutil.copy2(img_file, target_path)
                    self.dataset_stats[split][mapped_class] += 1
    
    def _save_merge_info(self, output_dir):
        """Save information about the merged dataset"""
        info = {
            'num_classes': len(self.merged_classes),
            'classes': self.merged_classes,
            'class_mapping': self.class_mapping,
            'statistics': dict(self.dataset_stats)
        }
        
        info_path = output_dir / 'merge_info.json'
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"\n  Merge info saved to: {info_path}")
    
    def _print_merge_statistics(self):
        """Print statistics about the merged dataset"""
        print("\n" + "=" * 60)
        print("Merged Dataset Statistics")
        print("=" * 60)
        
        total_images = 0
        for split in ['train', 'val', 'test']:
            split_total = sum(self.dataset_stats[split].values())
            total_images += split_total
            print(f"\n{split.capitalize()} Split: {split_total} images")
        
        print(f"\nTotal Images: {total_images}")
        print(f"Total Classes: {len(self.merged_classes)}")
        
        # Print class distribution
        print("\nTop 10 Classes by Image Count:")
        all_counts = Counter()
        for split in ['train', 'val', 'test']:
            for cls, count in self.dataset_stats[split].items():
                all_counts[cls] += count
        
        for cls, count in all_counts.most_common(10):
            print(f"  {cls}: {count}")

def main():
    """Main execution function"""
    print("=" * 60)
    print("SmartCulinary AI - Multi-Dataset Merger")
    print("=" * 60)
    
    # Load configuration
    config = load_config()
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    freiburg_dir = project_root / config['paths']['freiburg_dir']
    grocery_store_dir = project_root / config['paths']['grocery_store_dir']
    output_dir = project_root / config['paths']['dataset_dir']
    
    # Check if datasets exist
    if not freiburg_dir.exists():
        print(f"\n✗ Freiburg dataset not found at: {freiburg_dir}")
        print("Please run: python data/download_dataset.py --dataset freiburg")
        return
    
    if not grocery_store_dir.exists():
        print(f"\n✗ Grocery Store dataset not found at: {grocery_store_dir}")
        print("Please run: python data/download_dataset.py --dataset grocery_store")
        print("Note: Requires Kaggle API credentials")
        return
    
    # Check if output already exists
    if output_dir.exists() and any(output_dir.iterdir()):
        response = input(f"\nMerged dataset already exists at {output_dir}. Recreate? (y/n): ")
        if response.lower() != 'y':
            print("Using existing merged dataset.")
            return
        else:
            shutil.rmtree(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Merge datasets
    merger = DatasetMerger(config)
    merger.merge_datasets(freiburg_dir, grocery_store_dir, output_dir)
    
    print("\n" + "=" * 60)
    print("Dataset Merging Complete!")
    print("=" * 60)
    print(f"\nMerged dataset location: {output_dir}")
    print("\nNext steps:")
    print("  1. Explore merged dataset: python data/data_explorer.py")
    print("  2. Start training: python train.py")

if __name__ == "__main__":
    main()
