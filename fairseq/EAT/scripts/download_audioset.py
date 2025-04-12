#!/usr/bin/env python3
"""
Script to download AudioSet dataset from HuggingFace
"""

import os
import sys
import time
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Download AudioSet dataset from HuggingFace")
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="/scratch/cl6707/Shared_Datasets/AudioSet",
        help="Directory to save the downloaded dataset"
    )
    parser.add_argument(
        "--configs", 
        type=str, 
        default="balanced,unbalanced",
        help="Dataset configurations to download (comma-separated)"
    )
    args = parser.parse_args()
    
    # Check if output directory exists, create if not
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Try to import datasets, install if needed
    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing HuggingFace datasets library...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
        
        # Also install dependencies for audio loading
        print("Installing audio processing dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "soundfile", "librosa"])
        
        from datasets import load_dataset

    # Parse configurations
    configs = [cfg.strip() for cfg in args.configs.split(",")]
    
    # Download each configuration
    for config in configs:
        print(f"\n{'='*80}")
        print(f"Downloading AudioSet dataset with '{config}' configuration")
        print(f"{'='*80}")
        
        print(f"Destination: {args.output_dir}")
        
        try:
            start_time = time.time()
            # Load dataset (this will download it)
            dataset = load_dataset("agkphysics/AudioSet", config, cache_dir=args.output_dir)
            
            # Print dataset info
            print(f"\nDataset loaded successfully!")
            print(f"Configuration: {config}")
            print(f"Splits: {dataset.keys()}")
            for split in dataset:
                print(f"  - {split}: {len(dataset[split])} examples")
            
            duration = time.time() - start_time
            print(f"\nDownload completed in {duration:.2f} seconds ({duration/60:.2f} minutes)")
            
            # Try loading a sample to verify everything works
            print("\nVerifying dataset by loading a sample...")
            sample = dataset["train"][0]
            print(f"Sample ID: {sample['video_id']}")
            print(f"Audio shape: {sample['audio']['array'].shape}")
            print(f"Sample rate: {sample['audio']['sampling_rate']}")
            print(f"Labels: {sample['human_labels']}")
            
        except Exception as e:
            print(f"Error downloading configuration '{config}': {e}")
            continue
        
        print(f"\nConfiguration '{config}' successfully downloaded to {args.output_dir}")

    print("\nAll specified configurations downloaded!")
    print(f"Dataset is ready for use at: {args.output_dir}")

if __name__ == "__main__":
    main() 