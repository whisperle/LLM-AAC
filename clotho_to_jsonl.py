#!/usr/bin/env python3
import os
import json
import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm

def convert_clotho_to_jsonl(dataset_dir, output_dir, audio_base_path=None, splits=None):
    """
    Convert Clotho dataset to JSONL format.
    
    Args:
        dataset_dir (str): Path to the Clotho dataset directory
        output_dir (str): Directory to save the JSONL files
        audio_base_path (str, optional): Base path to prepend to audio paths in the JSONL file.
                                       If None, uses the absolute path from dataset_dir.
        splits (list, optional): List of splits to process. If None, processes all splits.
    """
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define splits to process
    if splits is None:
        splits = ['development', 'validation', 'evaluation']
    
    # Process each split
    for split in splits:
        print(f"Processing {split} split...")
        
        # Path to captions CSV file
        captions_path = dataset_dir / f"clotho_captions_{split}.csv"
        
        # Check if captions file exists
        if not os.path.exists(captions_path):
            print(f"Warning: Captions file not found: {captions_path}")
            continue
        
        # Load captions
        captions_df = pd.read_csv(captions_path)
        
        # Path to audio directory
        audio_dir = dataset_dir / split
        
        # Path to output JSONL file
        output_file = output_dir / f"clotho_{split}.jsonl"
        
        # Convert to JSONL
        with open(output_file, 'w') as f_out:
            # Process each audio file
            for idx, row in tqdm(captions_df.iterrows(), total=len(captions_df)):
                file_name = row['file_name']
                file_id = os.path.splitext(file_name)[0]
                
                # Check if audio file exists
                audio_path = audio_dir / file_name
                if not os.path.exists(audio_path):
                    print(f"Warning: Audio file not found: {audio_path}")
                    continue
                
                # Define the source path
                if audio_base_path is not None:
                    source = os.path.join(audio_base_path, split, file_name)
                else:
                    source = str(audio_path.absolute())
                
                # Process each caption
                for i in range(1, 6):  # Clotho has 5 captions per audio
                    caption_key = f'caption_{i}'
                    if caption_key in row and not pd.isna(row[caption_key]):
                        # Create a unique key for this audio-caption pair
                        key = f"{file_id}_{i}"
                        
                        # Create JSON object
                        json_obj = {
                            "key": key,
                            "source": source,
                            "target": row[caption_key]
                        }
                        
                        # Write to JSONL file
                        f_out.write(json.dumps(json_obj) + '\n')
        
        print(f"JSONL file created: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Convert Clotho dataset to JSONL format')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Path to the Clotho dataset directory')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Directory to save the JSONL files')
    parser.add_argument('--audio_base_path', type=str, default=None,
                        help='Base path to prepend to audio paths in the JSONL file')
    parser.add_argument('--splits', type=str, nargs='+', default=None,
                        help='Splits to process (e.g., development validation evaluation)')
    
    args = parser.parse_args()
    
    convert_clotho_to_jsonl(
        args.dataset_dir,
        args.output_dir,
        args.audio_base_path,
        args.splits
    )

if __name__ == "__main__":
    main() 