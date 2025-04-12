#!/usr/bin/env python3
"""
Script to pre-compute CWT (Continuous Wavelet Transform) features for AudioSet
and calculate their normalization statistics.

This script:
1. Loads AudioSet from HuggingFace
2. Computes CWT features for each audio file
3. Calculates mean and std statistics across the dataset
4. Saves the pre-computed features to disk
5. Saves the normalization statistics

Usage:
    python precompute_wavelets.py --config balanced --output_dir /path/to/output --cwt_scales 64
"""

import os
import sys
import time
import argparse
import json
import h5py
import numpy as np
import torch
import torchaudio
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import gc


def compute_cwt(waveform, sample_rate=16000, scales=64, width=8.0, max_len=1024):
    """
    Compute Continuous Wavelet Transform using Morlet wavelet
    
    Args:
        waveform: Audio waveform (1D numpy array or tensor)
        sample_rate: Sample rate of the audio
        scales: Number of scales for the CWT
        width: Width parameter for Morlet wavelet
        max_len: Maximum length of the output (time dimension)
        
    Returns:
        CWT coefficient matrix as numpy array
    """
    import pywt
    
    # Ensure input is numpy array
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.numpy()
    
    # Ensure reasonable data length to prevent memory issues
    data_len = len(waveform)
    if data_len > 160000:  # Cap at ~10 seconds for 16kHz audio
        waveform = waveform[:160000]
        data_len = 160000
    
    # Create scales (frequency bands) - using a logarithmic scale
    scales = np.logspace(np.log10(4), np.log10(data_len // 2), num=scales)
    
    # Compute CWT with Morlet wavelet
    coef, _ = pywt.cwt(waveform, scales, 'morl', 1.0 / sample_rate)
    
    # Use absolute value (magnitude) of the complex CWT
    coef = np.abs(coef)
    
    # Reshape to target length through interpolation
    if max_len is not None:
        from scipy import signal as sps
        # Resample the time dimension to target_length
        if coef.shape[1] != max_len:
            coef = sps.resample(coef, max_len, axis=1)
    
    return coef


def process_audio_file(args):
    """
    Process a single audio file and extract CWT features.
    To be used with multiprocessing.
    """
    index, audio_data, sample_rate, cwt_scales, cwt_width, target_length = args
    
    try:
        # Convert to torch tensor if needed
        if not isinstance(audio_data, torch.Tensor):
            audio_data = torch.tensor(audio_data).float()
        
        # Downmix to mono if stereo
        if len(audio_data.shape) > 1 and audio_data.shape[0] > 1:
            audio_data = audio_data.mean(dim=0)
        
        # Resample to target sample rate (16kHz) if needed
        if sample_rate != 16000:
            audio_data = torchaudio.functional.resample(audio_data, orig_freq=sample_rate, new_freq=16000)
            sample_rate = 16000
        
        # Normalize the audio
        audio_data = audio_data - audio_data.mean()
        
        # Compute CWT
        cwt_features = compute_cwt(
            audio_data,
            sample_rate=sample_rate, 
            scales=cwt_scales,
            width=cwt_width,
            max_len=target_length
        )
        
        return index, cwt_features, True
    except Exception as e:
        print(f"Error processing index {index}: {str(e)}")
        return index, None, False


def main():
    parser = argparse.ArgumentParser(description="Pre-compute CWT features for AudioSet")
    parser.add_argument("--output_dir", type=str, default="/scratch/cl6707/Shared_Datasets/AudioSet_Wavelets", 
                        help="Directory to save pre-computed CWT features")
    parser.add_argument("--config", type=str, default="balanced", choices=["balanced", "unbalanced"],
                        help="AudioSet configuration (balanced or unbalanced)")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"],
                        help="Dataset split to process")
    parser.add_argument("--cwt_scales", type=int, default=64, 
                        help="Number of frequency scales for CWT")
    parser.add_argument("--cwt_width", type=float, default=8.0, 
                        help="Width parameter for Morlet wavelet")
    parser.add_argument("--target_length", type=int, default=1024, 
                        help="Target length for the time dimension")
    parser.add_argument("--num_workers", type=int, default=None, 
                        help="Number of workers for parallel processing. Default is CPU count - 1")
    parser.add_argument("--chunk_size", type=int, default=200, 
                        help="Number of samples to process in each chunk")
    parser.add_argument("--max_samples", type=int, default=None, 
                        help="Maximum number of samples to process (for testing)")
    args = parser.parse_args()
    
    # Set number of workers
    if args.num_workers is None:
        args.num_workers = max(1, cpu_count() - 1)
    
    # Create output directories
    output_base_dir = os.path.join(args.output_dir, args.config)
    output_split_dir = os.path.join(output_base_dir, args.split)
    os.makedirs(output_split_dir, exist_ok=True)
    
    stats_dir = os.path.join(output_base_dir, "stats")
    os.makedirs(stats_dir, exist_ok=True)
    
    # Install necessary packages if not available
    requirements = ["datasets", "pywt", "scipy", "h5py"]
    for package in requirements:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    # Import packages inside the function to ensure they're installed
    from datasets import load_dataset
    import pywt
    
    print(f"Loading AudioSet from HuggingFace (config: {args.config}, split: {args.split})...")
    dataset = load_dataset("agkphysics/AudioSet", args.config)
    split_dataset = dataset[args.split]
    
    total_samples = len(split_dataset)
    if args.max_samples is not None:
        total_samples = min(total_samples, args.max_samples)
    
    print(f"Processing {total_samples} samples with {args.num_workers} workers...")
    
    # Create HDF5 file for features
    features_file = os.path.join(output_base_dir, f"{args.split}_cwt_features.h5")
    with h5py.File(features_file, 'w') as f:
        # Add dataset attributes
        f.attrs['cwt_scales'] = args.cwt_scales
        f.attrs['cwt_width'] = args.cwt_width
        f.attrs['target_length'] = args.target_length
        f.attrs['config'] = args.config
        f.attrs['split'] = args.split
        
        # Create a group for features
        features_group = f.create_group("features")
        
        # Create a group for metadata
        metadata_group = f.create_group("metadata")
        
        # Create datasets for statistics calculation
        sum_dataset = f.create_dataset("stats/sum", shape=(args.cwt_scales, args.target_length), dtype=np.float64)
        sum_squares_dataset = f.create_dataset("stats/sum_squares", shape=(args.cwt_scales, args.target_length), dtype=np.float64)
        count_dataset = f.create_dataset("stats/count", shape=(1,), dtype=np.int64)
        
        # Initialize statistics arrays
        sum_cwt = np.zeros((args.cwt_scales, args.target_length), dtype=np.float64)
        sum_squares_cwt = np.zeros((args.cwt_scales, args.target_length), dtype=np.float64)
        count = 0
        
        # Process in chunks to manage memory
        for start_idx in range(0, total_samples, args.chunk_size):
            end_idx = min(start_idx + args.chunk_size, total_samples)
            chunk_size = end_idx - start_idx
            
            print(f"Processing chunk {start_idx//args.chunk_size + 1}/{(total_samples-1)//args.chunk_size + 1} (samples {start_idx}-{end_idx-1})...")
            
            # Prepare task arguments
            tasks = []
            for i in range(start_idx, end_idx):
                sample = split_dataset[i]
                audio_data = sample["audio"]["array"]
                sample_rate = sample["audio"]["sampling_rate"]
                video_id = sample["video_id"]
                
                # Store metadata
                metadata_dict = {
                    "video_id": video_id,
                    "labels": sample["labels"],
                    "human_labels": sample["human_labels"]
                }
                metadata_json = json.dumps(metadata_dict)
                metadata_group.create_dataset(f"{i}", data=metadata_json.encode('utf-8'))
                
                tasks.append((i, audio_data, sample_rate, args.cwt_scales, args.cwt_width, args.target_length))
            
            # Process audio files in parallel
            with Pool(args.num_workers) as pool:
                results = list(tqdm(pool.imap(process_audio_file, tasks), total=len(tasks), desc="Calculating CWT"))
            
            # Store results and update statistics
            for idx, cwt_features, success in results:
                if success and cwt_features is not None:
                    # Store feature
                    features_group.create_dataset(f"{idx}", data=cwt_features, compression="gzip", compression_opts=9)
                    
                    # Update statistics
                    sum_cwt += cwt_features
                    sum_squares_cwt += cwt_features ** 2
                    count += 1
            
            # Clear memory
            del tasks, results
            gc.collect()
        
        # Save accumulated statistics to datasets
        sum_dataset[:] = sum_cwt
        sum_squares_dataset[:] = sum_squares_cwt
        count_dataset[0] = count
        
        # Calculate and save mean and std
        if count > 0:
            mean_cwt = sum_cwt / count
            std_cwt = np.sqrt((sum_squares_cwt / count) - (mean_cwt ** 2))
            
            # Handle potential numerical issues
            std_cwt = np.clip(std_cwt, 1e-8, None)
            
            f.create_dataset("stats/mean", data=mean_cwt)
            f.create_dataset("stats/std", data=std_cwt)
            
            # Calculate global scalar statistics
            global_mean = float(np.mean(mean_cwt))
            global_std = float(np.mean(std_cwt))
            
            f.attrs['global_mean'] = global_mean
            f.attrs['global_std'] = global_std
            
            print(f"Global statistics: mean={global_mean:.6f}, std={global_std:.6f}")
            
            # Save global stats to a separate JSON file for easy access
            stats_file = os.path.join(stats_dir, f"{args.split}_cwt_stats.json")
            with open(stats_file, 'w') as sf:
                json.dump({
                    "global_mean": global_mean,
                    "global_std": global_std,
                    "cwt_scales": int(args.cwt_scales),
                    "cwt_width": float(args.cwt_width),
                    "target_length": int(args.target_length),
                    "config": args.config,
                    "split": args.split,
                    "sample_count": int(count),
                    "total_samples": int(total_samples),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }, sf, indent=2)
    
    print(f"\nProcessing complete! Pre-computed features saved to: {features_file}")
    print(f"Statistics saved to: {stats_dir}")
    print(f"Successfully processed {count} out of {total_samples} samples")
    
    # Create symlink to latest features
    latest_link = os.path.join(args.output_dir, f"latest_{args.config}_{args.split}_cwt_features.h5")
    if os.path.exists(latest_link):
        os.remove(latest_link)
    os.symlink(features_file, latest_link)
    
    print(f"Created symlink: {latest_link}")


if __name__ == "__main__":
    main() 