import os
import h5py
import numpy as np
import pandas as pd
import librosa
import pywt
from tqdm import tqdm
import argparse
from pathlib import Path

def read_audio(audio_path, sr=16000):
    """
    Read audio file and return waveform
    """
    try:
        waveform, _ = librosa.load(audio_path, sr=sr, mono=True)
        return waveform
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return None

def extract_mel_spectrogram(waveform, sr=16000, n_fft=1024, hop_length=512, n_mels=64):
    """
    Extract mel-spectrogram from audio waveform
    """
    try:
        mel_spec = librosa.feature.melspectrogram(
            y=waveform, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )
        # Convert to dB scale
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec
    except Exception as e:
        print(f"Error extracting mel spectrogram: {e}")
        return None

def extract_cwt(waveform, wavelet='morl', scales=None):
    """
    Extract continuous wavelet transform
    """
    if scales is None:
        # Set default scales (can be adjusted)
        scales = np.arange(1, 128)
    
    try:
        # Downsample waveform for CWT (to reduce computation)
        # Using a factor of 8 as CWT can be computationally intensive
        downsampled = librosa.resample(waveform, orig_sr=16000, target_sr=16000//8)
        
        # Perform CWT
        coeffs, _ = pywt.cwt(downsampled, scales, wavelet)
        return coeffs
    except Exception as e:
        print(f"Error extracting CWT: {e}")
        return None

def process_dataset(dataset_dir, set_name, output_file, max_audio_length=np.inf):
    """
    Process a specific dataset (development, validation, or evaluation)
    and save to h5 file
    """
    dataset_path = os.path.join(dataset_dir, set_name)
    captions_path = os.path.join(dataset_dir, f"clotho_captions_{set_name}.csv")
    
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} does not exist")
        return
    
    if not os.path.exists(captions_path):
        print(f"Error: {captions_path} does not exist")
        return
    
    # Read captions
    captions_df = pd.read_csv(captions_path)
    
    # Create h5 file and groups
    with h5py.File(output_file, 'w') as h5f:
        # Create groups
        text_group = h5f.create_group('text')
        audio_group = h5f.create_group('audio')
        mel_spec_group = h5f.create_group('mel_spectrogram')
        cwt_group = h5f.create_group('cwt')
        
        # Add metadata
        h5f.attrs['set_name'] = set_name
        h5f.attrs['num_samples'] = len(captions_df)
        h5f.attrs['sample_rate'] = 16000
        
        # Process each audio file
        print(f"Processing {set_name} set...")
        for idx, row in tqdm(captions_df.iterrows(), total=len(captions_df)):
            file_name = row['file_name']
            audio_path = os.path.join(dataset_path, file_name)
            
            # Skip if file doesn't exist
            if not os.path.exists(audio_path):
                print(f"Warning: {audio_path} does not exist, skipping")
                continue
            
            # Read audio
            waveform = read_audio(audio_path)
            if waveform is None:
                continue
            
            # Extract features
            mel_spec = extract_mel_spectrogram(waveform)
            cwt_coeffs = extract_cwt(waveform)
            if mel_spec is None or cwt_coeffs is None:
                continue
            
            # Save to h5 file
            file_id = os.path.splitext(file_name)[0]  # Remove extension
            
            # Save audio waveform
            audio_group.create_dataset(file_id, data=waveform, compression='gzip')
            
            # Save mel spectrogram
            mel_spec_group.create_dataset(file_id, data=mel_spec, compression='gzip')
            
            # Save CWT coefficients
            cwt_group.create_dataset(file_id, data=cwt_coeffs, compression='gzip')
            
            # Save captions
            caption_group = text_group.create_group(file_id)
            for i in range(1, 6):  # Clotho has 5 captions per audio
                caption_key = f'caption_{i}'
                if caption_key in row:
                    caption_group.create_dataset(
                        caption_key, 
                        data=np.bytes_(row[caption_key])
                    )
        
        print(f"Finished processing {set_name} set")

def main():
    parser = argparse.ArgumentParser(description='Generate H5 file for Clotho dataset')
    parser.add_argument('--dataset_dir', type=str, default='clotho_dataset',
                        help='Path to the Clotho dataset directory')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Directory to save the H5 files')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each dataset
    for set_name in ['development', 'validation', 'evaluation']:
        output_file = os.path.join(args.output_dir, f'clotho_{set_name}.h5')
        process_dataset(args.dataset_dir, set_name, output_file)
    
    print("All datasets processed successfully!")

if __name__ == "__main__":
    main()
