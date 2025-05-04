import os
import h5py
import numpy as np
import json
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

def process_evaluation(jsonl_path, output_file):
    """
    Process evaluation data from jsonl file and save to h5 file
    """
    if not os.path.exists(jsonl_path):
        print(f"Error: {jsonl_path} does not exist")
        return
    
    # Create h5 file and groups
    with h5py.File(output_file, 'w') as h5f:
        # Create groups
        text_group = h5f.create_group('text')
        audio_group = h5f.create_group('audio')
        mel_spec_group = h5f.create_group('mel_spectrogram')
        cwt_group = h5f.create_group('cwt')
        
        # Add metadata
        h5f.attrs['set_name'] = 'evaluation'
        h5f.attrs['sample_rate'] = 16000
        
        # Read and process jsonl file
        print("Processing evaluation set...")
        with open(jsonl_path, 'r') as f:
            lines = f.readlines()
            h5f.attrs['num_samples'] = len(lines)
            
            for line in tqdm(lines):
                data = json.loads(line)
                audio_path = data['source']
                file_id = data['key']
                
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
                # Save audio waveform
                audio_group.create_dataset(file_id, data=waveform, compression='gzip')
                
                # Save mel spectrogram
                mel_spec_group.create_dataset(file_id, data=mel_spec, compression='gzip')
                
                # Save CWT coefficients
                cwt_group.create_dataset(file_id, data=cwt_coeffs, compression='gzip')
                
                # Save caption
                caption_group = text_group.create_group(file_id)
                caption_group.create_dataset('caption', data=np.bytes_(data['target']))
        
        print("Finished processing evaluation set")

def main():
    parser = argparse.ArgumentParser(description='Generate H5 file for Clotho evaluation dataset')
    parser.add_argument('--jsonl_path', type=str, default='clotho/evaluation_single.jsonl',
                        help='Path to the evaluation jsonl file')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Directory to save the H5 file')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process evaluation dataset
    output_file = os.path.join(args.output_dir, 'clotho_evaluation.h5')
    process_evaluation(args.jsonl_path, output_file)
    
    print("Evaluation dataset processed successfully!")

if __name__ == "__main__":
    main() 