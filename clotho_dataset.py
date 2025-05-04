import os
import pandas as pd
import numpy as np
import librosa
import torch
import h5py
from torch.utils.data import Dataset
from pathlib import Path


class ClothoDataset(Dataset):
    """Dataset class for Clotho audio captioning dataset.
    
    This class loads audio files directly from disk along with their captions.
    """
    
    def __init__(self, 
                 dataset_dir, 
                 split='development',
                 sr=16000, 
                 max_audio_length=300000,
                 transform=None,
                 target_transform=None):
        """
        Args:
            dataset_dir (str): Path to the Clotho dataset directory
            split (str): Dataset split ('development', 'validation', or 'evaluation')
            sr (int): Sample rate for audio loading
            max_audio_length (int): Maximum length for audio (will pad/trim)
            transform (callable, optional): Optional transform for audio
            target_transform (callable, optional): Optional transform for captions
        """
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.sr = sr
        self.max_audio_length = max_audio_length
        self.transform = transform
        self.target_transform = target_transform
        
        # Paths
        self.audio_dir = self.dataset_dir / split
        self.captions_path = self.dataset_dir / f"clotho_captions_{split}.csv"
        
        # Load captions
        if not os.path.exists(self.captions_path):
            raise FileNotFoundError(f"Captions file not found: {self.captions_path}")
        
        self.captions_df = pd.read_csv(self.captions_path)
        
        # Filter out entries where audio files don't exist
        valid_entries = []
        for idx, row in self.captions_df.iterrows():
            audio_path = self.audio_dir / row['file_name']
            if os.path.exists(audio_path):
                valid_entries.append(idx)
            else:
                print(f"Warning: Audio file not found: {audio_path}")
        
        self.captions_df = self.captions_df.iloc[valid_entries].reset_index(drop=True)
        
    def __len__(self):
        return len(self.captions_df)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset
        
        Returns:
            tuple: (audio, captions) where audio is the waveform and captions is a list of 5 caption strings
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # Get audio file path
        row = self.captions_df.iloc[idx]
        file_name = row['file_name']
        audio_path = self.audio_dir / file_name
        
        # Load audio
        waveform, _ = librosa.load(str(audio_path), sr=self.sr, mono=True)
        
        # Ensure consistent length (pad or trim)
        if len(waveform) > self.max_audio_length:
            waveform = waveform[:self.max_audio_length]
        else:
            pad_length = self.max_audio_length - len(waveform)
            waveform = np.pad(waveform, (0, pad_length), 'constant')
            
        # Extract captions
        captions = []
        for i in range(1, 6):  # Clotho has 5 captions per audio
            caption_key = f'caption_{i}'
            if caption_key in row:
                captions.append(row[caption_key])
            else:
                captions.append("")
                
        # Apply transformations if available
        if self.transform:
            waveform = self.transform(waveform)
        else:
            # Convert to tensor by default
            waveform = torch.tensor(waveform, dtype=torch.float32)
            
        if self.target_transform:
            captions = self.target_transform(captions)
            
        return waveform, captions
    
    def get_audio_path(self, idx):
        """Get the path to the audio file for a specific index"""
        row = self.captions_df.iloc[idx]
        return str(self.audio_dir / row['file_name'])
    
    def get_file_id(self, idx):
        """Get the file ID (without extension) for a specific index"""
        row = self.captions_df.iloc[idx]
        file_name = row['file_name']
        return os.path.splitext(file_name)[0]


class ClothoFeatureDataset(ClothoDataset):
    """Extended Clotho dataset that extracts features from audio"""
    
    def __init__(self, 
                 dataset_dir, 
                 split='development',
                 sr=16000, 
                 max_audio_length=300000,
                 extract_mel=True,
                 n_fft=1024, 
                 hop_length=512, 
                 n_mels=64,
                 transform=None,
                 target_transform=None):
        """
        Args:
            dataset_dir (str): Path to the Clotho dataset directory
            split (str): Dataset split ('development', 'validation', or 'evaluation')
            sr (int): Sample rate for audio loading
            max_audio_length (int): Maximum length for audio (will pad/trim)
            extract_mel (bool): Whether to extract mel spectrogram features
            n_fft (int): FFT size for mel spectrogram extraction
            hop_length (int): Hop length for mel spectrogram extraction
            n_mels (int): Number of mel bands to generate
            transform (callable, optional): Optional transform for features
            target_transform (callable, optional): Optional transform for captions
        """
        super().__init__(dataset_dir, split, sr, max_audio_length, transform, target_transform)
        self.extract_mel = extract_mel
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
    
    def __getitem__(self, idx):
        """Get a sample from the dataset with extracted features
        
        Returns:
            tuple: (audio_features, captions) where audio_features is the processed features
            and captions is a list of 5 caption strings
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # Get audio file path
        row = self.captions_df.iloc[idx]
        file_name = row['file_name']
        audio_path = self.audio_dir / file_name
        
        # Load audio
        waveform, _ = librosa.load(str(audio_path), sr=self.sr, mono=True)
        
        # Ensure consistent length (pad or trim)
        if len(waveform) > self.max_audio_length:
            waveform = waveform[:self.max_audio_length]
        else:
            pad_length = self.max_audio_length - len(waveform)
            waveform = np.pad(waveform, (0, pad_length), 'constant')
            
        # Extract features
        if self.extract_mel:
            # Extract mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=waveform, sr=self.sr, n_fft=self.n_fft, 
                hop_length=self.hop_length, n_mels=self.n_mels
            )
            # Convert to dB scale
            features = librosa.power_to_db(mel_spec, ref=np.max)
            features = torch.tensor(features, dtype=torch.float32)
        else:
            # Use raw waveform as features
            features = torch.tensor(waveform, dtype=torch.float32)
            
        # Extract captions
        captions = []
        for i in range(1, 6):  # Clotho has 5 captions per audio
            caption_key = f'caption_{i}'
            if caption_key in row:
                captions.append(row[caption_key])
            else:
                captions.append("")
                
        # Apply transformations if available
        if self.transform:
            features = self.transform(features)
            
        if self.target_transform:
            captions = self.target_transform(captions)
            
        return features, captions


class ClothoH5Dataset(Dataset):
    """Dataset class for Clotho dataset that loads from H5 files.
    
    This class loads pre-processed data from H5 files created with generate_h5.py
    """
    
    def __init__(self, 
                 h5_file_path,
                 feature_type='mel_spectrogram',  # 'audio', 'mel_spectrogram', or 'cwt'
                 transform=None,
                 target_transform=None):
        """
        Args:
            h5_file_path (str): Path to the H5 file
            feature_type (str): Type of feature to load ('audio', 'mel_spectrogram', or 'cwt')
            transform (callable, optional): Optional transform for features
            target_transform (callable, optional): Optional transform for captions
        """
        self.h5_file_path = Path(h5_file_path)
        self.feature_type = feature_type
        self.transform = transform
        self.target_transform = target_transform
        
        # Validate feature type
        valid_feature_types = ['audio', 'mel_spectrogram', 'cwt']
        if feature_type not in valid_feature_types:
            raise ValueError(f"feature_type must be one of {valid_feature_types}")
        
        # Open the H5 file and load metadata
        with h5py.File(self.h5_file_path, 'r') as h5f:
            # Store the set name
            self.set_name = h5f.attrs['set_name']
            
            # Get all file IDs
            self.file_ids = list(h5f[self.feature_type].keys())
            
            # Check if the features and text groups exist
            if self.feature_type not in h5f:
                raise ValueError(f"Feature group '{self.feature_type}' not found in H5 file")
            if 'text' not in h5f:
                raise ValueError("Text group not found in H5 file")
    
    def __len__(self):
        return len(self.file_ids)
    
    def __getitem__(self, idx):
        """Get a sample from the H5 dataset
        
        Returns:
            tuple: (features, captions) where features depend on feature_type
            and captions is a list of 5 caption strings
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        file_id = self.file_ids[idx]
        
        # Open the H5 file for reading
        with h5py.File(self.h5_file_path, 'r') as h5f:
            # Load features
            features = h5f[self.feature_type][file_id][()]
            
            # Load captions
            captions = []
            for i in range(1, 6):  # Clotho has 5 captions per audio
                caption_key = f'caption_{i}'
                if caption_key in h5f['text'][file_id]:
                    # Convert bytes to string
                    caption_bytes = h5f['text'][file_id][caption_key][()]
                    caption = caption_bytes.decode('utf-8') if isinstance(caption_bytes, bytes) else caption_bytes
                    captions.append(caption)
                else:
                    captions.append("")
        
        # Convert features to tensor
        features = torch.tensor(features, dtype=torch.float32)
        
        # Apply transformations if available
        if self.transform:
            features = self.transform(features)
            
        if self.target_transform:
            captions = self.target_transform(captions)
            
        return features, captions
    
    def get_file_id(self, idx):
        """Get the file ID for a specific index"""
        return self.file_ids[idx]


class ClothoAudioMetadataDataset(Dataset):
    """Dataset class for Clotho that loads audio files and metadata.
    
    This class loads audio files directly from disk along with their captions and metadata.
    """
    
    def __init__(self, 
                 dataset_dir, 
                 split='development',
                 sr=16000, 
                 max_audio_length=300000,
                 feature_type='audio',  # 'audio', 'mel_spectrogram', or 'cwt'
                 n_fft=1024,
                 hop_length=512,
                 n_mels=64,
                 cwt_scales=None,
                 cwt_wavelet='morlet',
                 transform=None,
                 target_transform=None):
        """
        Args:
            dataset_dir (str): Path to the Clotho dataset directory
            split (str): Dataset split ('development', 'validation', or 'evaluation')
            sr (int): Sample rate for audio loading
            max_audio_length (int): Maximum length for audio (will pad/trim)
            feature_type (str): Type of feature to return ('audio', 'mel_spectrogram', or 'cwt')
            n_fft (int): FFT size for mel spectrogram extraction
            hop_length (int): Hop length for mel spectrogram extraction
            n_mels (int): Number of mel bands to generate
            cwt_scales (list, optional): Scales for CWT. If None, default scales are computed
            cwt_wavelet (str): Wavelet to use for CWT extraction
            transform (callable, optional): Optional transform for features
            target_transform (callable, optional): Optional transform for captions
        """
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.sr = sr
        self.max_audio_length = max_audio_length
        self.feature_type = feature_type
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.cwt_scales = cwt_scales
        self.cwt_wavelet = cwt_wavelet
        self.transform = transform
        self.target_transform = target_transform
        
        # Validate feature type
        valid_feature_types = ['audio', 'mel_spectrogram', 'cwt']
        if feature_type not in valid_feature_types:
            raise ValueError(f"feature_type must be one of {valid_feature_types}")
        
        # Paths
        self.audio_dir = self.dataset_dir / split
        self.captions_path = self.dataset_dir / f"clotho_captions_{split}.csv"
        self.metadata_path = self.dataset_dir / f"clotho_metadata_{split}.csv"
        
        # Load captions
        if not os.path.exists(self.captions_path):
            raise FileNotFoundError(f"Captions file not found: {self.captions_path}")
        
        # Load metadata
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
        
        self.captions_df = pd.read_csv(self.captions_path)
        self.metadata_df = pd.read_csv(self.metadata_path)
        
        # Ensure file_name is the key in both dataframes
        if 'file_name' not in self.captions_df.columns:
            raise ValueError("captions CSV must contain 'file_name' column")
        if 'file_name' not in self.metadata_df.columns:
            raise ValueError("metadata CSV must contain 'file_name' column")
        
        # Filter out entries where audio files don't exist
        valid_entries = []
        for idx, row in self.captions_df.iterrows():
            audio_path = self.audio_dir / row['file_name']
            if os.path.exists(audio_path):
                valid_entries.append(idx)
            else:
                print(f"Warning: Audio file not found: {audio_path}")
        
        self.captions_df = self.captions_df.iloc[valid_entries].reset_index(drop=True)
        
        # Create a mapping from file_name to metadata row
        self.metadata_map = {row['file_name']: row for _, row in self.metadata_df.iterrows()}
        
    def extract_mel_spectrogram(self, waveform):
        """Extract mel spectrogram features from waveform"""
        mel_spec = librosa.feature.melspectrogram(
            y=waveform, sr=self.sr, n_fft=self.n_fft, 
            hop_length=self.hop_length, n_mels=self.n_mels
        )
        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db
    
    def extract_cwt(self, waveform):
        """Extract continuous wavelet transform features from waveform"""
        import pywt  # Import here to avoid dependency if not used
        
        # If scales are not provided, compute default scales
        if self.cwt_scales is None:
            # Create a reasonable set of scales based on the signal length
            # This is a simple heuristic, adjust as needed
            scales = np.arange(1, min(128, len(waveform) // 10))
        else:
            scales = self.cwt_scales
        
        # Compute CWT
        coef, _ = pywt.cwt(waveform, scales, self.cwt_wavelet)
        
        # CWT coefficients can have complex values, take absolute for feature representation
        cwt_features = np.abs(coef)
        
        return cwt_features
    
    def __len__(self):
        return len(self.captions_df)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset
        
        Returns:
            tuple: (features, captions, metadata) where features depend on feature_type,
                  captions is a list of 5 caption strings, and
                  metadata is a dictionary of metadata for the audio file
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # Get audio file path
        row = self.captions_df.iloc[idx]
        file_name = row['file_name']
        audio_path = self.audio_dir / file_name
        
        # Load audio
        waveform, _ = librosa.load(str(audio_path), sr=self.sr, mono=True)
        
        # Ensure consistent length (pad or trim)
        if len(waveform) > self.max_audio_length:
            waveform = waveform[:self.max_audio_length]
        else:
            pad_length = self.max_audio_length - len(waveform)
            waveform = np.pad(waveform, (0, pad_length), 'constant')
        
        # Extract features based on feature_type
        if self.feature_type == 'audio':
            features = waveform
        elif self.feature_type == 'mel_spectrogram':
            features = self.extract_mel_spectrogram(waveform)
        elif self.feature_type == 'cwt':
            features = self.extract_cwt(waveform)
        else:
            raise ValueError(f"Unsupported feature type: {self.feature_type}")
            
        # Extract captions
        captions = []
        for i in range(1, 6):  # Clotho has 5 captions per audio
            caption_key = f'caption_{i}'
            if caption_key in row:
                captions.append(row[caption_key])
            else:
                captions.append("")
        
        # Get metadata
        metadata = self.metadata_map.get(file_name, {})
        # Convert from pandas Series to dict if needed
        if hasattr(metadata, 'to_dict'):
            metadata = metadata.to_dict()
                
        # Apply transformations if available
        if self.transform:
            features = self.transform(features)
        else:
            # Convert to tensor by default
            features = torch.tensor(features, dtype=torch.float32)
            
        if self.target_transform:
            captions = self.target_transform(captions)
            
        return features, captions, metadata
    
    def get_audio_path(self, idx):
        """Get the path to the audio file for a specific index"""
        row = self.captions_df.iloc[idx]
        return str(self.audio_dir / row['file_name'])
    
    def get_file_id(self, idx):
        """Get the file ID (without extension) for a specific index"""
        row = self.captions_df.iloc[idx]
        file_name = row['file_name']
        return os.path.splitext(file_name)[0]


# Example usage
if __name__ == "__main__":
    # Dataset that loads from H5 file
    h5_dataset = ClothoH5Dataset(
        h5_file_path='/gpfs/scratch/cl6707/Shared/clotho/clotho_development.h5',
        feature_type='mel_spectrogram'
    )
    
    # Example of getting samples
    h5_features, h5_captions = h5_dataset[0]
    print(f"H5 features shape: {h5_features.shape}")
    print(f"H5 first caption: {h5_captions[0]}")
    
    # Print dataset information
    print(f"Number of samples in dataset: {len(h5_dataset)}")
    print(f"Dataset set name: {h5_dataset.set_name}")
    
    # Example of using the audio metadata dataset with raw audio
    audio_dataset = ClothoAudioMetadataDataset(
        dataset_dir='/gpfs/scratch/cl6707/Shared/clotho/clotho_dataset',
        split='development',
        feature_type='audio'
    )
    
    # Get a sample with audio, captions, and metadata
    audio, captions, metadata = audio_dataset[0]
    print(f"\nRaw audio shape: {audio.shape}")
    print(f"First caption: {captions[0]}")
    print(f"Metadata keys: {list(metadata.keys())}")
    
    # Example of using the audio metadata dataset with mel spectrogram features
    melspec_dataset = ClothoAudioMetadataDataset(
        dataset_dir='/gpfs/scratch/cl6707/Shared/clotho/clotho_dataset',
        split='development',
        feature_type='mel_spectrogram',
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )
    
    # Get a sample with mel spectrogram features
    mel_features, captions, _ = melspec_dataset[0]
    print(f"\nMel spectrogram features shape: {mel_features.shape}")
    
    try:
        # Example of using the audio metadata dataset with CWT features
        # Note: This requires PyWavelets library
        cwt_dataset = ClothoAudioMetadataDataset(
            dataset_dir='/gpfs/scratch/cl6707/Shared/clotho/clotho_dataset',
            split='development',
            feature_type='cwt',
            cwt_wavelet='morl'
        )
        
        # Get a sample with CWT features
        cwt_features, captions, _ = cwt_dataset[0]
        print(f"\nCWT features shape: {cwt_features.shape}")
    except ImportError:
        print("\nPyWavelets library not installed. Skipping CWT feature extraction example.") 