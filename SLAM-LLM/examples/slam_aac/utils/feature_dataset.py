import torch
import logging
import json
import sys
import os
import random
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np

# Add the path to the clotho_dataset.py file
sys.path.append('/gpfs/scratch/cl6707/Shared/clotho')
from clotho_dataset import ClothoFeatureDataset, ClothoAudioMetadataDataset

logger = logging.getLogger(__name__)

# Check if we should use metadata
USE_METADATA = os.environ.get('USE_METADATA', '1') == '1'

class SLAMFeatureDataset(Dataset):
    def __init__(self, 
                 tokenizer,
                 dataset_dir,
                 split='development',
                 feature_type='mel_spectrogram',  # 'audio', 'mel_spectrogram', or 'cwt'
                 max_length=256,
                 prompt="Describe the audio you hear.",
                 random_caption=True,
                 sr=16000,
                 max_audio_length=300000,
                 # Mel spectrogram params
                 n_fft=1024,
                 hop_length=512,
                 n_mels=64,
                 # CWT params
                 cwt_scales=None,
                 cwt_wavelet='morlet',
                 # Model integration params
                 encoder_dim=64,
                 encoder_projector_ds_rate=5):
        """
        Args:
            tokenizer: The tokenizer for text encoding
            dataset_dir: Path to the Clotho dataset directory
            split: Dataset split ('development', 'validation', or 'evaluation')
            feature_type: Type of feature to load ('audio', 'mel_spectrogram', or 'cwt')
            max_length: Maximum sequence length for tokenized output
            prompt: The instruction prompt to use
            random_caption: Whether to use a random caption (True) or just the first one (False)
            sr: Sample rate for audio
            max_audio_length: Maximum audio length in samples
            n_fft: FFT size for mel spectrogram
            hop_length: Hop length for mel spectrogram
            n_mels: Number of mel bands
            cwt_scales: Scales for CWT (None for default)
            cwt_wavelet: Wavelet to use for CWT
            encoder_dim: Feature dimension expected by the encoder (for reshaping)
            encoder_projector_ds_rate: Downsampling rate in encoder projector
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt = prompt
        self.random_caption = random_caption
        self.feature_type = feature_type
        
        # Store model integration parameters
        self.encoder_dim = encoder_dim
        self.encoder_projector_ds_rate = encoder_projector_ds_rate
        
        # Initialize the appropriate dataset based on feature_type
        try:
            if feature_type == 'cwt':
                # For CWT features, try using ClothoFeatureDataset instead if metadata is unavailable
                use_metadata = USE_METADATA
                if use_metadata:
                    try:
                        # For CWT features, use ClothoAudioMetadataDataset which has CWT extraction
                        self.feature_dataset = ClothoAudioMetadataDataset(
                            dataset_dir=dataset_dir,
                            split=split,
                            sr=sr,
                            max_audio_length=max_audio_length,
                            feature_type='cwt',
                            cwt_scales=cwt_scales,
                            cwt_wavelet=cwt_wavelet
                        )
                        self._manual_cwt = False
                    except (FileNotFoundError, UnicodeDecodeError) as e:
                        logger.warning(f"Error loading metadata: {e}. Falling back to ClothoFeatureDataset.")
                        use_metadata = False
                
                if not use_metadata:
                    self.feature_dataset = ClothoFeatureDataset(
                        dataset_dir=dataset_dir,
                        split=split,
                        sr=sr,
                        max_audio_length=max_audio_length,
                        extract_mel=False  # Using raw audio, we'll extract CWT manually
                    )
                    self._manual_cwt = True
                    self._cwt_scales = cwt_scales
                    self._cwt_wavelet = cwt_wavelet
            elif feature_type == 'mel_spectrogram':
                # For mel spectrogram features
                use_metadata = USE_METADATA
                if use_metadata:
                    try:
                        self.feature_dataset = ClothoAudioMetadataDataset(
                            dataset_dir=dataset_dir,
                            split=split,
                            sr=sr,
                            max_audio_length=max_audio_length,
                            feature_type='mel_spectrogram',
                            n_fft=n_fft,
                            hop_length=hop_length,
                            n_mels=n_mels
                        )
                        self._manual_mel = False
                    except (FileNotFoundError, UnicodeDecodeError) as e:
                        logger.warning(f"Error loading metadata: {e}. Falling back to ClothoFeatureDataset.")
                        use_metadata = False
                
                if not use_metadata:
                    self.feature_dataset = ClothoFeatureDataset(
                        dataset_dir=dataset_dir,
                        split=split,
                        sr=sr,
                        max_audio_length=max_audio_length,
                        extract_mel=True,  # Extract mel spectrogram
                        n_fft=n_fft,
                        hop_length=hop_length,
                        n_mels=n_mels
                    )
                    self._manual_mel = False  # Already handled by ClothoFeatureDataset
            elif feature_type == 'audio':
                # For raw audio waveform
                self.feature_dataset = ClothoFeatureDataset(
                    dataset_dir=dataset_dir,
                    split=split,
                    sr=sr,
                    max_audio_length=max_audio_length,
                    extract_mel=False  # Don't extract mel, use raw audio
                )
                self._manual_cwt = False
                self._manual_mel = False
            else:
                raise ValueError(f"Unsupported feature type: {feature_type}")
        except Exception as e:
            logger.error(f"Error initializing dataset: {e}")
            raise
            
        # Define the collator function as a method
        self.collator = self.collate_fn
            
    def __len__(self):
        return len(self.feature_dataset)
    
    def __getitem__(self, idx):
        # Get features and captions from feature dataset
        if hasattr(self, '_manual_cwt') and self._manual_cwt:
            # If we need to manually extract CWT features
            features, captions = self.feature_dataset[idx]
            try:
                import pywt
                # Check available wavelets and use a fallback if needed
                available_wavelets = pywt.wavelist(kind='continuous')
                wavelet_name = self._cwt_wavelet
                if wavelet_name not in available_wavelets:
                    logger.warning(f"Wavelet '{wavelet_name}' not found. Using 'morl' instead. Available wavelets: {available_wavelets}")
                    wavelet_name = 'morl'  # Default to 'morl' (Morlet wavelet)
                
                # Create a reasonable set of scales based on the signal length if not provided
                if self._cwt_scales is None:
                    scales = np.arange(1, min(128, len(features) // 10))
                else:
                    scales = self._cwt_scales
                
                # Compute CWT
                coef, _ = pywt.cwt(features.numpy(), scales, wavelet_name)
                
                # CWT coefficients can have complex values, take absolute for feature representation
                features = torch.tensor(np.abs(coef), dtype=torch.float32)
                logger.debug(f"Manually extracted CWT features with shape: {features.shape}")
            except ImportError:
                logger.warning("PyWavelets not installed. Using raw audio features instead.")
            except Exception as e:
                logger.warning(f"Error computing CWT: {e}. Using raw audio features instead.")
        elif self.feature_type == 'cwt' or self.feature_type == 'mel_spectrogram':
            try:
                # ClothoAudioMetadataDataset returns (features, captions, metadata)
                features, captions, _ = self.feature_dataset[idx]
            except ValueError:
                # If the unpack fails, assume it's returning (features, captions)
                features, captions = self.feature_dataset[idx]
        else:
            # ClothoFeatureDataset returns (features, captions)
            features, captions = self.feature_dataset[idx]
        
        # Debug print for initial feature shape
        logger.debug(f"Original feature shape: {features.shape}")
        
        # Filter out empty captions
        valid_captions = [c for c in captions if c.strip()]
        if not valid_captions:
            # Use a placeholder caption if all are empty
            valid_captions = ["Audio sound."]
        
        # Select caption - either random or first
        if self.random_caption and len(valid_captions) > 1:
            caption = random.choice(valid_captions)
        else:
            caption = valid_captions[0]
            
        # Create prompt + instruction
        tokenized_prompt = self.tokenizer.encode(self.prompt, add_special_tokens=False)
        tokenized_output = self.tokenizer.encode(caption, add_special_tokens=False)
        
        # Truncate if needed
        if len(tokenized_output) > self.max_length - len(tokenized_prompt) - 3:  # 3 for special tokens
            tokenized_output = tokenized_output[:self.max_length - len(tokenized_prompt) - 3]
        
        # Combine for model input
        input_ids = torch.tensor(tokenized_prompt + tokenized_output + [self.tokenizer.eos_token_id])
        labels = torch.tensor([-100] * len(tokenized_prompt) + tokenized_output + [self.tokenizer.eos_token_id])
        attention_mask = torch.ones_like(input_ids)
        
        # Mark prompt tokens for replacement with audio features
        modality_mask = torch.zeros_like(input_ids).bool()
        modality_mask[:len(tokenized_prompt)] = True
        
        # Handle features with different shapes
        # Make sure features have shape [B, L, D] where:
        # B = 1 (batch dimension), L = sequence length, D = feature dimension
        
        # First, convert to tensor properly if needed
        if isinstance(features, np.ndarray):
            features_tensor = torch.tensor(features, dtype=torch.float32)
        else:
            # Already a tensor, just clone and detach
            features_tensor = features.clone().detach()
            
        # Explicitly handle 2D features (CWT, mel_spectrogram)
        if len(features_tensor.shape) == 2:
            # Always treat the first dimension as frequency bins and second as time steps
            # This is standard for most audio features
            features_tensor = features_tensor.transpose(0, 1).unsqueeze(0)  # [F, T] -> [1, T, F]
            logger.debug(f"Converted 2D features to shape: {features_tensor.shape}")
        
        # For raw audio (1D), reshape to [1, L, 1]
        elif len(features_tensor.shape) == 1:
            features_tensor = features_tensor.unsqueeze(0).unsqueeze(-1)  # [L] -> [1, L, 1]
            logger.debug(f"Converted 1D features to shape: {features_tensor.shape}")
        
        # For 3D features, ensure shape is [B, T, F]
        elif len(features_tensor.shape) == 3:
            # For features with shape [B, F, T]
            if features_tensor.shape[1] > features_tensor.shape[2]:
                features_tensor = features_tensor.transpose(1, 2)  # [B, F, T] -> [B, T, F]
                logger.debug(f"Transposed 3D features to shape: {features_tensor.shape}")
        
        # For higher dimensions, flatten to [B, T, F]
        elif len(features_tensor.shape) > 3:
            old_shape = features_tensor.shape
            # Reshape to [B, T, -1]
            features_tensor = features_tensor.reshape(1, old_shape[1], -1)
            logger.debug(f"Reshaped features from {old_shape} to {features_tensor.shape}")
        
        # Ensure we have a 3D tensor with first dim as batch (1)
        if len(features_tensor.shape) < 3:
            # This shouldn't happen with our processing above, but just in case
            if len(features_tensor.shape) == 1:
                features_tensor = features_tensor.unsqueeze(0).unsqueeze(0)  # [D] -> [1, 1, D]
            else:  # 2D tensor
                features_tensor = features_tensor.unsqueeze(0)  # [L, D] -> [1, L, D]
        
        # Final check - ensure we have exactly 3 dimensions
        assert len(features_tensor.shape) == 3, f"Features must be 3D after processing, got {features_tensor.shape}"
        assert features_tensor.shape[0] == 1, f"Batch dim must be 1, got {features_tensor.shape}"

        # Add feature dimension processing for CWT and other features to match model's expected dimensions
        if self.feature_type == 'cwt':
            # For CWT, we need to reshape to match the expected dimensions for encoder_dim
            # First, we reshape to a more reasonable sequence length
            B, T, F = features_tensor.shape
            
            # Downsample the sequence (time) dimension to a reasonable length
            # Try to match encoder_projector_ds_rate from the model config (typically 5)
            encoder_ds_rate = getattr(self, 'encoder_projector_ds_rate', 5)
            target_len = min(1024, T // encoder_ds_rate)  # Reasonable sequence length factoring in downsampling rate
            
            # For extremely long sequences, use a more aggressive downsampling strategy
            if T > 10000:
                # Use adaptive pooling for very long sequences to ensure consistent output size
                pool = torch.nn.AdaptiveAvgPool1d(target_len)
                # Reshape for pooling: [B, T, F] -> [B, F, T] -> pool -> [B, F, target_len] -> [B, target_len, F]
                features_tensor = pool(features_tensor.transpose(1, 2)).transpose(1, 2)
                # logger.info(f"Used adaptive pooling to downsample CWT time dim from {T} to {features_tensor.shape[1]}")
            elif T > target_len:
                # For moderately long sequences, use strided sampling
                stride = max(1, T // target_len)
                features_tensor = features_tensor[:, ::stride, :]
                # logger.info(f"Downsampled CWT time dim from {T} to {features_tensor.shape[1]} using stride {stride}")
            
            # Ensure feature dimension matches encoder_dim (typically 64 for this model)
            target_feat_dim = self.encoder_dim  # Get from class variable
            if features_tensor.shape[2] != target_feat_dim:
                # If feature dim is larger, use average pooling to reduce it
                if features_tensor.shape[2] > target_feat_dim:
                    # Create a simple pooling layer and apply it
                    pool = torch.nn.AdaptiveAvgPool1d(target_feat_dim)
                    # Reshape for pooling (B, T, F) -> (B*T, F, 1) -> pool -> (B*T, target_feat_dim) -> reshape back
                    features_flat = features_tensor.reshape(-1, features_tensor.shape[2]).unsqueeze(-1)
                    pooled = pool(features_flat.transpose(1, 2)).transpose(1, 2).squeeze(-1)
                    features_tensor = pooled.reshape(B, -1, target_feat_dim)
                else:
                    # If smaller, pad with zeros
                    pad_size = target_feat_dim - features_tensor.shape[2]
                    features_tensor = torch.nn.functional.pad(features_tensor, (0, pad_size))
                
                # logger.info(f"Adjusted CWT feature dim from {F} to {features_tensor.shape[2]}")
        
        # Similarly for mel spectrograms if needed
        elif self.feature_type == 'mel_spectrogram':
            # Check if feature dimension needs adjustment
            target_feat_dim = self.encoder_dim  # Get from class variable
            if features_tensor.shape[2] != target_feat_dim:
                if features_tensor.shape[2] > target_feat_dim:
                    # Use average pooling to reduce dimension
                    pool = torch.nn.AdaptiveAvgPool1d(target_feat_dim)
                    features_flat = features_tensor.reshape(-1, features_tensor.shape[2]).unsqueeze(-1)
                    pooled = pool(features_flat.transpose(1, 2)).transpose(1, 2).squeeze(-1)
                    features_tensor = pooled.reshape(features_tensor.shape[0], -1, target_feat_dim)
                else:
                    # Pad if smaller
                    pad_size = target_feat_dim - features_tensor.shape[2]
                    features_tensor = torch.nn.functional.pad(features_tensor, (0, pad_size))
        
        # Log the final shape
        logger.debug(f"Final feature shape after processing: {features_tensor.shape}")
            
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "modality_mask": modality_mask,
            "audio_mel": features_tensor,
            "keys": str(idx),
            "targets": caption
        }
    
    def collate_fn(self, batch: list) -> dict:
        """
        Custom collate function for batching samples with variable lengths.
        
        Args:
            batch: List of dictionaries, each containing a single sample
            
        Returns:
            Batched data with appropriate padding
        """
        # Get max length for text sequences in this batch
        max_text_len = max([b["input_ids"].size(0) for b in batch])
        
        # Initialize tensors
        input_ids = []
        labels = []
        attention_mask = []
        modality_mask = []
        
        # Get feature dimensions from first batch item
        # Now all features should be in format [1, T, D] where T is time steps and D is feature dim
        first_mel = batch[0]["audio_mel"]
        batch_size = len(batch)
        feature_dim = first_mel.size(-1)
        
        # Find maximum sequence length across all items
        max_seq_len = max([b["audio_mel"].size(1) for b in batch])
        
        # Initialize audio tensor with standardized shape [B, T, D]
        audio_mel = torch.zeros((batch_size, max_seq_len, feature_dim), dtype=first_mel.dtype)
        
        # Pad text sequences and fill audio tensor
        for i, sample in enumerate(batch):
            # Text padding
            sample_len = sample["input_ids"].size(0)
            padding_len = max_text_len - sample_len
            
            # Pad text tensors
            input_ids.append(torch.cat([
                sample["input_ids"],
                torch.ones(padding_len, dtype=sample["input_ids"].dtype) * self.tokenizer.pad_token_id
            ]))
            
            labels.append(torch.cat([
                sample["labels"],
                torch.ones(padding_len, dtype=sample["labels"].dtype) * -100  # -100 is ignored in loss
            ]))
            
            attention_mask.append(torch.cat([
                sample["attention_mask"],
                torch.zeros(padding_len, dtype=sample["attention_mask"].dtype)
            ]))
            
            modality_mask.append(torch.cat([
                sample["modality_mask"],
                torch.zeros(padding_len, dtype=sample["modality_mask"].dtype)
            ]))
            
            # Audio features - all should now be in [1, T, D] format
            # Get actual sequence length for this sample
            seq_len = sample["audio_mel"].size(1)
            # Copy features to the batch tensor
            audio_mel[i, :seq_len, :] = sample["audio_mel"].squeeze(0)
        
        # Stack text tensors
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)
        attention_mask = torch.stack(attention_mask)
        modality_mask = torch.stack(modality_mask)
        
        keys = [b["keys"] for b in batch]
        targets = [b["targets"] for b in batch]
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "modality_mask": modality_mask,
            "audio_mel": audio_mel,
            "keys": keys,
            "targets": targets,
        }


def get_feature_audio_dataset(dataset_config, tokenizer, split="train", **kwargs):
    """Get train and validation datasets using feature extraction
    
    Args:
        dataset_config: Configuration for the dataset
        tokenizer: Tokenizer for text processing
        split: Which split to use ("train" or "validation"/"eval")
        **kwargs: Additional arguments
        
    Returns:
        Dataset for the requested split
    """
    # Map split names to Clotho split names
    clotho_split = {
        "train": "development",
        "validation": "validation",
        "test": "evaluation"
    }.get(split, "development")
    
    # Determine randomness based on split
    random_caption = split == "train"
    
    # Get wavelet name from config or use default
    cwt_wavelet = getattr(dataset_config, "cwt_wavelet", "morl")
    
    # Get encoder dimensions from config
    encoder_dim = getattr(dataset_config, "encoder_dim", 64)
    encoder_projector_ds_rate = getattr(dataset_config, "encoder_projector_ds_rate", 5)
    
    # Create the dataset
    dataset = SLAMFeatureDataset(
        tokenizer=tokenizer,
        dataset_dir=dataset_config.data_path,  # Path to Clotho dataset directory
        split=clotho_split,
        feature_type=dataset_config.feature_type,  # 'audio', 'mel_spectrogram', or 'cwt'
        max_length=dataset_config.target_length,
        prompt=dataset_config.prompt,
        random_caption=random_caption,
        # Audio parameters
        sr=16000,
        max_audio_length=dataset_config.fix_length_audio if dataset_config.fix_length_audio > 0 else 480000,  # Default to 30s
        # Mel spectrogram parameters
        n_fft=1024,
        hop_length=512,
        n_mels=dataset_config.mel_size,
        # CWT parameters
        cwt_scales=None,  # Use default scales
        cwt_wavelet=cwt_wavelet,  # Use wavelet name from config
        # Model integration parameters
        encoder_dim=encoder_dim,
        encoder_projector_ds_rate=encoder_projector_ds_rate
    )
    
    return dataset 