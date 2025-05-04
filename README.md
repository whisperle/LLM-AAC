# LLM-AAC

This is a repository for the LLM Automated Audio Captioning project.

# Basic Pipeline

- Audio --> Audio Feature --> EAT ---> Audio Tokens + Prompt Tokens --> Linear Projector--> LLM (Vicuna) --> Caption --> CLAP --> Score --> Final Caption

# Dataset
- **AudioSet** 
  - Used for pretraining the EAT. 
  - Save Dir : /scratch/cl6707/Shared_Datasets/AudioSet_Wavelets & /scratch/cl6707/Shared_Datasets/AudioSet
- **Clotho**

# EAT Pretraining with Wavelet Features

## Overview
We've implemented a system for using Continuous Wavelet Transform (CWT) with Morlet wavelet features in the EAT model. These features may provide richer audio representations compared to standard mel-spectrograms.

## Feature Pre-computation (Recommended)
Pre-computing CWT features significantly speeds up training:

```bash
# Pre-compute features for AudioSet balanced split
bash fairseq/EAT/scripts/precompute_wavelets.sh --output_dir /scratch/cl6707/Shared_Datasets/AudioSet_Wavelets --config balanced

# Options:
# --config: balanced or unbalanced
# --split: train or test
# --cwt_scales: number of frequency scales (default: 64)
# --cwt_width: width parameter for Morlet wavelet (default: 8.0)
# --target_length: target time dimension length (default: 1024)
```

## Training with Pre-computed Features
Once features are pre-computed, run training with:

```bash
bash fairseq/EAT/scripts/pretraining_precomputed_cwt.sh
```

The script will:
1. Check if pre-computed features exist, generating them if needed
2. Run EAT pre-training using the prepared features
3. Save checkpoints to `/scratch/cl6707/Projects/LLM-AAC/checkpoints/eat_audioset_cwt_precomputed`

## Conclusion

TBD

