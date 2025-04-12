# LLM-AAC

This is a repository for the LLM Automated Audio Captioning project.

# Basic Pipeline

- Audio --> Audio Feature --> EAT ---> Audio Tokens + Prompt Tokens --> Linear Projector--> LLM (Vicuna) --> Caption --> CLAP --> Score --> Final Caption

# Dataset
- **AudioSet** 
  - Used for pretraining the EAT. 
  - Save Dir : /scratch/cl6707/Shared_Datasets/AudioSet_Wavelets & /scratch/cl6707/Shared_Datasets/AudioSet
- **Clotho**

## TODO

- Pretrain the EAT with Wavelet Features 
  - Under folder fairseq/EAT
- Replace the Linear Projector with a Q-former projector
  - TBD
- Finetune the LLM with LORA
  - TBD

