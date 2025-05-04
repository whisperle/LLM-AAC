# LLM-AAC

This is a repository for the LLM Automated Audio Captioning project.

Team Members: Nika Emami, Chuyang Chen, Chenqian Le

# Basic Pipeline

- Audio --> Audio Feature --> Projector--> LLM (Vicuna) --> Caption --> CLAP --> Score --> Final Caption
- [TO BE UPDATED with pipeline figure]
# Dataset
- **Clotho**

## Feature Pre-computation (Recommended)
Pre-computing CWT features significantly speeds up training, saved to h5 file with `generate_h5.py` and `generate_h5_eval.py` .


## Training with Pre-computed Features
Bash files are saved in `/gpfs/scratch/cl6707/Projects/LLM-AAC/LLM-AAC/SLAM-LLM/examples/slam_aac/scripts`

For example, `finetune_clotho_qformer_cwt.sh` is for finetuning the Q-former with pre-computed CWT features.

The script will:
1. Run Q-former projector training and LLM finetuning
2. Save checkpoints to `exps`

## Evaluation

Evaluation is done with `compute_metrics.py`

## Results


| Method | METEOR | CIDEr | SPIDEr | SPIDEr-FL | FENSE |
|--------|---------|--------|---------|------------|--------|
| CWT + MLP | 24.54 | 29.87 | 14.94 | 14.93 | 20.58 |
| Mel Spec + MLP | 13.94 | 19.77 | 9.88 | 9.87 | 11.43 |
| CWT + Q-former | 26.88 | 30.77 | 15.38 | 15.02 | 22.47 |


## Conclusion

TBD

