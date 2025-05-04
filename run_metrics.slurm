#!/bin/bash
#SBATCH --job-name=compute_metrics
#SBATCH --output=metrics_%j.out
#SBATCH --error=metrics_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

# Load required modules
module load python/3.9

# Create and activate virtual environment
python -m venv metrics_env
source metrics_env/bin/activate

# Install required packages
pip install nltk rouge numpy

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('wordnet')"

# Run the metrics computation
python compute_metrics.py

# Deactivate virtual environment
deactivate 