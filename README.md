# Wavenet Torch Implementation for Gravitational Wave Classification

## Introduction
This project is a PyTorch implementation of the Wavenet architecture, specifically designed for the classification of gravitational waves. Based on arXiv:2306.15728, this implementation focuses on providing a robust solution for the classification of gravitational wave signals.

## Installation
To get started with this project, clone the repository and install the required dependencies:
```
git clone https://github.com/victoria-tiki/Wavenet_torch.git
cd Wavenet_torch
pip install -r requirements.txt
```

## Usage
This project includes slurm scripts for both training and inference, facilitating its use in high-performance computing environments. To train the model, execute:
```
sbatch slurm_train.sh
```
For inference, run:
```
sbatch slurm_inference.sh
```
Please adjust the slurm script parameters according to your specific computational environment and requirements.

