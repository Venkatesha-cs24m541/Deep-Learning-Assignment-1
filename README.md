# Deep-Learning-Assignment-1
---
### CS6886W — System Engineering for Deep Learning
### Assignment 1: VGG6 on CIFAR-10
### Department of Computer Science and Engineering, IIT Madras

---
## Overview
This repository implements VGG6 on the CIFAR-10 dataset to explore model performance under different configurations. <br>
Experiments include varying activation functions, optimizers, batch sizes, and learning rates, using Weights & Biases (W&B) for logging and visualization. <br>
The goal is to analyze how each configuration affects accuracy — and to identify the best-performing setup. <br>

---
## Environment Setup
 Clone the repository
git clone https://github.com/Venkatesha-cs24m541/Deep-Learning-Assignment-1

### Running from Drive
1. Upload all the files to a folder in Drive
2. %cd /content/drive/MyDrive/Semester_3/Deep_Learning/Assignment_1/
3. !pip install torch torchvision wandb matplotlib numpy wandb
4. import wandb
5. wandb.login()
6. To run the wandb sweep for first time - !python run_sweep.py
7. To continue configurations from previous sweep - !wandb agent venkatesha-cs24m541-iitm/vgg6-cifar10_1/fzu01va9 --count 7

### Running Directly
1. pip install torch torchvision wandb matplotlib numpy wandb
2. python
3. import wandb
4. wandb.login()
5. python run_sweep.py

---
 ## File Structure
 vgg6-cifar10-experiments <br>
 ┣ train_vgg6.py # Core training script <br> 
 ┣ sweep.yaml # W&B sweep configuration <br>
 ┣ run_sweep.py # Launches automated experiments <br>
 ┣ README.md # This documentation <br>
 ┣ best_model.pth # Saved best model (after training) <br>
 ┣ Plots/ # The report plots <br>
 ┣ Report_Runs_Results_Table.csv/ # The report Run Results Table <br>
 ┗ Report_Results_Observation.pdf/ # The report results and observations <br>

---
## Best Configuration (Based on W&B Parallel Plot)
Activation: GELU <br>
Optimizer: Nadam <br>
Batch size: 64 <br>
Epochs: 80 <br>
Learning rate: 0.001 <br>
Autoaugment: True <br>
Batch norm: True <br>
Use cutout: False <br>
Weight decay: 0.0 <br>
Seed: 42 <br>

---
## Reproducibility Details
Random Seed: 42 <br>
Hardware: NVIDIA GPU (Colab / RTX 3060 recommended) <br>
Framework: PyTorch 2.2+ <br>
Dataset: CIFAR-10 (auto-downloaded via torchvision) <br>
Logging: Weights & Biases (wandb) <br>

All runs are reproducible using the same random seed and environment configuration.

