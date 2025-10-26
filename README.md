# Deep-Learning-Assignment-1
---
### CS6886W — System Engineering for Deep Learning
### Assignment 1: VGG6 on CIFAR-10
### Department of Computer Science and Engineering, IIT Madras

---
## Overview
This repository implements VGG6 on the CIFAR-10 dataset to explore model performance under different configurations.
Experiments include varying activation functions, optimizers, batch sizes, and learning rates, using Weights & Biases (W&B) for logging and visualization.
The goal is to analyze how each configuration affects accuracy — and to identify the best-performing setup.

---
## Environment Setup
 Clone the repository
git clone https://github.com/Venkatesha-cs24m541/Deep-Learning-Assignment-1

Running from Drive
1. Upload all the files to a folder in Drive
2. %cd /content/drive/MyDrive/Semester_3/Deep_Learning/Assignment_1/
3. !pip install torch torchvision wandb matplotlib numpy wandb
4. import wandb
5. wandb.login()
6. To run the wandb sweep for first time - !python run_sweep.py
7. To continue configurations from previous sweep - !wandb agent venkatesha-cs24m541-iitm/vgg6-cifar10_1/fzu01va9 --count 7

---
 ## File Structure
 vgg6-cifar10-experiments <br>
 ┣ train_vgg6.py # Core training script <br> 
 ┣ sweep.yaml # W&B sweep configuration <br>
 ┣ run_sweep.py # Launches automated experiments <br>
 ┣ README.md # This documentation <br>
 ┣ best_model.pth # Saved best model (after training) <br>
 ┗ data/ # Automatically downloaded CIFAR-10 dataset <br>

---
## Running the Baseline Experiment
To train the VGG6 model with a single configuration:
python train_vgg6.py --activation ReLU --optimizer Adam --lr 0.001 --batch_size 128 --epochs 50
You can modify arguments as:
--activation [ReLU | Sigmoid | Tanh | SiLU | GELU]--optimizer [SGD | Nesterov-SGD | Adam | Adagrad | RMSprop | Nadam]--lr Learning rate (default: 0.001)--batch_size Batch size (default: 128)--epochs Number of epochs (default: 50)
 Output:
Logs training & validation metrics to W&B
Saves the best model weights as best_model.pth


---
## Running Automated Sweeps (W&B)
 Create the sweep
wandb sweep sweep.yaml
This will print a SWEEP_ID.
 Launch the sweep agent
wandb agent <your-wandb-username>/<project-name>/<SWEEP_ID>
OR simply run:
python run_sweep.py
 Each sweep run:
Trains a new configuration
Logs metrics automatically
Contributes to W&B plots such as:
Parallel Coordinates (accuracy vs configuration)
Validation Accuracy vs Step
Loss/Accuracy curves



---
## Results and Plots
All experiments are tracked in the W&B project dashboard.
You should include the following plots in your final report:
Plot	Description
 Parallel Coordinates	Shows how each configuration (activation, optimizer, LR, etc.) affects accuracy Validation Accuracy vs Step	Illustrates convergence rate for different configs Training/Validation Curves	Shows overfitting trends and stability

Example table for report:
Activation	Optimizer	LR	Batch Size	Val Accuracy
ReLU	Adam	0.001	128	87.3%SiLU	RMSprop	0.001	128	86.8%GELU	Adam	0.001	256	86.5%Tanh	SGD	0.01	64	78.2%


---
## Best Configuration (Based on W&B Parallel Plot)
Parameter	Best Value
Activation	ReLUOptimizer	AdamLearning Rate	0.001Batch Size	128Epochs	50Validation Accuracy	87.3%

 This configuration was re-run independently and reproduced the same accuracy.

---
## Reproducibility Details
Random Seed: 42
Hardware: NVIDIA GPU (Colab / RTX 3060 recommended)
Framework: PyTorch 2.2+
Dataset: CIFAR-10 (auto-downloaded via torchvision)
Logging: Weights & Biases (wandb)

All runs are reproducible using the same random seed and environment configuration.

