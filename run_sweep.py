#!/usr/bin/env python3
"""
run_sweep.py
Launches W&B sweeps for the improved VGG6 CIFAR-10 experiments.
"""

import wandb

def main():
    sweep_config = {
        'method': 'bayes',
        'metric': {'name': 'val_acc', 'goal': 'maximize'},
        'parameters': {
            'activation': {'values': ['ReLU', 'Sigmoid', 'Tanh', 'SiLU', 'GELU']},
            'optimizer': {'values': ['SGD', 'Nesterov-SGD', 'Adam', 'Adagrad', 'RMSprop', 'Nadam']},
            'lr': {'values': [0.01, 0.001, 0.0001]},
            'batch_size': {'values': [64, 128, 256]},
            'batch_norm': {'values': [True, False]},
            'use_cutout': {'values': [True, False]},
            'weight_decay': {'values': [0.0, 0.0005]},
            'epochs': {'values': [20, 30, 50, 80]},
            'autoaugment': {'values': [True,False]}, # set true if you want to include CIFAR10Policy in experiments
            'seed': {'value': 42}
        },
        'program': 'train_vgg6.py'
    }

    project_name = "vgg6-cifar10_1"

    # Step 1: Create sweep
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    print(f"Sweep created with ID: {sweep_id}")

    # Step 2: Launch agent to run multiple experiments
    # You can adjust 'count' depending on how many runs you want
    wandb.agent(sweep_id, function=None, count=15)

if __name__ == "__main__":
    main()

