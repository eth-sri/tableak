#!/bin/bash

# Inverting Gradients
echo 'Running 1/3 -- Inverting Gradients'
python run_noise_defended_inversion_attacks.py --dataset $1 --experiment 0 --n_samples $2 --random_seed $3 --max_n_cpus $4 --first_cpu $5 --force

# TabLeak
echo 'Running 2/3 -- TabLeak'
python run_noise_defended_inversion_attacks.py --dataset $1 --experiment 46 --n_samples $2 --random_seed $3 --max_n_cpus $4 --first_cpu $5 --force

# Training the network for accuracy measurement
echo 'Running 3/3 -- Training networks'
python train_noise_defended_networks.py --dataset $1 --random_seed $3 --force