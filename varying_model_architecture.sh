#!/bin/bash

# Inverting Gradients
echo 'Running 1/3 -- Inverting Gradients -- 2 calls'
python run_network_size_variation.py --dataset $1 --experiment 0 --n_samples $2 --random_seed $3 --max_n_cpus $4 --first_cpu $5 --force
python run_network_type_variation.py --dataset $1 --experiment 0 --n_samples $2 --random_seed $3 --max_n_cpus $4 --first_cpu $5 --force

# GradInversion
echo 'Running 2/3 -- GradInversion -- 5 calls'
python run_attacks_with_priors.py --dataset $1 --experiment 3 --n_samples $2 --random_seed $3 --max_n_cpus $4 --first_cpu $5 --architecture 'linear' --force
python run_attacks_with_priors.py --dataset $1 --experiment 3 --n_samples $2 --random_seed $3 --max_n_cpus $4 --first_cpu $5 --architecture 'fc' --force
python run_attacks_with_priors.py --dataset $1 --experiment 3 --n_samples $2 --random_seed $3 --max_n_cpus $4 --first_cpu $5 --architecture 'fc_large' --force
python run_attacks_with_priors.py --dataset $1 --experiment 3 --n_samples $2 --random_seed $3 --max_n_cpus $4 --first_cpu $5 --architecture 'cnn' --force
python run_attacks_with_priors.py --dataset $1 --experiment 3 --n_samples $2 --random_seed $3 --max_n_cpus $4 --first_cpu $5 --architecture 'residual' --force

# TabLeak
echo 'Running 3/3 -- TabLeak -- 2 calls'
python run_network_size_variation.py --dataset $1 --experiment 46 --n_samples $2 --random_seed $3 --max_n_cpus $4 --first_cpu $5 --force
python run_network_type_variation.py --dataset $1 --experiment 46 --n_samples $2 --random_seed $3 --max_n_cpus $4 --first_cpu $5 --force
