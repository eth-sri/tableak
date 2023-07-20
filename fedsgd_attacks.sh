#!/bin/bash

# Inverting Gradients baselines with and without the true labels
echo 'Running 1/10 -- Inverting Gradients with true labels'
python run_inversion_attacks.py --dataset $1 --experiment 0 --n_samples $2 --random_seed $3 --max_n_cpus $4 --first_cpu $5 --force
echo 'Running 2/10 -- Inverting Gradients without true labels'
python run_inversion_attacks.py --dataset $1 --experiment 90 --n_samples $2 --random_seed $3 --max_n_cpus $4 --first_cpu $5 --force

# Deep Gradient Leakage baselines with and without the true labels
echo 'Running 3/10 -- Deep Gradient Leakage with true labels'
python run_inversion_attacks.py --dataset $1 --experiment 1000 --n_samples $2 --random_seed $3 --max_n_cpus $4 --first_cpu $5 --force
echo 'Running 4/10 -- Deep Gradient Leakage without true labels'
python run_inversion_attacks.py --dataset $1 --experiment 91000 --n_samples $2 --random_seed $3 --max_n_cpus $4 --first_cpu $5 --force

# TabLeak (no softmax) with and without the true labels
echo 'Running 5/10 -- TabLeak (no softmax) with true labels'
python run_inversion_attacks.py --dataset $1 --experiment 4103 --n_samples $2 --random_seed $3 --max_n_cpus $4 --first_cpu $5 --force
echo 'Running 6/10 -- TabLeak (no softmax) without true labels'
python run_inversion_attacks.py --dataset $1 --experiment 94103 --n_samples $2 --random_seed $3 --max_n_cpus $4 --first_cpu $5 --force

# TabLeak (no pooling) with and without the true labels
echo 'Running 7/10 -- TabLeak (no pooling) with true labels'
python run_inversion_attacks.py --dataset $1 --experiment 47 --n_samples $2 --random_seed $3 --max_n_cpus $4 --first_cpu $5 --force
echo 'Running 8/10 -- TabLeak (no softmax) without true labels'
python run_inversion_attacks.py --dataset $1 --experiment 947 --n_samples $2 --random_seed $3 --max_n_cpus $4 --first_cpu $5 --force

# TabLeak with and without the true labels
echo 'Running 9/10 -- TabLeak with true labels'
python run_inversion_attacks.py --dataset $1 --experiment 46 --n_samples $2 --random_seed $3 --max_n_cpus $4 --first_cpu $5 --force
echo 'Running 10/10 -- TabLeak without true labels'
python run_inversion_attacks.py --dataset $1 --experiment 946 --n_samples $2 --random_seed $3 --max_n_cpus $4 --first_cpu $5 --force
