#!/bin/bash

# Inverting Gradients baseline
echo 'Running 1/2 -- Inverting Gradients'
python run_fed_avg_attacks.py --dataset $1 --experiment 0 --n_samples $2 --random_seed $3 --max_n_cpus $4 --first_cpu $5 --force

# TabLeak
echo 'Running 2/2 -- TabLeak'
python run_fed_avg_attacks.py --dataset $1 --experiment 52 --n_samples $2 --random_seed $3 --max_n_cpus $4 --first_cpu $5 --force
