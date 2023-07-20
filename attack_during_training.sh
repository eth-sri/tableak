#!/bin/bash

echo 'Running 1/2 -- Inverting Gradients'
python run_attack_over_training.py --dataset $1 --experiment 20 --n_samples $2 --random_seed $3 --max_n_cpus $4 --first_cpu $5 --force

echo 'Running 2/2 -- Inverting Gradients'
python run_attack_over_training.py --dataset $1 --experiment 52 --n_samples $2 --random_seed $3 --max_n_cpus $4 --first_cpu $5 --force