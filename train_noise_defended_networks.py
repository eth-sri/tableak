import torch
import utils
import models
from defenses import dp_defense
from models import FullyConnected
from datasets import ADULT, German, Lawschool, HealthHeritage
import numpy as np
from utils import Timer, get_acc_and_bac
import os
import argparse


parser = argparse.ArgumentParser('noise_defended_training_parser')
parser.add_argument('--dataset', type=str, default='ADULT', help='Select the dataset')
parser.add_argument('--random_seed', type=int, default=42, help='Set the random state for reproducibility')
parser.add_argument('--force', action='store_true', help='If set to true, this will force the program to redo a given experiment')
args = parser.parse_args()


random_seed = args.random_seed
n_samples = 10
batch_size = 32
n_epochs = 10
noise_scales = [0.0, 0.001, 0.01, 0.1]
lr = 0.01

# set the random seed
np.random.seed(random_seed)
torch.manual_seed(random_seed)

datasets = {
    'ADULT': ADULT,
    'German': German,
    'Lawschool': Lawschool,
    'HealthHeritage': HealthHeritage
}

training_data = {}

dataset_name = args.dataset
dataset_class = datasets[args.dataset]

base_path = f'experiment_data/DP_experiments/{dataset_name}/'
os.makedirs(base_path, exist_ok=True)
save_path = base_path + f'trained_model_accuracies_{dataset_name}_{random_seed}.npy'

if os.path.isfile(save_path) and not args.force:
    print('experiment already conducted')
    training_data[dataset_name] = np.load(save_path)

else:
    
    dataset = dataset_class()
    dataset.standardize()

    X_train, y_train = dataset.get_Xtrain(), dataset.get_ytrain()
    X_test, y_test = dataset.get_Xtest(), dataset.get_ytest()

    criterion = torch.nn.CrossEntropyLoss()
    timer = Timer(len(noise_scales) * n_samples * n_epochs)
    collected_data = np.zeros((len(noise_scales), n_samples, n_epochs, 2))

    for i, noise_scale in enumerate(noise_scales):

        for j in range(n_samples):

            net = FullyConnected(X_train.size()[1], [100, 100, 2])
            n_batches = np.ceil(len(X_train)/batch_size).astype(int)
            acc, bac = 0.0, 0.0

            for l in range(n_epochs):
                timer.start()
                print(f'Noise Scale: {noise_scale}    Sample: {j+1}    Epoch: {l+1}    Acc: {100*acc:.1f}%    Bac: {100*bac:.1f}%    {timer}', end='\r')

                permutation_indices = np.random.permutation(len(X_train))
                X_train_permuted, y_train_permuted = X_train[permutation_indices].detach().clone(), y_train[permutation_indices].detach().clone()

                for k in range(n_batches):

                    # optimizer.zero_grad()
                    net.zero_grad()
                    X_batch, y_batch = X_train_permuted[k*batch_size:max(len(X_train_permuted), (k+1)*batch_size)], y_train_permuted[k*batch_size:max(len(X_train_permuted), (k+1)*batch_size)]
                    loss = criterion(net(X_batch), y_batch)

                    # defense
                    grad = [g.detach() for g in torch.autograd.grad(loss, net.parameters())]
                    perturbed_grad = dp_defense(grad, noise_scale) if noise_scale > 0 else grad

                    with torch.no_grad():
                        # make the update
                        for p, g in zip(net.parameters(), perturbed_grad):
                            p.data = p.data - lr * g

                acc, bac = get_acc_and_bac(net, X_test, y_test)
                collected_data[i, j, l] = acc, bac

                timer.end()
    timer.duration()
    np.save(save_path, collected_data)
