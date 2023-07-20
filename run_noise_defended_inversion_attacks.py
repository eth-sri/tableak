import warnings
warnings.filterwarnings('ignore')
import attacks
import os
import numpy as np
import torch
from torch import nn as nn
from utils import match_reconstruction_ground_truth, get_acc_and_bac, Timer, batch_feature_wise_accuracy_score, \
    post_process_continuous
from models import FullyConnected, FullyConnectedTrainer
from datasets import ADULT, Lawschool, HealthHeritage, German
import argparse
import pickle
import multiprocessing


def caller(x):
    return os.system(x)


def calculate_batch_inversion_performance_parallelized(dataset, network_layout, training_epochs,
                                                       training_batch_size, reconstruction_batch_sizes, noise_scales,
                                                       tolerance_map, n_samples, config, first_cpu, max_n_cpus,
                                                       metadata_path, device):
    """
    Calculates the gradient inversion errors for given target reconstruction batch sizes and training epochs on a given
    network architecture and dataset for a given reconstruction loss function.

    :param dataset: (datasets.BaseDataset) An instantiated child of the datasets.BaseDataset object.
    :param network_layout: (list) The layout of the fully connected neural network which we intend to invert the
        gradients of. Note: for now this function only support fully connected networks and does not take an nn.Module
        object as an argument instead. This would be the next step to make.
    :param training_epochs: (list) The training epochs until which we train the network before we try to invert its
        gradients. Note that after each inversion experiment at a given number of training epochs we reinstantiate the
        network and train it until the next target epoch. This removes some statistical bias from the process but takes
        longer to execute than just simply continuing training.
    :param training_batch_size: (int) The batch size used for training.
    :param reconstruction_batch_sizes: (list) A list of all batch sizes we want to estimate the recovery error for.
    :param noise_scales: (list) A list of the noise scales for DP.
    :param tolerance_map: (list) The tolerance map required to calculate the error between the guessed and the true
        batch.
    :param n_samples: (int) Number of monte carlo samples we take at each parameter setup, i.e. for a given number of
        training epochs and batch size, we try to invert this many independently gradients to estimate the mean and the
        standard deviation of the reconstruction error.
    :param config: (dict) The inversion configuration of the given experiment.
    :param first_cpu: (int) The first cpu index in the pool.
    :param max_n_cpus: (int) The number of cpus available to the process.
    :param metadata_path: (str) Path for the metadata created during the experiment.
    :param device: (str) The device on which the tensors in the dataset are located. Not used for now.
    :return: (np.ndarray) The estimated inversion error means and standard deviations. The returned array has the
        dimensions ((len(training_epochs), len(recover_batch_sizes), 3 + num_features, 5)), where the second to last
        dimensions store the data as (each (mean, std, median, min, max)): 0: complete batch error, 1: categorical
        feature error, 2: continuous feature error, and the rest of each individual features mean error.
    """

    collected_data = np.zeros((len(training_epochs), len(reconstruction_batch_sizes), len(noise_scales), 3 + len(dataset.train_features), 5))

    # get the data with which we are working
    Xtrain, ytrain = dataset.get_Xtrain(), dataset.get_ytrain()
    Xtest, ytest = dataset.get_Xtest(), dataset.get_ytest()

    timer = Timer(len(training_epochs) * len(reconstruction_batch_sizes)* len(noise_scales))

    for i, training_epoch in enumerate(training_epochs):

        # prepare, train and evaluate the network we are attacking
        net = FullyConnected(dataset.num_features, network_layout).to(device)
        optimizer = torch.optim.Adam(net.parameters())
        criterion = nn.CrossEntropyLoss()#weight=class_weight)
        trainer = FullyConnectedTrainer(data_x=Xtrain.detach().clone(), data_y=ytrain.detach().clone(),
                                        optimizer=optimizer, criterion=criterion, device=device, verbose=False)
        trainer.train(net, training_epoch, training_batch_size)
        acc, bac = get_acc_and_bac(net, Xtest, ytest)
        print(f'Pure Test Accuracy:       {np.around(acc * 100, 2)}%')
        print(f'Balanced Test Accuracy:   {np.around(bac * 100, 2)}%')

        # save all these things for the parallel processes to access
        curr_metadata_path = metadata_path + f'_epoch{training_epoch}'
        os.makedirs(curr_metadata_path, exist_ok=True)
        with open(f'{curr_metadata_path}/net.pickle', 'wb') as f:
            pickle.dump(net, f)
        with open(f'{curr_metadata_path}/criterion.pickle', 'wb') as f:
            pickle.dump(criterion, f)
        with open(f'{curr_metadata_path}/config.pickle', 'wb') as f:
            pickle.dump(config, f)
        with open(f'{curr_metadata_path}/dataset.pickle', 'wb') as f:
            pickle.dump(dataset, f)

        # now do the attacks per batch size
        for j, reconstruction_batch_size in enumerate(reconstruction_batch_sizes):

            for l, scale in enumerate(noise_scales):

                timer.start()
                print(timer, end='\r')

                os.makedirs(f'{curr_metadata_path}/scale_{scale}/batch_size_{reconstruction_batch_size}', exist_ok=True)

                recon_score_all = []
                recon_score_cat = []
                recon_score_cont = []
                per_feature_recon_scores = []

                # prepare the prompts
                random_seeds = np.random.randint(0, 15000, n_samples)
                sample_grouping = np.array_split(np.arange(n_samples), np.ceil(n_samples/max_n_cpus))
                for sample_group in sample_grouping:
                    process_pool = multiprocessing.Pool(processes=len(sample_group))
                    prompts = [f'taskset -c {cpu + first_cpu} python single_inversion_fedsgd.py --metadata_path {curr_metadata_path} --batch_size {reconstruction_batch_size} --sample {s} --random_seed {random_seeds[s]} --dp_defense --dp_scale {scale} --device {device}' for cpu, s in enumerate(sample_group)]
                    process_pool.map(caller, tuple(prompts))

                for s in range(n_samples):
                    target_batch = torch.tensor(np.load(f'{curr_metadata_path}/scale_{scale}/batch_size_{reconstruction_batch_size}/ground_truth_{reconstruction_batch_size}_{s}.npy'), device=device)
                    batch_recon = torch.tensor(np.load(f'{curr_metadata_path}/scale_{scale}/batch_size_{reconstruction_batch_size}/reconstruction_{reconstruction_batch_size}_{s}.npy'), device=device)

                    # postprocess the reconstruction and convert back to categorical features
                    target_batch_cat = dataset.decode_batch(target_batch, standardized=dataset.standardized)
                    batch_recon_cat = dataset.decode_batch(post_process_continuous(batch_recon, dataset),
                                                           standardized=dataset.standardized)

                    # perform the Hungarian algorithm to align the reconstructed batch with the ground truth and calculate the mean reconstruction score
                    batch_recon_cat, batch_cost_all, batch_cost_cat, batch_cost_cont = match_reconstruction_ground_truth(target_batch_cat, batch_recon_cat, tolerance_map)
                    recon_score_all.append(np.mean(batch_cost_all))
                    recon_score_cat.append(np.mean(batch_cost_cat))
                    recon_score_cont.append(np.mean(batch_cost_cont))

                    # calculate the reconstruction accuracy also per feature
                    per_feature_recon_scores.append(batch_feature_wise_accuracy_score(target_batch_cat, batch_recon_cat, tolerance_map, dataset.train_features))

                timer.end()

                collected_data[i, j, l, 0] = np.mean(recon_score_all), np.std(recon_score_all), np.median(recon_score_all), np.min(recon_score_all), np.max(recon_score_all)
                collected_data[i, j, l, 1] = np.mean(recon_score_cat), np.std(recon_score_cat), np.median(recon_score_cat), np.min(recon_score_cat), np.max(recon_score_cat)
                collected_data[i, j, l, 2] = np.mean(recon_score_cont), np.std(recon_score_cont), np.median(recon_score_cont), np.min(recon_score_cont), np.max(recon_score_cont)

                # aggregate and add the feature-wise data as well
                for k, feature_name in enumerate(dataset.train_features.keys()):
                    curr_feature_errors = [feature_errors[feature_name] for feature_errors in per_feature_recon_scores]
                    collected_data[i, j, l, 3 + k] = np.mean(curr_feature_errors), np.std(curr_feature_errors), np.median(curr_feature_errors), np.min(curr_feature_errors), np.max(curr_feature_errors)

    return collected_data


def main(args):
    print(args)

    datasets = {
        'ADULT': ADULT,
        'German': German,
        'Lawschool': Lawschool,
        'HealthHeritage': HealthHeritage
    }

    configs = {
        # Inverting Gradients
        0: {
            'reconstruction_loss': 'cosine_sim',
            'initialization_mode': 'uniform',
            'learning_rates': 0.06,
            'priors': None,
            'max_iterations': 1500,
            'optimization_mode': 'naive',
            'refill': 'fuzzy',
            'post_selection': 1,
            'return_all': False,
            'sign_trick': True,
            'weight_trick': False,
            'softmax_trick': False,
            'gumbel_softmax_trick': False,
            'temperature_mode': 'constant',
            'pooling': None,
            'perfect_pooling': False,
            'sigmoid_trick': False,
            'dp_noise_distribution': 'gaussian',
            'device': args.device,
            'invert_labels': False
        },
        # TabLeak
        46: {
            'reconstruction_loss': 'cosine_sim',
            'initialization_mode': 'uniform',
            'learning_rates': 0.06,
            'priors': [(8e-5, 'cont_uniform')],
            'max_iterations': 1500,
            'optimization_mode': 'naive',
            'refill': 'fuzzy',
            'post_selection': 30,
            'return_all': False,
            'sign_trick': True,
            'weight_trick': False,
            'softmax_trick': True,
            'gumbel_softmax_trick': False,
            'temperature_mode': 'constant',
            'pooling': 'median+softmax',
            'perfect_pooling': False,
            'dp_noise_distribution': 'gaussian',
            'sigmoid_trick': True,
            'device': args.device,
            'invert_labels': False
        }
    }

    # ------------ PARAMETERS ------------ #
    training_epochs = [0]
    training_batch_size = 256  # does not really matter for now
    reconstruction_batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    noise_scales = [1e-3, 1e-2, 1e-1]
    n_samples = args.n_samples  # monte carlo samples for any experiment involving randomness
    architecture_layout = [100, 100, 2]  # network architectures (fully connected)
    tol = 0.319
    # ------------ END ------------ #

    # get the configuration
    config = configs[args.experiment]

    # prepare the dataset
    dataset = datasets[args.dataset](device=args.device, random_state=args.random_seed)
    dataset.standardize()
    tolerance_map = dataset.create_tolerance_map(tol=tol)

    # set the random seed
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    # ------------ INVERSION EXPERIMENT ------------ #
    base_path = f'experiment_data/DP_experiments/{args.dataset}/experiment_{args.experiment}'
    os.makedirs(base_path, exist_ok=True)
    file_name_post_selection = config['post_selection']
    file_name_max_iterations = config['max_iterations']
    specific_file_path = base_path + f'/DP_data_all_{args.experiment}_{args.dataset}_{args.n_samples}_{file_name_post_selection}_{file_name_max_iterations}_{reconstruction_batch_sizes[-1]}_{noise_scales[-1]}_{tol}_{args.random_seed}.npy'
    if os.path.isfile(specific_file_path) and not args.force:
        print('This experiment has already been conducted')
    else:
        inversion_data = calculate_batch_inversion_performance_parallelized(
            dataset=dataset,
            network_layout=architecture_layout,
            training_epochs=training_epochs,
            training_batch_size=training_batch_size,
            reconstruction_batch_sizes=reconstruction_batch_sizes,
            tolerance_map=tolerance_map,
            n_samples=n_samples,
            config=config,
            max_n_cpus=args.max_n_cpus,
            first_cpu=args.first_cpu,
            noise_scales=noise_scales,
            metadata_path=base_path + f'/metadata_{args.experiment}_{args.dataset}_{args.n_samples}_{file_name_post_selection}_{file_name_max_iterations}_{reconstruction_batch_sizes[-1]}_{tol}_{args.random_seed}',
            device=args.device)
        np.save(specific_file_path, inversion_data)
    print('Complete                           ')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('run_defended_inversion_parser')
    parser.add_argument('--dataset', type=str, default='ADULT', help='Select the dataset')
    parser.add_argument('--experiment', type=int, help='Select the experiment you wish to run')
    parser.add_argument('--n_samples', type=int, help='Set the number of MC samples taken for each experiment')
    parser.add_argument('--random_seed', type=int, default=42, help='Set the random state for reproducibility')
    parser.add_argument('--force', action='store_true', help='If set to true, this will force the program to redo a given experiment')
    parser.add_argument('--device', type=str, default='cpu', help='Select the device to run the program on')
    parser.add_argument('--first_cpu', type=int, default=0, help='Mark the cpu at which the program starts')
    parser.add_argument('--max_n_cpus', type=int, default=50, help='Number of availables cores')
    in_args = parser.parse_args()
    main(in_args)
