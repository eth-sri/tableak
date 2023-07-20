import sys
import attacks
# sys.path.append("..")
import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
import torch
import numpy as np
from attacks.inversion_losses import _squared_error_loss, _cosine_similarity_loss, _weighted_CS_SE_loss, \
    _gradient_norm_weighted_CS_SE_loss
from attacks.initializations import _uniform_initialization, _gaussian_initialization, _mean_initialization, \
    _dataset_sample_initialization, _likelihood_prior_sample_initialization, _mixed_initialization, \
    _best_sample_initialization
from attacks.ensembling import pooled_ensemble
from attacks.fed_avg_inversion_attack import simulate_local_training_for_attack
import argparse
import pickle
import copy
from models.monkey import MetaMonkey
from utils import continuous_sigmoid_bound


def main(args):

    # attack setups
    rec_loss_function = {
        'squared_error': _squared_error_loss,
        'cosine_sim': _cosine_similarity_loss,
        'weighted_combined': _weighted_CS_SE_loss,
        'norm_weighted_combined': _gradient_norm_weighted_CS_SE_loss
    }

    initialization = {
        'uniform': _uniform_initialization,
        'gaussian': _gaussian_initialization,
        'mean': _mean_initialization,
        'dataset_sample': _dataset_sample_initialization,
        'likelihood_sample': _likelihood_prior_sample_initialization,
        'mixed': _mixed_initialization,
        'best_sample': _best_sample_initialization
    }

    temperature_configs = {
        'cool': (1000., 0.98),
        'constant': (1., 1.),
        'heat': (0.1, 1.01)
    }

    # we need to load all the necessary stuff
    with open(f'{args.metadata_path}/original_net.pickle', 'rb') as f:
        original_net = pickle.load(f)
    with open(f'{args.metadata_path}/attacked_clients_params.pickle', 'rb') as f:
        attacked_clients_params = pickle.load(f)
    with open(f'{args.metadata_path}/per_client_ground_truth_data.pickle', 'rb') as f:
        per_client_ground_truth_data = pickle.load(f)
    with open(f'{args.metadata_path}/per_client_ground_truth_labels.pickle', 'rb') as f:
        per_client_ground_truth_labels = pickle.load(f)
    with open(f'{args.metadata_path}/dataset.pickle', 'rb') as f:
        dataset = pickle.load(f)
    with open(f'{args.metadata_path}/apply_projection_to_features.pickle', 'rb') as f:
        apply_projection_to_features = pickle.load(f)
    # prior could also be None
    if os.path.isfile(f'{args.metadata_path}/priors.pickle'):
        with open(f'{args.metadata_path}/priors.pickle', 'rb') as f:
            priors = pickle.load(f)
    else:
        priors = None
    # epoch matching prior could also be None
    if os.path.isfile(f'{args.metadata_path}/epoch_matching_prior.pickle'):
        with open(f'{args.metadata_path}/epoch_matching_prior.pickle', 'rb') as f:
            epoch_matching_prior = pickle.load(f)
    else:
        epoch_matching_prior = None

    # set the random seed
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    # extract all the data from the loaded data that is relevant to this client
    attacked_client_params = attacked_clients_params[args.client]
    ground_truth_labels = per_client_ground_truth_labels[args.client]
    ground_truth_data = per_client_ground_truth_data[args.client]

    # fix the client network and extract its starting parameters
    original_params = [param.detach().clone() for param in original_net.parameters()]
    true_two_point_gradient = [(original_param - new_param).detach().clone() for original_param, new_param in zip(original_params, attacked_client_params)]

    # we reconstruct independently in each epoch and aggregate in the end, as per Dimitrov et al.
    # initialize the data
    reconstructed_data_per_epoch = [initialization[args.initialization_mode](ground_truth_data, dataset, args.device) for _ in range(args.n_local_epochs)]
    for reconstructed_data in reconstructed_data_per_epoch:
        reconstructed_data.requires_grad = True

    optimizer = torch.optim.Adam(reconstructed_data_per_epoch, lr=args.attack_learning_rate)

    T = temperature_configs[args.temperature_mode][0]

    for it in range(args.attack_iterations):

        optimizer.zero_grad()
        original_net.zero_grad()
        client_net = MetaMonkey(copy.deepcopy(original_net))
        criterion = torch.nn.CrossEntropyLoss()

        resulting_two_point_gradient, regularizer = simulate_local_training_for_attack(
            client_net=client_net,
            lr=args.lr,
            criterion=criterion,
            dataset=dataset,
            labels=ground_truth_labels,
            original_params=original_params,
            reconstructed_data_per_epoch=reconstructed_data_per_epoch,
            local_batch_size=args.local_batch_size,
            priors=priors,
            epoch_matching_prior=epoch_matching_prior,
            softmax_trick=args.softmax_trick,
            gumbel_softmax_trick=args.gumbel_softmax_trick,
            sigmoid_trick=args.sigmoid_trick,
            apply_projection_to_features=apply_projection_to_features,
            temperature=T
        )

        # calculate the final objective
        loss = rec_loss_function[args.reconstruction_loss](resulting_two_point_gradient, true_two_point_gradient, args.device)
        loss += regularizer
        loss.backward()

        if args.sign_trick:
            for reconstructed_data in reconstructed_data_per_epoch:
                reconstructed_data.grad.sign_()

        optimizer.step()

        # adjust the temperature
        T *= temperature_configs[args.temperature_mode][1]

    # if we used the sigmoid trick, we reapply it
    if args.sigmoid_trick:
        sigmoid_reconstruction = [continuous_sigmoid_bound(rd, dataset=dataset, T=T) for rd in reconstructed_data_per_epoch]
        reconstructed_data_per_epoch = sigmoid_reconstruction

    # after the optimization has finished for the given client, we project and match the data
    epoch_pooling = 'soft_avg+softmax' if args.softmax_trick or args.gumbel_softmax_trick else 'soft_avg'
    final_reconstruction = pooled_ensemble([reconstructed_data.clone().detach() for reconstructed_data in reconstructed_data_per_epoch],
                                           reconstructed_data_per_epoch[0].clone().detach(), dataset,
                                           pooling=epoch_pooling)

    # with the aggregated datapoint, we can finally run it again through the process to record its loss
    final_reconstruction_projected = dataset.project_batch(final_reconstruction, standardized=dataset.standardized)
    client_net = MetaMonkey(copy.deepcopy(original_net))
    criterion = torch.nn.CrossEntropyLoss()
    final_resulting_two_point_gradient, _ = simulate_local_training_for_attack(
        client_net=client_net,
        lr=args.lr,
        criterion=criterion,
        dataset=dataset,
        labels=ground_truth_labels,
        original_params=original_params,
        reconstructed_data_per_epoch=[final_reconstruction_projected for _ in range(args.n_local_epochs)],
        local_batch_size=args.local_batch_size,
        priors=None,
        softmax_trick=args.softmax_trick,
        gumbel_softmax_trick=args.gumbel_softmax_trick,
        apply_projection_to_features=apply_projection_to_features,
        temperature=T
    )
    final_loss = rec_loss_function[args.reconstruction_loss](final_resulting_two_point_gradient, true_two_point_gradient,
                                                             args.device)

    np.save(f'{args.metadata_path}/single_inversion_{args.client}.npy', final_reconstruction.detach().to('cpu').numpy())
    np.save(f'{args.metadata_path}/single_inversion_loss_{args.client}.npy', final_loss.detach().to('cpu').numpy())


if __name__ == '__main__':
    parser = argparse.ArgumentParser('fedavg_per_client')
    parser.add_argument('--random_seed', type=int, help='Set the random state for reproducibility')
    parser.add_argument('--client', type=int, help='The client index of this process')
    parser.add_argument('--metadata_path', type=str, help='Path pointing to the metadata')
    parser.add_argument('--lr', type=float, help='The training learning rate')
    parser.add_argument('--local_batch_size', type=int, help='Batch size for local FedAVG training')
    parser.add_argument('--n_local_epochs', type=int, help='Number of local epochs at client side')
    parser.add_argument('--attack_learning_rate', type=float, help='Learning rate of the attack')
    parser.add_argument('--attack_iterations', type=int, help='Number of iterations for the attack')
    parser.add_argument('--softmax_trick', action='store_true', help='Toggle if the softmax trick is used')
    parser.add_argument('--gumbel_softmax_trick', action='store_true', help='Toggle if the gumbel softmax trick is used')
    parser.add_argument('--sigmoid_trick', action='store_true', help='Toggle if the sigmoid trick is used')
    parser.add_argument('--sign_trick', action='store_true', help='Toggle to use the sign trick')
    parser.add_argument('--temperature_mode', type=str, help='Temperature mode of the annealing process')
    parser.add_argument('--initialization_mode', type=str, help='Initialization mode of each epoch sample')
    parser.add_argument('--reconstruction_loss', type=str, help='Name of the used reconstruction loss')
    parser.add_argument('--device', type=str, help='Name of the device on which the tensors are stored')
    in_args = parser.parse_args()
    main(in_args)
