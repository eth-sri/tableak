import sys
sys.path.append("..")
from utils import categorical_gumbel_softmax_sampling, categorical_softmax, get_acc_and_bac, continuous_sigmoid_bound, Timer
import torch
from .initializations import _uniform_initialization, _gaussian_initialization, _mean_initialization, \
    _dataset_sample_initialization, _likelihood_prior_sample_initialization, _mixed_initialization, \
    _best_sample_initialization
from .priors import _joint_gmm_prior, _mean_field_gmm_prior, _categorical_prior, _categorical_l2_prior, \
    _categorical_mean_field_jensen_shannon_prior, _continuous_uniform_prior, _theoretical_optimal_prior, \
    _theoretical_typicality_prior, _theoretical_marginal_prior, _theoretical_marginal_typicality_prior
from .inversion_losses import _weighted_CS_SE_loss, _gradient_norm_weighted_CS_SE_loss, _squared_error_loss, _cosine_similarity_loss
from .ensembling import pooled_ensemble
from collections import OrderedDict
from models import MetaMonkey
import numpy as np
import copy
import pickle
import os
import multiprocessing


def caller(x):
    return os.system(x)


def epoch_matching_prior_mean_square_error(epoch_data, device=None):
    """
    Permutation invariant prior that can be applied over the individual datapoints in the epochs. We first average up
    each dataset in the epoch and then calculate pairwise L2 distances between the epoch-data. It is normalized for
    number of features and number of epochs.

    :param epoch_data: (list of torch.tensor) List of the data-tensors used for each epoch.
    :param device: (str) Name of the device on which the tensors are stored. If None is given, the device on which the
        first of the epoch data is taken.
    :return: prior (torch.tensor) The calculated value of the prior with gradient information.

    """
    n_epochs = len(epoch_data)
    n_features = epoch_data[0].size()[-1]
    if device is None:
        device = epoch_data[0].device
    average_local_data = torch.stack([1/data.size()[0] * data.sum(dim=0) for data in epoch_data]).to(device)
    prior = torch.tensor([0.], device=device)
    for i in range(n_epochs):
        prior += 1/(n_epochs**2) * 1/n_features * (average_local_data - average_local_data[i]).pow(2).sum()
    return prior


def simulate_local_training_for_attack(client_net, lr, criterion, dataset, labels, original_params,
                                       reconstructed_data_per_epoch, local_batch_size, priors=None,
                                       epoch_matching_prior=None, softmax_trick=True, gumbel_softmax_trick=False,
                                       sigmoid_trick=False, temperature=None, apply_projection_to_features=None,
                                       device=None):
    """
    Simulates the local training such that it can be differentiated through with the Pytorch engine.

    :param client_net: (MetaMonkey) A MetaMonkey wrapped nn.Module neural network that supports parameter assignment$
        directly through assigning and OrderedDict.
    :param lr: (float) The learning rate of the local training.
    :param criterion: (nn.Module) The loss function of the training.
    :param dataset: (datasets.BaseDataset) The dataset with which we work. It contains usually the data necessary for
        the calculation of the prior.
    :param labels: (torch.tensor) The labels for a whole local epoch, ordered as the batches should be.
    :param original_params: (OrderedDict) The original parameter dictionary of the network before training.
    :param reconstructed_data_per_epoch: (list of torch.tensor) List of the concatenated batches of data used for
        training. This is what we optimize for.
    :param local_batch_size: (int) The batch size of the local training.
    :param priors: (list of tuple(float, str)) The regularization parameter(s) plus the name(s) of the prior(s) we wish
        to use. Default None accounts to no prior.
    :param epoch_matching_prior: tuple(float, str) The regularization parameter of the epoch matching prior plus its
        name. If None is given (default), then no epoch matching prior will be applied.
    :param softmax_trick: (bool) Toggle to apply the softmax trick to the categorical features. Effectively, it serves
        as a structural prior on the features.
    :param gumbel_softmax_trick: (bool) Toggle to apply the gumbel-softmax trick to the categorical features.
    :param sigmoid_trick: (bool) Apply the sigmoid trick to the continuous features to enforce the bounds.
    :param apply_projection_to_features: (list) If given, both the softmax trick and the gumbel softmax trick will be
        applied only to the set of features given in this list.
    :param temperature: (float) Temperature parameter for the softmax in the categorical prior.
    :param device: (str) Name of the device on which the tensors are stored.
    :return: resulting_two_point_gradient: (list of torch.tensor) Two-point gradient estimate over a local training.
    """
    if device is None:
        device = dataset.device

    if apply_projection_to_features is None:
        apply_projection_to_features = 'all'

    available_priors = {
        'categorical_prior': _categorical_prior,
        'cont_uniform': _continuous_uniform_prior,
        'cont_joint_gmm': _joint_gmm_prior,
        'cont_mean_field_gmm': _mean_field_gmm_prior,
        'cat_mean_field_JS': _categorical_mean_field_jensen_shannon_prior,
        'cat_l2': _categorical_l2_prior,
        'theoretical_optimal': _theoretical_optimal_prior,
        'theoretical_typicality': _theoretical_typicality_prior,
        'theoretical_marginal': _theoretical_marginal_prior,
        'theoretical_marginal_typicality': _theoretical_marginal_typicality_prior
    }

    available_epoch_matching_priors = {
        'mean_squared_error': epoch_matching_prior_mean_square_error
    }

    if priors is not None:
        # will raise a key error of we chose a non-implemented prior
        prior_params = [prior_params[0] for prior_params in priors]
        prior_loss_functions = [available_priors[prior_params[1]] for prior_params in priors]
    else:
        prior_loss_functions = None
        prior_params = None

    regularizer = torch.as_tensor([0.0], device=device)

    n_data_lines = labels.size()[0]
    for local_epoch, reconstructed_data in enumerate(reconstructed_data_per_epoch):

        n_batches = int(np.ceil(n_data_lines / local_batch_size))
        for b in range(n_batches):
            current_batch_X = reconstructed_data[b*local_batch_size:min(n_data_lines, (b+1)*local_batch_size)]
            current_batch_y = labels[b*local_batch_size:min(n_data_lines, (b+1)*local_batch_size)].clone().detach()

            # apply softmax or gumbel-softmax
            if gumbel_softmax_trick:
                x_rec = categorical_gumbel_softmax_sampling(current_batch_X, tau=temperature, dataset=dataset)
                categoricals_projected = True
            elif softmax_trick:
                x_rec = categorical_softmax(current_batch_X, tau=temperature, dataset=dataset,
                                            apply_to=apply_projection_to_features)
                categoricals_projected = True
            else:
                x_rec = current_batch_X * 1.
                categoricals_projected = False

            if sigmoid_trick:
                x_rec = continuous_sigmoid_bound(x_rec, dataset=dataset, T=temperature)

            outputs = client_net(x_rec, client_net.parameters)
            training_loss = criterion(outputs, current_batch_y)
            grad = torch.autograd.grad(training_loss, client_net.parameters.values(), retain_graph=True,
                                       create_graph=True, only_inputs=True, allow_unused=True)

            client_net.parameters = OrderedDict((name, param - lr * param_grad) for ((name, param), param_grad) in zip(client_net.parameters.items(), grad))

            # keep track of a regularizer if needed
            if priors is not None:
                for prior_param, prior_function in zip(prior_params, prior_loss_functions):
                    regularizer += 1/(n_batches*local_epoch) * prior_param * prior_function(x_reconstruct=x_rec,
                                                                                            dataset=dataset,
                                                                                            softmax_trick=categoricals_projected,
                                                                                            labels=current_batch_y,
                                                                                            T=temperature)

    # if we have an epoch matching prior, we calculate its value, for this, we have to reapply any projections made on
    # the data previously
    if epoch_matching_prior is not None:
        epoch_matching_prior_param = epoch_matching_prior[0]
        epoch_matching_prior_function = available_epoch_matching_priors[epoch_matching_prior[1]]

        # reapply the projections if any
        if softmax_trick or gumbel_softmax_trick:
            projected_epoch_data = [categorical_softmax(epoch_data, dataset=dataset, tau=temperature,
                                                        apply_to=apply_projection_to_features) for epoch_data in reconstructed_data_per_epoch]
        else:
            projected_epoch_data = reconstructed_data_per_epoch
        # reapply the sigmoid if given
        if sigmoid_trick:
            projected_bounded_epoch_data = [continuous_sigmoid_bound(pd, dataset=dataset, T=temperature) for pd in projected_epoch_data]
        else:
            projected_bounded_epoch_data = projected_epoch_data
        regularizer += epoch_matching_prior_param * epoch_matching_prior_function(projected_bounded_epoch_data, device=device)

    # end of training, time to extract the parameters
    resulting_parameters = list(client_net.parameters.values())
    resulting_two_point_gradient = [original_param - param for original_param, param in
                                    zip(original_params, resulting_parameters)]

    return resulting_two_point_gradient, regularizer


def fed_avg_attack(original_net, attacked_clients_params, n_local_epochs, local_batch_size, lr,
                   dataset, per_client_ground_truth_data, per_client_ground_truth_labels, attack_iterations=1000,
                   attack_learning_rate=0.06, reconstruction_loss='cosine_sim', priors=None, epoch_matching_prior=None,
                   initialization_mode='uniform', softmax_trick=True, gumbel_softmax_trick=False, temperature_mode=None,
                   sigmoid_trick=False, sign_trick=True, apply_projection_to_features=None, device=None):
    """
    FedAVG attack following Dimitrov et al. 2022.
    """
    if device is None:
        device = dataset.device

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

    if reconstruction_loss not in list(rec_loss_function.keys()):
        raise NotImplementedError(
            f'The desired loss function is not implemented, available loss function are: {list(rec_loss_function.keys())}')

    final_reconstructions_per_client = []
    final_loss_per_client = []

    # we will go by attacked client and then completely restart every time
    for attacked_client, (attacked_client_params, ground_truth_data, ground_truth_labels) in enumerate(zip(attacked_clients_params, per_client_ground_truth_data, per_client_ground_truth_labels)):
        # fix the client network and extract its starting parameters
        original_params = [param.detach().clone() for param in original_net.parameters()]
        true_two_point_gradient = [(original_param - new_param).detach().clone() for original_param, new_param in zip(original_params, attacked_client_params)]

        # we reconstruct independently in each epoch and aggregate in the end, as per Dimitrov et al.
        # initialize the data
        reconstructed_data_per_epoch = [initialization[initialization_mode](ground_truth_data, dataset, device) for _ in range(n_local_epochs)]
        for reconstructed_data in reconstructed_data_per_epoch:
            reconstructed_data.requires_grad = True

        optimizer = torch.optim.Adam(reconstructed_data_per_epoch, lr=attack_learning_rate)

        T = temperature_configs[temperature_mode][0]

        for it in range(attack_iterations):

            optimizer.zero_grad()
            original_net.zero_grad()
            client_net = MetaMonkey(copy.deepcopy(original_net))
            criterion = torch.nn.CrossEntropyLoss()

            resulting_two_point_gradient, regularizer = simulate_local_training_for_attack(
                client_net=client_net,
                lr=lr,
                criterion=criterion,
                dataset=dataset,
                labels=ground_truth_labels,
                original_params=original_params,
                reconstructed_data_per_epoch=reconstructed_data_per_epoch,
                local_batch_size=local_batch_size,
                priors=priors,
                epoch_matching_prior=epoch_matching_prior,
                softmax_trick=softmax_trick,
                gumbel_softmax_trick=gumbel_softmax_trick,
                sigmoid_trick=sigmoid_trick,
                apply_projection_to_features=apply_projection_to_features,
                temperature=T
            )

            # calculate the final objective
            loss = rec_loss_function[reconstruction_loss](resulting_two_point_gradient, true_two_point_gradient, device)
            loss += regularizer
            loss.backward()

            if sign_trick:
                for reconstructed_data in reconstructed_data_per_epoch:
                    reconstructed_data.grad.sign_()

            optimizer.step()

            # adjust the temperature
            T *= temperature_configs[temperature_mode][1]

        # if we used the sigmoid trick, we reapply it
        if sigmoid_trick:
            sigmoid_reconstruction = [continuous_sigmoid_bound(rd, dataset=dataset, T=T) for rd in reconstructed_data_per_epoch]
            reconstructed_data_per_epoch = sigmoid_reconstruction

        # after the optimization has finished for the given client, we project and match the data
        epoch_pooling = 'soft_avg+softmax' if softmax_trick or gumbel_softmax_trick else 'soft_avg'
        final_reconstruction = pooled_ensemble([reconstructed_data.clone().detach() for reconstructed_data in reconstructed_data_per_epoch],
                                               reconstructed_data_per_epoch[0].clone().detach(), dataset,
                                               pooling=epoch_pooling)
        final_reconstructions_per_client.append(final_reconstruction)

        # with the aggregated datapoint, we can finally run it again through the process to record its loss
        final_reconstruction_projected = dataset.project_batch(final_reconstruction, standardized=dataset.standardized)
        client_net = MetaMonkey(copy.deepcopy(original_net))
        criterion = torch.nn.CrossEntropyLoss()
        final_resulting_two_point_gradient, _ = simulate_local_training_for_attack(
                client_net=client_net,
                lr=lr,
                criterion=criterion,
                dataset=dataset,
                labels=ground_truth_labels,
                original_params=original_params,
                reconstructed_data_per_epoch=[final_reconstruction_projected for _ in range(n_local_epochs)],
                local_batch_size=local_batch_size,
                priors=None,
                softmax_trick=softmax_trick,
                gumbel_softmax_trick=gumbel_softmax_trick,
                apply_projection_to_features=apply_projection_to_features,
                temperature=T
        )
        final_loss = rec_loss_function[reconstruction_loss](final_resulting_two_point_gradient, true_two_point_gradient, device)
        final_loss_per_client.append(final_loss.detach().item())

    return final_reconstructions_per_client, final_loss_per_client


def fed_avg_attack_parallelized_over_clients(original_net, attacked_clients_params, n_local_epochs, local_batch_size,
                                             lr, dataset, per_client_ground_truth_data, per_client_ground_truth_labels,
                                             metadata_path='metadata', attack_iterations=1000, attack_learning_rate=0.06,
                                             reconstruction_loss='cosine_sim', priors=None, epoch_matching_prior=None,
                                             initialization_mode='uniform', softmax_trick=True, sigmoid_trick=False,
                                             gumbel_softmax_trick=False, temperature_mode=None, sign_trick=True,
                                             apply_projection_to_features=None, max_n_cpus=50, first_cpu=0, device=None):
    """
    FedAVG attack following Dimitrov et al. 2022.
    """
    if device is None:
        device = dataset.device

    n_clients = len(per_client_ground_truth_data)

    # create the working directory
    metadata_path += f'{np.random.randint(0, 1000, 1).item()}'  # disturb by a random integer to avoid conflicts
    os.makedirs(metadata_path, exist_ok=True)

    # save everything that is needed for the per client parallelization
    with open(f'{metadata_path}/original_net.pickle', 'wb') as f:
        pickle.dump(original_net, f)
    with open(f'{metadata_path}/attacked_clients_params.pickle', 'wb') as f:
        pickle.dump(attacked_clients_params, f)
    with open(f'{metadata_path}/per_client_ground_truth_data.pickle', 'wb') as f:
        pickle.dump(per_client_ground_truth_data, f)
    with open(f'{metadata_path}/per_client_ground_truth_labels.pickle', 'wb') as f:
        pickle.dump(per_client_ground_truth_labels, f)
    with open(f'{metadata_path}/dataset.pickle', 'wb') as f:
        pickle.dump(dataset, f)
    with open(f'{metadata_path}/apply_projection_to_features.pickle', 'wb') as f:
        pickle.dump(apply_projection_to_features, f)
    # prior could also be None
    if priors is not None:
        with open(f'{metadata_path}/priors.pickle', 'wb') as f:
            pickle.dump(priors, f)
    # epoch matching prior could also be None
    if epoch_matching_prior is not None:
        with open(f'{metadata_path}/epoch_matching_prior.pickle', 'wb') as f:
            pickle.dump(epoch_matching_prior, f)

    # call all scripts to complete the individual inversions
    random_seeds = np.random.randint(0, 15000, size=n_clients)
    split_seeds = np.array_split(random_seeds, int(np.ceil(n_clients/max_n_cpus)))
    split_client_ranges = np.array_split(np.arange(n_clients), int(np.ceil(n_clients/max_n_cpus)))
    for split_seed, split_client_range in zip(split_seeds, split_client_ranges):
        process_pool = multiprocessing.Pool(processes=n_clients)
        all_processes_to_execute = []
        for idx, (client, rs) in enumerate(zip(split_client_range, split_seed)):
            command = f'taskset -c {first_cpu + idx} python single_fedavg_inversion_for_client.py --random_seed {rs} ' \
                    f'--client {client} --metadata_path {metadata_path} --lr {lr} --local_batch_size {local_batch_size} ' \
                    f'--n_local_epochs {n_local_epochs} --attack_learning_rate {attack_learning_rate} ' \
                    f'--attack_iterations {attack_iterations} --temperature_mode {temperature_mode} ' \
                    f'--initialization_mode {initialization_mode} --reconstruction_loss {reconstruction_loss} ' \
                    f'--device {device}'
            if softmax_trick:
                command += ' --softmax_trick'
            if gumbel_softmax_trick:
                command += ' --gumbel_softmax_trick'
            if sign_trick:
                command += ' --sign_trick'
            if sigmoid_trick:
                command += ' --sigmoid_trick'
            all_processes_to_execute.append(command)
        process_pool.map(caller, tuple(all_processes_to_execute))

    # after all inversions have been executed, load the data and organize it in the desired format
    final_reconstructions_per_client, final_loss_per_client = [], []
    for client in range(n_clients):
        final_reconstructions_per_client.append(torch.tensor(np.load(f'{metadata_path}/single_inversion_{client}.npy')).to(device))
        final_loss_per_client.append(np.load(f'{metadata_path}/single_inversion_loss_{client}.npy').item())

    # delete the metadata
    os.system(f'rm -rf {metadata_path}')

    return final_reconstructions_per_client, final_loss_per_client


def train_and_attack_fed_avg(net, n_clients, n_global_epochs, n_local_epochs, local_batch_size, lr, dataset, shuffle=False,
                             attacked_clients=None, attack_iterations=1000, reconstruction_loss='cosine_sim', priors=None,
                             epoch_matching_prior=None, post_selection=1, attack_learning_rate=0.06, return_all=False,
                             pooling=None, perfect_pooling=False, initialization_mode='uniform', softmax_trick=True,
                             gumbel_softmax_trick=False, sigmoid_trick=False, temperature_mode='constant',
                             sign_trick=True, fish_for_features=None, device=None, verbose=False, max_n_cpus=50, first_cpu=0,
                             max_client_dataset_size=None, parallelized=False, metadata_path='metadata'):
    """
    Train a network using federated averaging while also attacking a subset of the clients at each global epoch,
    simulating a real world data fishing scenario.

    :param net: (torch.nn.module) The subjected network.
    :param n_clients: (int) The number of participating clients.
    :param n_global_epochs: (int) The number of global epochs conducted by the clients to train net.
    :param n_local_epochs: (int) The number of local epochs made by a client.
    :param local_batch_size: (int) The batch size used by the clients.
    :param lr: (float) The training learning rate used for the local updates of the clients.
    :param dataset: (BaseDataset) The instantiated dataset over which we conduct our attack and training.
    :param shuffle: (bool) Toggle to shuffle the dataset before training.
    :param attacked_clients: (list) List of the client indices we intend to attack.
    :param attack_iterations: (int) The number of iterations we conduct during the optimization in our attack.
    :param reconstruction_loss: (str) The name of the inversion loss function to be used.
    :param priors: (list of tuple(float, str)) The regularization parameter(s) plus the name(s) of the prior(s) we wish
        to use. Default None accounts to no prior.
    :param epoch_matching_prior: tuple(float, str) The regularization parameter of the epoch matching prior plus its
        name. If None is given (default), then no epoch matching prior will be applied.
    :param post_selection: (int) The best reconstruction based on the reconstruction loss will be returned from
        'post_selection' number of randomly reinitialized trials.
    :param attack_learning_rate: (float) The learning rate/step size of the data leakage attack optimization.
    :param return_all
    :param pooling: (None or str) If this argument is given (i.e. not None) the 'post_selection' number of runs are
        collected and pooled according to the pooling mode given. Available are
        - 'soft_avg': Simply average up all samples.
        - 'hard_avg': Project all features and average afterwards.
        - 'median': Take the median of all features.
        - 'soft_avg+softmax': Take first the softmax over each samples' categorical features and then do the soft
            averaging over samples.
        - 'median+softmax': Take first the softmax over each samples' categorical features and then take the median over
            samples.
    :param perfect_pooling: (bool) Choose the true datapoint as the one to match towards for the pooling operation.
    :param initialization_mode:
    :param softmax_trick: (bool) Toggle to apply the softmax trick to the categorical features. Effectively, it serves
        as a structural prior on the features.
    :param gumbel_softmax_trick: (bool) Apply the gumbel-softmax trick to optimizing the categorical features.
    :param sigmoid_trick: (bool) Apply the sigmoid trick to the continuous features to enforce the bounds.
    :param temperature_mode: (str) Any time we have to apply a softmax to approximate the argmax in the categorical
        features, we use a softmax with a temperature. If we choose to cool this softmax, then we start at a high
        temperature in the optimization and as the optimization progresses we cool the softmax in order that it is more
        concentrated on the maximum. When we choose heating, the opposite process occurs. Accepted modes are: ['cool',
        'constant', 'heat'].
    :param sign_trick: (bool) Toggle to use the optimization trick, where we take the sign of the gradient for a
        datapoint to update it (FGSM-like updates).
    :param fish_for_features: (list) If this argument is given, the optimization is restricted only to these features.
    :param device: The device on which the tensors are located and the calculation should take place. Note that pytorch
        will throw an error if some tensors are not on the same device.
    :param verbose: (bool) Toggle to print the training progress.
    :param max_n_cpus: (int) The number of cpus available for parallelization.
    :param first_cpu: (int) The first cpu in the pool.
    :param max_client_dataset_size: (int) The maximum amount of data that can be owned by a given client.
    :param parallelized: (bool) Toggle if you wish to parallelize over clients.
    :param metadata_path: (str) If the process is parallelized over clients, the metadata during the process will be
        saved here.
    """
    if device is None:
        device = dataset.device

    if attacked_clients is None:
        attacked_clients = []
    elif attacked_clients == 'all':
        attacked_clients = list(np.arange(n_clients))

    if max_client_dataset_size is None:
        max_client_dataset_size = len(dataset)

    # attack data and training statistics
    per_global_epoch_per_client_reconstructions = []
    per_global_epoch_per_client_ground_truth = []
    training_data = np.zeros((n_global_epochs, 2))

    # get the data and then split it into client datasets
    if shuffle:
        dataset.shuffle()
    Xtrain, ytrain = dataset.get_Xtrain(), dataset.get_ytrain()
    split_size = min(max_client_dataset_size, int(np.ceil(Xtrain.size()[0] / n_clients)))  # ceiling
    Xtrain_splits = [Xtrain[i*split_size:min(int(Xtrain.size()[0]), (i+1)*split_size)].clone().detach() for i in range(n_clients)]
    ytrain_splits = [ytrain[i*split_size:min(int(Xtrain.size()[0]), (i+1)*split_size)].clone().detach() for i in range(n_clients)]

    # instantiate the loss
    criterion = torch.nn.CrossEntropyLoss()

    timer = Timer(n_global_epochs)

    # training loop
    for global_epoch in range(n_global_epochs):
        timer.start()

        acc, bac = get_acc_and_bac(net, dataset.get_Xtest(), dataset.get_ytest())
        if verbose:
            print(f'Global Epochs: {global_epoch + 1}/{n_global_epochs}    Acc: {acc * 100:.2f}%    BAcc: {bac * 100:.2f}%    {timer}', end='\r')
        training_data[global_epoch] = acc, bac

        # create the current client net copies
        client_nets = [copy.deepcopy(net) for _ in range(n_clients)]

        # iterate through each client (this should be done in parallel in theory)
        for client, (client_X, client_y, client_net) in enumerate(zip(Xtrain_splits, ytrain_splits, client_nets)):

            # do the local training for each client
            n_batches = int(np.ceil(client_X.size()[0] / local_batch_size))
            for local_epoch in range(n_local_epochs):

                # complete an epoch
                for b in range(n_batches):
                    current_batch_X = client_X[b * local_batch_size:min(int(client_X.size()[0]), (b+1)*local_batch_size)].clone().detach()
                    current_batch_y = client_y[b * local_batch_size:min(int(client_X.size()[0]), (b+1)*local_batch_size)].clone().detach()

                    outputs = client_net(current_batch_X)
                    loss = criterion(outputs, current_batch_y)
                    grad = torch.autograd.grad(loss, client_net.parameters(), retain_graph=True)

                    with torch.no_grad():
                        for param, param_grad in zip(client_net.parameters(), grad):
                            param -= lr * param_grad

        # extract the parameters from the client nets
        clients_params = [[param.clone().detach() for param in client_net.parameters()] for client_net in client_nets]

        # -------------- ATTACK -------------- #
        per_client_all_reconstructions = [[] for _ in range(len(attacked_clients))]
        per_client_best_reconstructions = [None for _ in range(len(attacked_clients))]
        per_client_best_scores = [None for _ in range(len(attacked_clients))]
        per_client_ground_truth_data = [Xtrain_splits[attacked_client].detach().clone() for attacked_client in attacked_clients]
        per_client_ground_truth_labels = [ytrain_splits[attacked_client].detach().clone() for attacked_client in attacked_clients]
        attacked_clients_params = [[param.clone().detach() for param in clients_params[attacked_client]] for attacked_client in attacked_clients]

        for _ in range(post_selection):

            if parallelized:
                per_client_candidate_reconstructions, per_client_final_losses = fed_avg_attack_parallelized_over_clients(
                    original_net=copy.deepcopy(net),
                    attacked_clients_params=attacked_clients_params,
                    attack_iterations=attack_iterations,
                    attack_learning_rate=attack_learning_rate,
                    n_local_epochs=n_local_epochs,
                    local_batch_size=local_batch_size,
                    lr=lr,
                    dataset=dataset,
                    per_client_ground_truth_data=per_client_ground_truth_data,
                    per_client_ground_truth_labels=per_client_ground_truth_labels,
                    reconstruction_loss=reconstruction_loss,
                    priors=priors,
                    epoch_matching_prior=epoch_matching_prior,
                    initialization_mode=initialization_mode,
                    softmax_trick=softmax_trick,
                    gumbel_softmax_trick=gumbel_softmax_trick,
                    sigmoid_trick=sigmoid_trick,
                    temperature_mode=temperature_mode,
                    sign_trick=sign_trick,
                    apply_projection_to_features=fish_for_features,
                    max_n_cpus=max_n_cpus,
                    first_cpu=first_cpu,
                    device=device,
                    metadata_path=metadata_path
                )
            else:
                per_client_candidate_reconstructions, per_client_final_losses = fed_avg_attack(
                    original_net=copy.deepcopy(net),
                    attacked_clients_params=attacked_clients_params,
                    attack_iterations=attack_iterations,
                    attack_learning_rate=attack_learning_rate,
                    n_local_epochs=n_local_epochs,
                    local_batch_size=local_batch_size,
                    lr=lr,
                    dataset=dataset,
                    per_client_ground_truth_data=per_client_ground_truth_data,
                    per_client_ground_truth_labels=per_client_ground_truth_labels,
                    reconstruction_loss=reconstruction_loss,
                    priors=priors,
                    epoch_matching_prior=epoch_matching_prior,
                    initialization_mode=initialization_mode,
                    softmax_trick=softmax_trick,
                    gumbel_softmax_trick=gumbel_softmax_trick,
                    sigmoid_trick=sigmoid_trick,
                    temperature_mode=temperature_mode,
                    sign_trick=sign_trick,
                    apply_projection_to_features=fish_for_features,
                    device=device
                )

            # enter the results in the collectors
            for client_idx in range(len(attacked_clients)):
                per_client_all_reconstructions[client_idx].append(per_client_candidate_reconstructions[client_idx].detach().clone())
                if (per_client_best_scores[client_idx] is None) or (per_client_best_scores[client_idx] > per_client_final_losses[client_idx]):
                    per_client_best_scores[client_idx] = per_client_final_losses[client_idx]
                    per_client_best_reconstructions[client_idx] = per_client_candidate_reconstructions[client_idx].detach().clone()

        if return_all:
            per_global_epoch_per_client_reconstructions.append(per_client_all_reconstructions)
        elif pooling is not None:
            if perfect_pooling:
                per_client_pooled = [pooled_ensemble(all_reconstructions, ground_truth_data, dataset, pooling=pooling)
                                     for all_reconstructions, ground_truth_data in zip(per_client_all_reconstructions, per_client_ground_truth_data)]
            else:
                per_client_pooled = [pooled_ensemble(all_reconstructions, best_reconstruction, dataset, pooling=pooling)
                                     for all_reconstructions, best_reconstruction in zip(per_client_all_reconstructions, per_client_best_reconstructions)]
            per_global_epoch_per_client_reconstructions.append(per_client_pooled)
        else:
            per_global_epoch_per_client_reconstructions.append(per_client_best_reconstructions)
        per_global_epoch_per_client_ground_truth.append(per_client_ground_truth_data)
        # -------------- ATTACK END -------------- #

        # Continue the training
        # transpose the list
        transposed_clients_params = [[] for _ in range(len(clients_params[0]))]
        for client_params in clients_params:
            for i, param in enumerate(client_params):
                transposed_clients_params[i].append(param.clone().detach())

        # aggregate the params using mean aggregation
        aggregated_params = [torch.mean(torch.stack(params_over_clients), dim=0) for params_over_clients in transposed_clients_params]

        # write the new parameters into the main network
        with torch.no_grad():
            for param, agg_param in zip(net.parameters(), aggregated_params):
                param.copy_(agg_param)
        
        timer.end()
    timer.duration()

    return net, training_data, per_global_epoch_per_client_reconstructions, per_global_epoch_per_client_ground_truth
