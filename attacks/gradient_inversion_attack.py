import sys
sys.path.append("..")
from utils import categorical_gumbel_softmax_sampling, categorical_softmax, create_feature_mask, \
    continuous_sigmoid_bound
import torch
from .initializations import _uniform_initialization, _gaussian_initialization, _mean_initialization, \
    _dataset_sample_initialization, _likelihood_prior_sample_initialization, _mixed_initialization, \
    _best_sample_initialization
from .priors import _joint_gmm_prior, _mean_field_gmm_prior, _categorical_prior, _categorical_l2_prior, \
    _categorical_mean_field_jensen_shannon_prior, _continuous_uniform_prior, _theoretical_optimal_prior, \
    _theoretical_typicality_prior, _theoretical_marginal_prior, _theoretical_marginal_typicality_prior, \
    _categorical_l1_prior, _all_line_l2_prior, _batch_norm_prior
from .inversion_losses import _weighted_CS_SE_loss, _gradient_norm_weighted_CS_SE_loss, _squared_error_loss, _cosine_similarity_loss
from .ensembling import pooled_ensemble
import inspect
import numpy as np


def closure(optimizer, net, training_criterion, reconstruction_loss, x_reconstruct, true_grad, true_label, alpha=1.,
            priors=None, prior_params=None, dataset=None, temperature=None, mask=None, sign_trick=True, weight_trick=True,
            gumbel_softmax_trick=False, softmax_trick=True, apply_projection_to_features=None, sigmoid_trick=False,
            true_bn_stats=None, verbose=False, current_it=None, soteria_defended_layer=None, device='cpu'):
    """
    Provides the full objective for the optimizer.

    :param optimizer: An instantiated pytorch optimizer.
    :param net: (nn.Module) The neural network subject to the gradient inversion attack.
    :param training_criterion: (nn.Module) The training loss function of the neural network with respect to which the
        'received' true gradient has been calculated.
    :param reconstruction_loss: (callable) The loss function of the gradient inversion.
    :param x_reconstruct: (torch.Tensor) The reconstructed datapoint/batch in its current state.
    :param true_grad: (list of torch.Tensor) The received gradient.
    :param true_label: (torch.Tensor) The true label of the datapoint/batch we wish to reconstruct
        (simplifying assumption).
    :param alpha: (float) A weighting parameter for combined losses.
    :param priors: (list of callables) The prior(s) we wish to use. Default None accounts to no prior.
    :param prior_params: (list of floats) The prior regularization parameters ordered as 'priors'.
    :param dataset: (datasets.BaseDataset) The dataset with which we work. It contains usually the data necessary for
        the calculation of the prior. The argument can be ignored if no prior is given.
    :param temperature: (float) Temperature parameter for the softmax in the categorical prior.
    :param mask: (torch.tensor) If given, this mask is applied to the gradient. With the help of it we can restrict the
        optimization to a subset of the parameters.
    :param sign_trick: (bool) Toggle to use the sign trick or not (FGSM-like optimization).
    :param weight_trick: (bool) Toggle to use the weight trick introduced by Geiping et al. The idea behind the trick is
        that by giving more weight to the gradients closer to the input, we help the optimization process to first get
        a good enough grip on what the actual data might be, and afterwards fine-tune.
    :param gumbel_softmax_trick: (bool) Toggle to apply the gumbel-softmax trick to the categorical features.
    :param softmax_trick: (bool) Toggle to apply the softmax trick to the categorical features. Effectively, it serves
        as a structural prior on the features.
    :param apply_projection_to_features: (list) If given, both the softmax trick and the gumbel softmax trick will be
        applied only to the set of features given in this list.
    :param sigmoid_trick: (bool) Apply the sigmoid trick to the continuous features to enforce the bounds.
    :param true_bn_stats: (list of tuples) Optional, only if the bathcnorm prior is used. These are the true batch norm
        parameters from the client.
    :param verbose: (bool) Set to True to display the progress of the current iteration.
    :param current_it: (int) The current iteration number, only used if 'verbose' is True.
    :param soteria_defended_layer: (int) The index of the layer that is defended by SOTERIA.
    :param device: The device on which the tensors are located and the calculation should take place. Note that pytorch
        will throw an error if some tensors are not on the same device.
    :return: (callable) The reconstruction objective ready to be delivered to the optimizer.
    """
    if apply_projection_to_features is None:
        apply_projection_to_features = 'all'

    def full_rec_loss():

        optimizer.zero_grad()
        net.zero_grad()

        if gumbel_softmax_trick:
            x_rec = categorical_gumbel_softmax_sampling(x_reconstruct, tau=temperature, dataset=dataset)
            categoricals_projected = True
        elif softmax_trick:
            x_rec = categorical_softmax(x_reconstruct, tau=temperature, dataset=dataset, apply_to=apply_projection_to_features)
            categoricals_projected = True
        else:
            x_rec = x_reconstruct * 1.
            categoricals_projected = False
        # if we have a mask, we want to keep the masked features undisturbed of any projections
        if mask is not None:
            x_rec = (1-mask) * x_reconstruct + mask * x_rec

        if sigmoid_trick:
            x_rec = continuous_sigmoid_bound(x_rec, dataset=dataset)

        if true_bn_stats is not None:
            output, measured_bn_stats = net(x_rec, return_bn_stats=True)
        else:
            output = net(x_rec)
        loss = training_criterion(output, true_label)
        reconstruct_gradient = torch.autograd.grad(loss, net.parameters(), create_graph=True)
        regularizer = torch.as_tensor([0.0], device=device)

        # optimization trick
        layer_weights = torch.arange(len(true_grad), 0, -1, dtype=x_rec.dtype, device=device)
        layer_weights = layer_weights.softmax(dim=0)
        layer_weights = layer_weights / layer_weights[0]  # Apply layer weights to reconstruction error

        layer_weights = layer_weights if weight_trick else None

        if soteria_defended_layer is not None and soteria_defended_layer < len(true_grad) / 2:
            # eliminate the defended layer
            defended_indices = np.reshape(np.arange(len(true_grad)), (-1, 2))[soteria_defended_layer]
            true_gradient = [grad for k, grad in enumerate(true_grad) if k not in defended_indices]
            rec_gradient = [grad for k, grad in enumerate(reconstruct_gradient) if k not in defended_indices]
        else:
            true_gradient = true_grad
            rec_gradient = reconstruct_gradient
        
        rec_loss = reconstruction_loss(reconstruct_gradient=rec_gradient,
                                       true_grad=true_gradient, device=device, weights=layer_weights, alpha=alpha)

        # calculate the regularizer if given
        if priors is not None:
            for prior_param, prior_function in zip(prior_params, priors):
                if 'true_bn_stats' in inspect.signature(prior_function).parameters:
                    regularizer += prior_param * prior_function(x_reconstruct=x_rec, dataset=dataset,
                                                                true_bn_stats=true_bn_stats, measured_bn_stats=measured_bn_stats,
                                                                softmax_trick=categoricals_projected, labels=true_label,
                                                                T=temperature)
                else:                                                
                    regularizer += prior_param * prior_function(x_reconstruct=x_rec, dataset=dataset,
                                                                softmax_trick=categoricals_projected, labels=true_label,
                                                                T=temperature)

        # add the regularizer
        comb_loss = rec_loss + regularizer

        comb_loss.backward()

        # Optimization trick
        if sign_trick:
            x_reconstruct.grad.sign_()

        # If we only want to update parts of the weights we mask away the rest of the gradient
        if mask is not None:
            x_reconstruct.grad *= mask

        if verbose:
            print('It:', current_it, 'Reconstr:', rec_loss.item(), 'Regularizer:', regularizer.item())

        return rec_loss

    return full_rec_loss


def naive_invert_grad(net, training_criterion, true_grad, true_label, true_data, reconstruction_loss='squared_error',
                      initialization_mode='uniform', initialization_sample=None, learning_rates=None, alpha=1.,
                      priors=None, dataset=None, max_iterations=None, two_staged=False, sign_trick=True, mask=None,
                      weight_trick=True, gumbel_softmax_trick=False, softmax_trick=True, sigmoid_trick=False,
                      apply_projection_to_features=None, temperature_mode='cool', true_bn_stats=None, 
                      lr_scheduler=False, soteria_defended_layer=None, verbose=False, device='cpu'):
    """
    Performs the gradient inversion and return the guessed datapoint/batch. For now, it assumes the knowledge of the
    true labels.

    :param net: (nn.Module) The neural network subject to the gradient inversion attack.
    :param training_criterion: (nn.Module) The training loss function of the neural network with respect to which the
        'received' true gradient has been calculated.
    :param true_grad: (list of torch.Tensor) The received gradient.
    :param true_label: (torch.Tensor) The true label of the datapoint/batch we wish to reconstruct
        (simplifying assumption).
    :param true_data: (torch.Tensor) The original/true data that produced the received gradient. This argument is not
        used in the inversion process, merely, it facilitates to display the progress of the inversion process.
    :param reconstruction_loss: (str) The name of the inversion loss function to be used.
    :param initialization_mode: (str) The initialization scheme to be employed.
    :param initialization_sample: (torch.tensor) The sample from which we start the inversion optimization if the
        initialization mode 'from_sample' is selected. The sample should be of the same size as the true data.
    :param learning_rates: (float or tuple of floats) If 'two_staged' is False, we require a single learning rate for
        the optimizer. If we optimize in two stages, we require two learning rates, one for each stage respectively,
        packed in a tuple.
    :param alpha: (float) A weight parameter for combined loss functions.
    :param priors: (list of tuple(float, str)) The regularization parameter(s) plus the name(s) of the prior(s) we wish
        to use. Default None accounts to no prior.
    :param dataset: (datasets.BaseDataset) The dataset with which we work. It contains usually the data necessary for
        the calculation of the prior. The argument can be ignored if no prior is given.
    :param max_iterations: (int or tuple) Maximum number of iterations to be performed by the optimizer to recover the
        data.
    :param two_staged: (bool) If true, after the joint reconstruction has converged, we freeze the categorical features
        and fine tune only the continuous features.
    :param sign_trick: (bool) Toggle to use the optimization trick, where we take the sign of the gradient for a
        datapoint to update it (FGSM-like updates).
    :param mask: (torch.tensor) A binary mask to restrict the optimization to just a subset of the features.
    :param weight_trick: (bool) Toggle to use the weight trick introduced by Geiping et al. The idea behind the trick is
        that by giving more weight to the gradients closer to the input, we help the optimization process to first get
        a good enough grip on what the actual data might be, and afterwards fine-tune.
    :param gumbel_softmax_trick: (bool) Toggle to apply the gumbel-softmax trick to the categorical features.
    :param softmax_trick: (bool) Toggle to apply the softmax trick to the categorical features. Effectively, it serves
        as a structural prior on the features.
    :param sigmoid_trick: (bool) Apply the sigmoid trick to the continuous features to enforce the bounds.
    :param temperature_mode: (str) Any time we have to apply a softmax to approximate the argmax in the categorical
        features, we use a softmax with a temperature. If we choose to cool this softmax, then we start at a high
        temperature in the optimization and as the optimization progresses we cool the softmax in order that it is more
        concentrated on the maximum. When we choose heating, the opposite process occurs. Accepted modes are: ['cool',
        'constant', 'heat'].
    :param true_bn_stats: (list of tuples) Optional, only if the bathcnorm prior is used. These are the true batch norm
        parameters from the client.
    :param lr_scheduler: (bool) Toggle to use an lr_scheduler.
    :param soteria_defended_layer: (int) The index of the layer that is defended by SOTERIA.
    :param verbose: (bool) Toggle to display the progress of the inversion process.
    :param device: The device on which the tensors are located and the calculation should take place. Note that pytorch
        will throw an error if some tensors are not on the same device.
    :return: (torch.Tensor) The reconstructed datapoint/batch.
    """

    if two_staged:
        assert len(learning_rates) == 2, 'For two staged optimization, we require two learning rates'
        assert len(max_iterations) == 2, 'For two staged optimization, we require two iteration limits'

    if not two_staged:
        learning_rates = (learning_rates, )
        max_iterations = (max_iterations, )

    rec_loss_function = {
        'squared_error': _squared_error_loss,
        'cosine_sim': _cosine_similarity_loss,
        'weighted_combined': _weighted_CS_SE_loss,
        'norm_weighted_combined': _gradient_norm_weighted_CS_SE_loss
    }

    available_priors = {
        'categorical_prior': _categorical_prior,
        'cont_uniform': _continuous_uniform_prior,
        'cont_joint_gmm': _joint_gmm_prior,
        'cont_mean_field_gmm': _mean_field_gmm_prior,
        'cat_mean_field_JS': _categorical_mean_field_jensen_shannon_prior,
        'cat_l2': _categorical_l2_prior,
        'cat_l1': _categorical_l1_prior,
        'theoretical_optimal': _theoretical_optimal_prior,
        'theoretical_typicality': _theoretical_typicality_prior,
        'theoretical_marginal': _theoretical_marginal_prior,
        'theoretical_marginal_typicality': _theoretical_marginal_typicality_prior,
        'l2': _all_line_l2_prior,
        'batch_norm': _batch_norm_prior
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

    if priors is not None:
        # will raise a key error of we chose a non-implemented prior
        prior_params = [prior_params[0] for prior_params in priors]
        prior_loss_functions = [available_priors[prior_params[1]] for prior_params in priors]
    else:
        prior_loss_functions = None
        prior_params = None

    if reconstruction_loss not in list(rec_loss_function.keys()):
        raise NotImplementedError(
            f'The desired loss function is not implemented, available loss function are: {list(rec_loss_function.keys())}')

    # initialize the reconstruction batch
    if initialization_mode.startswith('best'):
        x_reconstruct = initialization[initialization_mode](true_data, dataset, true_grad, net, training_criterion,
                                                            true_label, reconstruction_loss=reconstruction_loss,
                                                            device=device)
    elif initialization_mode == 'from_sample':
        assert initialization_sample.size() == true_data.size()
        x_reconstruct = initialization_sample.detach().clone()
    else:
        x_reconstruct = initialization[initialization_mode](true_data, dataset, device)

    # save the non-masked features
    if mask is not None:
        old_non_masked_features = x_reconstruct * (torch.ones_like(x_reconstruct, device=device) - mask)

    T = temperature_configs[temperature_mode][0]

    x_reconstruct.requires_grad = True

    optimizer = torch.optim.Adam([x_reconstruct], lr=learning_rates[0])
    if lr_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iterations[0], eta_min=1e-8)

    for it in range(max_iterations[0]):
        subverbosity = (it % 100 == 0) if verbose else False
        full_rec_loss = closure(optimizer=optimizer, net=net, training_criterion=training_criterion,
                                reconstruction_loss=rec_loss_function[reconstruction_loss],
                                x_reconstruct=x_reconstruct, true_grad=true_grad, true_label=true_label, alpha=alpha,
                                priors=prior_loss_functions, prior_params=prior_params, dataset=dataset, temperature=T,
                                sign_trick=sign_trick, weight_trick=weight_trick, mask=mask, soteria_defended_layer=soteria_defended_layer,
                                gumbel_softmax_trick=gumbel_softmax_trick, softmax_trick=softmax_trick,
                                apply_projection_to_features=apply_projection_to_features, sigmoid_trick=sigmoid_trick,
                                true_bn_stats=true_bn_stats, verbose=subverbosity, current_it=it, device=device)
        optimizer.step(full_rec_loss)
        if lr_scheduler:
            scheduler.step()
        T *= temperature_configs[temperature_mode][1]
        if mask is not None:
            with torch.no_grad():
                x_reconstruct *= mask
                x_reconstruct += old_non_masked_features

    if two_staged:

        # create feature masks
        index_map = dataset.train_feature_index_map
        cat_mask = torch.ones_like(x_reconstruct)
        for feature_type, feature_index in index_map.values():
            if feature_type == 'cont':
                cat_mask[:, feature_index] = 0.
        cont_mask = torch.ones_like(x_reconstruct) - cat_mask

        # project the categorical features
        original_continuous_features = x_reconstruct.clone().detach() * cont_mask
        x_reconstruct.detach()
        x_reconstruct = dataset.decode_batch(x_reconstruct, standardized=dataset.standardized)
        x_reconstruct = dataset.encode_batch(x_reconstruct, standardize=dataset.standardized)
        reconstructed_categorical_features = x_reconstruct * cat_mask
        x_reconstruct = x_reconstruct * cat_mask + cont_mask * original_continuous_features

        x_reconstruct.requires_grad_()
        optimizer = torch.optim.Adam([x_reconstruct], lr=learning_rates[1])

        # remove the priors concerning categorical features
        if priors is not None:
            prior_params = [prior_params[0] for prior_params in priors if not prior_params[1].startswith('cat')]
            prior_loss_functions = [available_priors[prior_params[1]] for prior_params in priors if
                                    not prior_params[1].startswith('cat')]
        else:
            prior_loss_functions = None
            prior_params = None

        # optimization loop
        for it in range(max_iterations[1]):
            subverbosity = (it % 100 == 0) if verbose else False
            full_rec_loss = closure(optimizer=optimizer, net=net, training_criterion=training_criterion,
                                    reconstruction_loss=rec_loss_function[reconstruction_loss],
                                    x_reconstruct=x_reconstruct, true_grad=true_grad, true_label=true_label,
                                    alpha=alpha, apply_projection_to_features=apply_projection_to_features,
                                    priors=prior_loss_functions, prior_params=prior_params, dataset=dataset,
                                    mask=cont_mask, weight_trick=weight_trick, gumbel_softmax_trick=False,
                                    softmax_trick=False, sigmoid_trick=sigmoid_trick, sign_trick=sign_trick,
                                    true_bn_stats=true_bn_stats, verbose=subverbosity, current_it=it, 
                                    soteria_defended_layer=soteria_defended_layer, device=device)
            optimizer.step(full_rec_loss)
            with torch.no_grad():
                x_reconstruct *= cont_mask
                x_reconstruct += reconstructed_categorical_features

    if sigmoid_trick:
        x_reconstruct = continuous_sigmoid_bound(x_reconstruct, dataset=dataset)

    return x_reconstruct


def alternating_invert_grad(net, training_criterion, true_grad, true_label, true_data,
                            reconstruction_loss='squared_error', initialization_mode='uniform',
                            initialization_sample=None, learning_rates=(0.06, 0.06), alpha=1., priors=None,
                            dataset=None, max_iterations=(100, 100, 100), refill='fuzzy', sign_trick=True,
                            weight_trick=True, gumbel_softmax_trick=False, softmax_trick=True, sigmoid_trick=False,
                            apply_projection_to_features=None, temperature_mode='cool', true_bn_stats=None, 
                            soteria_defended_layer=None, verbose=False, device='cpu'):
    """
    Performs the gradient inversion by alternating optimization between the continuous and the categorical features, and
    returns the guessed datapoint/batch. For now, it assumes the knowledge of the true labels.

    :param net: (nn.Module) The neural network subject to the gradient inversion attack.
    :param training_criterion: (nn.Module) The training loss function of the neural network with respect to which the
        'received' true gradient has been calculated.
    :param true_grad: (list of torch.Tensor) The received gradient.
    :param true_label: (torch.Tensor) The true label of the datapoint/batch we wish to reconstruct
        (simplifying assumption).
    :param true_data: (torch.Tensor) The original/true data that produced the received gradient. This argument is not
        used in the inversion process, merely, it facilitates to display the progress of the inversion process.
    :param reconstruction_loss: (str) The name of the inversion loss function to be used.
    :param initialization_mode: (str) The initialization scheme to be employed.
    :param initialization_sample: (torch.tensor) The sample from which we start the inversion optimization if the
        initialization mode 'from_sample' is selected. The sample should be of the same size as the true data.
    :param learning_rates: (tuple of floats) The learning rates for the two optimization sub-problems. The first entry
        is the learning rate of the categorical optimization and the second is that of the continuous feature
        optimization.
    :param alpha: (float) A weight parameter for combined loss functions.
    :param priors: (list of tuple(float, str)) The regularization parameter(s) plus the name(s) of the prior(s) we wish
        to use. Default None accounts to no prior.
    :param dataset: (datasets.BaseDataset) The dataset with which we work. It contains usually the data necessary for
        the calculation of the prior. The argument can be ignored if no prior is given.
    :param max_iterations: (tuple of ints) max_iterations[0]: maximum number of optimization rounds, max_iterations[1]:
        maximum number of categorical optimization steps in each optimization round, max_iterations[2]: maximum number
        of continuous feature optimization steps in each optimization round.
    :param refill: (str) If 'fuzzy' is selected, we refill the last non-projected state, if 'hard' is selected, we
        refill the projected categorical entries after each round of optimizing the continuous features.
    :param sign_trick: (bool) Toggle to use the optimization trick, where we take the sign of the gradient for a
        datapoint to update it (FGSM-like updates).
    :param weight_trick: (bool) Toggle to use the weight trick introduced by Geiping et al. The idea behind the trick is
        that by giving more weight to the gradients closer to the input, we help the optimization process to first get
        a good enough grip on what the actual data might be, and afterwards fine-tune.
    :param gumbel_softmax_trick: (bool) Toggle to apply the gumbel-softmax trick to the categorical features.
    :param softmax_trick: (bool) Toggle to apply the softmax trick to the categorical features. Effectively, it serves
        as a structural prior on the features.
    :param sigmoid_trick: (bool) Apply the sigmoid trick to the continuous features to enforce the bounds.
    :param temperature_mode: (str) Any time we have to apply a softmax to approximate the argmax in the categorical
        features, we use a softmax with a temperature. If we choose to cool this softmax, then we start at a high
        temperature in the optimization and as the optimization progresses we cool the softmax in order that it is more
        concentrated on the maximum. When we choose heating, the opposite process occurs. Accepted modes are: ['cool',
        'constant', 'heat'].
    :param true_bn_stats: (list of tuples) Optional, only if the bathcnorm prior is used. These are the true batch norm
        parameters from the client.
    :param soteria_defended_layer: (int) The index of the layer that is defended by SOTERIA.
    :param verbose: (bool) Toggle to display the progress of the inversion process.
    :param device: The device on which the tensors are located and the calculation should take place. Note that pytorch
        will throw an error if some tensors are not on the same device.
    :return: (torch.Tensor) The reconstructed datapoint/batch.
    """
    assert refill in ['fuzzy', 'hard'], 'The selected refill scheme is not available, please select from ' \
                                        'fuzzy or hard'

    rec_loss_function = {
        'squared_error': _squared_error_loss,
        'cosine_sim': _cosine_similarity_loss,
        'weighted_combined': _weighted_CS_SE_loss,
        'norm_weighted_combined': _gradient_norm_weighted_CS_SE_loss
    }

    available_priors = {
        'categorical_prior': _categorical_prior,
        'cont_uniform': _continuous_uniform_prior,
        'cont_joint_gmm': _joint_gmm_prior,
        'cont_mean_field_gmm': _mean_field_gmm_prior,
        'cat_mean_field_JS': _categorical_mean_field_jensen_shannon_prior,
        'cat_l2': _categorical_l2_prior,
        'cat_l1': _categorical_l1_prior,
        'theoretical_optimal': _theoretical_optimal_prior,
        'theoretical_typicality': _theoretical_typicality_prior,
        'theoretical_marginal': _theoretical_marginal_prior,
        'theoretical_marginal_typicality': _theoretical_marginal_typicality_prior,
        'l2': _all_line_l2_prior,
        'batch_norm': _batch_norm_prior
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
        'cool': (100., 0.99),
        'constant': (1., 1.),
        'heat': (0.1, 1.01)
    }

    # create a grouped prior for continuous feature optimization
    if priors is not None:
        cont_prior_params = [prior_params[0] for prior_params in priors if not prior_params[1].startswith('cat')]
        cont_prior_loss_functions = [available_priors[prior_params[1]] for prior_params in priors if not prior_params[1].startswith('cat')]
    else:
        cont_prior_loss_functions = None
        cont_prior_params = None

    # create a grouped prior for categorical feature optimization
    if priors is not None:
        cat_prior_params = [prior_params[0] for prior_params in priors if not prior_params[1].startswith('cont')]
        cat_prior_loss_functions = [available_priors[prior_params[1]] for prior_params in priors if not prior_params[1].startswith('cont')]
    else:
        cat_prior_loss_functions = None
        cat_prior_params = None

    if reconstruction_loss not in list(rec_loss_function.keys()):
        raise NotImplementedError(
            f'The desired loss function is not implemented, available loss function are: {list(rec_loss_function.keys())}')

    # initialize the reconstruction batch
    if initialization_mode.startswith('best'):
        x_reconstruct = initialization[initialization_mode](true_data, dataset, true_grad, net, training_criterion,
                                                            true_label, reconstruction_loss=reconstruction_loss,
                                                            device=device)
    elif initialization_mode == 'from_sample':
        assert initialization_sample.size() == true_data.size()
        x_reconstruct = initialization_sample.detach().clone()
    else:
        x_reconstruct = initialization[initialization_mode](true_data, dataset, device)

    # create feature masks
    index_map = dataset.train_feature_index_map
    cat_mask = torch.ones_like(x_reconstruct)
    for feature_type, feature_index in index_map.values():
        if feature_type == 'cont':
            cat_mask[:, feature_index] = 0.
    cont_mask = torch.ones_like(x_reconstruct) - cat_mask

    for it in range(max_iterations[0]):

        # save the reconstructed continuous features
        continuous_reconstructed_features = (x_reconstruct * cont_mask).detach().clone()
        x_reconstruct = x_reconstruct.detach().clone()
        x_reconstruct.requires_grad = True

        # optimize the categorical features
        optimizer = torch.optim.Adam([x_reconstruct], lr=learning_rates[0])
        T = temperature_configs[temperature_mode][0]
        for cat_it in range(max_iterations[1]):
            subverbosity = (cat_it % 100 == 0) if verbose else False
            full_rec_loss = closure(optimizer=optimizer, net=net, training_criterion=training_criterion,
                                    reconstruction_loss=rec_loss_function[reconstruction_loss],
                                    x_reconstruct=x_reconstruct, true_grad=true_grad, true_label=true_label,
                                    alpha=alpha, temperature=T, apply_projection_to_features=apply_projection_to_features,
                                    priors=cat_prior_loss_functions, prior_params=cat_prior_params, dataset=dataset,
                                    mask=cat_mask, weight_trick=weight_trick, gumbel_softmax_trick=gumbel_softmax_trick,
                                    softmax_trick=softmax_trick, sigmoid_trick=sigmoid_trick,
                                    sign_trick=sign_trick, verbose=subverbosity, soteria_defended_layer=soteria_defended_layer,
                                    true_bn_stats=true_bn_stats, current_it=cat_it, device=device)
            optimizer.step(full_rec_loss)
            T *= temperature_configs[temperature_mode][1]
        with torch.no_grad():
            x_reconstruct *= cat_mask
            x_reconstruct += continuous_reconstructed_features

        # project the categorical features and save the fuzzy categoricals
        x_reconstruct = x_reconstruct.detach().clone()
        fuzzy_reconstructed_categorical_features = (x_reconstruct * cat_mask).detach().clone()
        x_reconstruct = dataset.decode_batch(x_reconstruct, standardized=True)
        x_reconstruct = dataset.encode_batch(x_reconstruct, standardize=True)

        # save the projected categorical features
        if refill == 'hard':
            reconstructed_categorical_features = (x_reconstruct * cat_mask).detach().clone()
        else:  # note that this will always only be triggered if the refill mode is fuzzy, due to the assertion above
            reconstructed_categorical_features = fuzzy_reconstructed_categorical_features.detach().clone()

        # optimize the continuous features
        x_reconstruct.requires_grad = True
        optimizer = torch.optim.Adam([x_reconstruct], lr=learning_rates[1])
        for cont_it in range(max_iterations[2]):
            subverbosity = (cont_it % 100 == 0) if verbose else False
            full_rec_loss = closure(optimizer=optimizer, net=net, training_criterion=training_criterion,
                                    reconstruction_loss=rec_loss_function[reconstruction_loss],
                                    x_reconstruct=x_reconstruct, true_grad=true_grad, true_label=true_label,
                                    alpha=alpha, apply_projection_to_features=apply_projection_to_features,
                                    priors=cont_prior_loss_functions, prior_params=cont_prior_params, dataset=dataset,
                                    mask=cont_mask, weight_trick=weight_trick, gumbel_softmax_trick=False,
                                    softmax_trick=False, sigmoid_trick=sigmoid_trick, sign_trick=sign_trick,
                                    true_bn_stats=true_bn_stats, verbose=subverbosity, 
                                    soteria_defended_layer=soteria_defended_layer, current_it=cont_it, device=device)
            optimizer.step(full_rec_loss)
        with torch.no_grad():
            x_reconstruct *= cont_mask
            x_reconstruct += reconstructed_categorical_features

    if sigmoid_trick:
        x_reconstruct = continuous_sigmoid_bound(x_reconstruct, dataset=dataset)

    return x_reconstruct


def invert_grad(net, training_criterion, true_grad, true_label, true_data, reconstruction_loss='squared_error',
                initialization_mode='uniform', initialization_sample=None, learning_rates=None, fish_for_features=None,
                alpha=1., priors=None, dataset=None, max_iterations=None, optimization_mode='naive', refill='fuzzy',
                post_selection=1, return_all=False, sign_trick=True, weight_trick=None, gumbel_softmax_trick=False,
                softmax_trick=True, sigmoid_trick=False, temperature_mode='constant', pooling=None,
                perfect_pooling=False, mask=None, true_bn_stats=None, verbose=False, soteria_defended_layer=None,
                lr_scheduler=False, return_all_reconstruction_losses=False, device='cpu'):
    """
    Gradient inversion wrapper.

    :param net: (nn.Module) The neural network subject to the gradient inversion attack.
    :param training_criterion: (nn.Module) The training loss function of the neural network with respect to which the
        'received' true gradient has been calculated.
    :param true_grad: (list of torch.Tensor) The received gradient.
    :param true_label: (torch.Tensor) The true label of the datapoint/batch we wish to reconstruct
        (simplifying assumption).
    :param true_data: (torch.Tensor) The original/true data that produced the received gradient. This argument is not
        used in the inversion process, merely, it facilitates to display the progress of the inversion process.
    :param reconstruction_loss: (str) The name of the inversion loss function to be used.
    :param initialization_mode: (str) The initialization scheme to be employed.
    :param initialization_sample: (torch.tensor) The sample from which we start the inversion optimization if the
        initialization mode 'from_sample' is selected. The sample should be of the same size as the true data.
    :param learning_rates: (tuple of floats or float) Depending on the optimization mode, a tuple of learning rates for
        the categorical and then the continuous optimizer shall be given (for modes: 'two_staged' and 'alternating').
        For single stage 'naive' optimization, a single learning rate is required. If the argument remains unfilled, we
        default to 0.06 for all learning rates.
    :param fish_for_features: (list) If this argument is given, the optimization is restricted only to these features.
    :param alpha: (float) A weight parameter for combined loss functions.
    :param priors: (list of tuple(float, str)) The regularization parameter(s) plus the name(s) of the prior(s) we wish
        to use. Default None accounts to no prior.
    :param dataset: (datasets.BaseDataset) The dataset with which we work. It contains usually the data necessary for
        the calculation of the prior. The argument can be ignored if no prior is given.
    :param max_iterations: (int) Maximum number of iterations to be performed by the optimizer to recover the data.
    :param optimization_mode: (str) Control the optimization process for the inversion. Available modes are:
        - 'naive': Uses a single optimization loop of max_iterations iterations to recover the data.
        - 'two_staged': First optimizes the data jointly, then projects the categorical features and optimizes only for
                        the continuous features.
        - 'alternating': Alternates between optimizing the categorical features and the continuous features. In this case
                         max_iterations has to be a tuple (or any iterable), where the first entry stands for the number
                         of optimization rounds, the second for the number of max iterations for the categorical
                         features in each round, and the third for the maximum steps for the continuous optimization
                         stage.
    :param refill: (str) The mode of refilling the categorical entries into the reconstructed sample in the alternating
        optimization scheme. If 'fuzzy' is selected, we refill the last non-projected state, if 'hard' is selected, we
        refill the projected categorical entries.
    :param post_selection: (int) The best reconstruction based on the reconstruction loss will be returned from
        'post_selection' number of randomly reinitialized trials.
    :param return_all: (bool) The 'post_selection' argument is greater than 1, toggling this flag allows the user to
        retrieve not just the best data point based on the loss, but all the candidates over the restarts.
    :param sign_trick: (bool) Toggle to use the optimization trick, where we take the sign of the gradient for a
        datapoint to update it (FGSM-like updates).
    :param weight_trick: (bool) Toggle to use the weight trick introduced by Geiping et al. The idea behind the trick is
        that by giving more weight to the gradients closer to the input, we help the optimization process to first get
        a good enough grip on what the actual data might be, and afterwards fine-tune.
    :param gumbel_softmax_trick: (bool) Apply the gumbel-softmax trick to optimizing the categorical features.
    :param softmax_trick: (bool) Toggle to apply the softmax trick to the categorical features. Effectively, it serves
        as a structural prior on the features.
    :param sigmoid_trick: (bool) Apply the sigmoid trick to the continuous features to enforce the bounds.
    :param temperature_mode: (str) Any time we have to apply a softmax to approximate the argmax in the categorical
        features, we use a softmax with a temperature. If we choose to cool this softmax, then we start at a high
        temperature in the optimization and as the optimization progresses we cool the softmax in order that it is more
        concentrated on the maximum. When we choose heating, the opposite process occurs. Accepted modes are: ['cool',
        'constant', 'heat'].
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
    :param mask: (torch.tensor) If a mask is given, only the unmasked features will be optimized. This mask has to be of
        the same dimensions as the ground truth data.
    :param true_bn_stats: (list of tuples) Optional, only if the bathcnorm prior is used. These are the true batch norm
        parameters from the client.
    :param verbose: (bool) Toggle to display the progress of the inversion process.
    :param soteria_defended_layer: (int) The index of the layer that is defended by SOTERIA.
    :param lr_scheduler: (bool) Toggle to use an lr_scheduler.
    :param return_all_reconstruction_losses: (bool) Toggle to return all the reconstruction losses. Note that this
        parameter only has an effect if 'return_all' is already set to True.
    :param device: The device on which the tensors are located and the calculation should take place. Note that pytorch
        will throw an error if some tensors are not on the same device.
    :return: (torch.tensor or tuple(torch.tensor, list(torch.tensor))) The reconstructed datapoint/batch. Or if the
        flag 'return_all' is set to true, next to the best reconstruction also all the 'post_selection' number of
        reconstructions are returned.
    """
    available_optimization_modes = ['naive', 'two_staged', 'alternating', 'fish']
    if weight_trick is None:
        weight_trick = False  # if reconstruction_loss == 'cosine_sim' else True

    rec_loss_function = {
        'squared_error': _squared_error_loss,
        'cosine_sim': _cosine_similarity_loss,
        'weighted_combined': _weighted_CS_SE_loss,
        'norm_weighted_combined': _gradient_norm_weighted_CS_SE_loss
    }

    best_reconstruction = None
    best_score = None
    all_reconstructions = []
    all_reconstruction_losses = []
    for ps in range(post_selection):

        if verbose:
            print(f'Reconstructing Ensemble Sample: {ps+1}/{post_selection}', end='\r')

        if optimization_mode == 'naive':
            if max_iterations is None:
                max_iterations = 1000
            if learning_rates is None:
                learning_rates = 0.06
            current_candidate = naive_invert_grad(net=net,
                                                  training_criterion=training_criterion,
                                                  true_grad=true_grad,
                                                  true_label=true_label,
                                                  true_data=true_data,
                                                  reconstruction_loss=reconstruction_loss,
                                                  initialization_mode=initialization_mode,
                                                  initialization_sample=initialization_sample,
                                                  learning_rates=learning_rates,
                                                  alpha=alpha,
                                                  priors=priors,
                                                  dataset=dataset,
                                                  max_iterations=max_iterations,
                                                  two_staged=False,
                                                  temperature_mode=temperature_mode,
                                                  sign_trick=sign_trick,
                                                  mask=mask,
                                                  weight_trick=weight_trick,
                                                  gumbel_softmax_trick=gumbel_softmax_trick,
                                                  softmax_trick=softmax_trick,
                                                  sigmoid_trick=sigmoid_trick,
                                                  apply_projection_to_features=None,
                                                  true_bn_stats=true_bn_stats,
                                                  lr_scheduler=lr_scheduler,
                                                  soteria_defended_layer=soteria_defended_layer,
                                                  verbose=False,
                                                  device=device)
        elif optimization_mode == 'fish':
            if max_iterations is None:
                max_iterations = 1000
            if learning_rates is None:
                learning_rates = 0.06
            assert fish_for_features is not None, 'Please list for which features to fish'
            mask = create_feature_mask(x=true_data, dataset=dataset, feature_names=fish_for_features)
            current_candidate = naive_invert_grad(net=net,
                                                  training_criterion=training_criterion,
                                                  true_grad=true_grad,
                                                  true_label=true_label,
                                                  true_data=true_data,
                                                  reconstruction_loss=reconstruction_loss,
                                                  initialization_mode=initialization_mode,
                                                  initialization_sample=initialization_sample,
                                                  learning_rates=learning_rates,
                                                  alpha=alpha,
                                                  priors=priors,
                                                  dataset=dataset,
                                                  max_iterations=max_iterations,
                                                  two_staged=False,
                                                  temperature_mode=temperature_mode,
                                                  sign_trick=sign_trick,
                                                  mask=mask,
                                                  weight_trick=weight_trick,
                                                  gumbel_softmax_trick=gumbel_softmax_trick,
                                                  softmax_trick=softmax_trick,
                                                  sigmoid_trick=sigmoid_trick,
                                                  apply_projection_to_features=fish_for_features,
                                                  true_bn_stats=true_bn_stats,
                                                  lr_scheduler=lr_scheduler,
                                                  soteria_defended_layer=soteria_defended_layer,
                                                  verbose=False,
                                                  device=device)

        elif optimization_mode == 'two_staged':
            if max_iterations is None:
                max_iterations = (1000, 1000)
            if learning_rates is None:
                learning_rates = (0.06, 0.06)
            current_candidate = naive_invert_grad(net=net,
                                                  training_criterion=training_criterion,
                                                  true_grad=true_grad,
                                                  true_label=true_label,
                                                  true_data=true_data,
                                                  reconstruction_loss=reconstruction_loss,
                                                  initialization_mode=initialization_mode,
                                                  initialization_sample=initialization_sample,
                                                  learning_rates=learning_rates,
                                                  alpha=alpha,
                                                  priors=priors,
                                                  dataset=dataset,
                                                  max_iterations=max_iterations,
                                                  two_staged=True,
                                                  temperature_mode=temperature_mode,
                                                  sign_trick=sign_trick,
                                                  weight_trick=weight_trick,
                                                  gumbel_softmax_trick=gumbel_softmax_trick,
                                                  softmax_trick=softmax_trick,
                                                  sigmoid_trick=sigmoid_trick,
                                                  apply_projection_to_features=None,
                                                  true_bn_stats=true_bn_stats,
                                                  lr_scheduler=lr_scheduler,
                                                  soteria_defended_layer=soteria_defended_layer,
                                                  verbose=False,
                                                  device=device)

        elif optimization_mode == 'alternating':
            if max_iterations is None:
                max_iterations = (100, 50, 50)
            assert refill in ['fuzzy', 'hard'], 'The selected refill scheme is not available, please select from ' \
                                                'fuzzy or hard'
            if learning_rates is None:
                learning_rates = (0.06, 0.06)
            current_candidate = alternating_invert_grad(net=net,
                                                        training_criterion=training_criterion,
                                                        true_grad=true_grad,
                                                        true_label=true_label,
                                                        true_data=true_data,
                                                        reconstruction_loss=reconstruction_loss,
                                                        initialization_mode=initialization_mode,
                                                        initialization_sample=initialization_sample,
                                                        learning_rates=learning_rates,
                                                        alpha=alpha,
                                                        priors=priors,
                                                        dataset=dataset,
                                                        max_iterations=max_iterations,
                                                        refill=refill,
                                                        temperature_mode=temperature_mode,
                                                        sign_trick=sign_trick,
                                                        weight_trick=weight_trick,
                                                        gumbel_softmax_trick=gumbel_softmax_trick,
                                                        softmax_trick=softmax_trick,
                                                        apply_projection_to_features=None,
                                                        true_bn_stats=true_bn_stats,
                                                        soteria_defended_layer=soteria_defended_layer,
                                                        verbose=False,
                                                        device=device)

        else:
            raise ValueError(f'Optimization scheme not implemented, please choose from: {available_optimization_modes}')

        # save the current candidate for return if required
        if return_all or pooling is not None:
            all_reconstructions.append(current_candidate.detach().clone())

        # get its gradient and check how well it fits
        # reapply projections
        candidate_loss = training_criterion(net(dataset.project_batch(current_candidate.detach().clone(), standardized=dataset.standardized)), true_label)
        candidate_gradient = torch.autograd.grad(candidate_loss, net.parameters())
        candidate_gradient = [grad.detach() for grad in candidate_gradient]
        candidate_reconstruction_loss = rec_loss_function[reconstruction_loss](candidate_gradient, true_grad, device,
                                                                               weights=None, alpha=alpha).item()
        all_reconstruction_losses.append(candidate_reconstruction_loss)
        # check if this loss is better than our current best, if yes replace it and the current datapoint
        if best_reconstruction is None or candidate_reconstruction_loss < best_score:
            best_reconstruction = current_candidate.detach().clone()
            best_score = candidate_reconstruction_loss


    if pooling is not None:

        # pool
        if perfect_pooling:
            pooled_reconstruction = pooled_ensemble(all_reconstructions, true_data, dataset, pooling=pooling)
        else:
            pooled_reconstruction = pooled_ensemble(all_reconstructions, best_reconstruction, dataset, pooling=pooling)

        # return
        if return_all:
            if return_all_reconstruction_losses:
                return pooled_reconstruction, all_reconstructions, all_reconstruction_losses
            else:
                return pooled_reconstruction, all_reconstructions
        else:
            return pooled_reconstruction
    
    else:
        if return_all:
            if return_all_reconstruction_losses:
                return best_reconstruction, all_reconstructions, all_reconstruction_losses
            else:
                return best_reconstruction, all_reconstructions
        else:
            return best_reconstruction
