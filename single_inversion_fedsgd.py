import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
import torch
from attacks import invert_grad, restore_labels
from defenses import dp_defense
import pickle
import numpy as np
import argparse


def main(args):

    # load all the necessary stuff
    with open(f'{args.metadata_path}/net.pickle', 'rb') as f:
        net = pickle.load(f)
    with open(f'{args.metadata_path}/criterion.pickle', 'rb') as f:
        criterion = pickle.load(f)
    with open(f'{args.metadata_path}/config.pickle', 'rb') as f:
        config = pickle.load(f)
    with open(f'{args.metadata_path}/dataset.pickle', 'rb') as f:
        dataset = pickle.load(f)

    if 'lr_scheduler' not in config:
        config['lr_scheduler'] = False

    # set the random seed
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    # extract the dataset
    Xtrain, ytrain = dataset.get_Xtrain(), dataset.get_ytrain()

    # check if we have the batchnorm prior
    if config['priors'] is not None:
        prior_names = [name for param, name in config['priors']]
        bn_prior_present = 'batch_norm' in prior_names
    else:
        bn_prior_present = False

    # check if there is soteria defense
    if 'soteria_defended_layer' not in config:
        config['soteria_defended_layer'] = None

    # sample a batch form the data and get the gradient
    batchindices = torch.tensor(np.random.randint(Xtrain.size()[0], size=args.batch_size)).to(args.device)
    target_batch = Xtrain[batchindices].clone().detach()
    target_batch_labels = ytrain[batchindices].clone().detach()
    if bn_prior_present:
        output, true_bn_stats_attached = net(target_batch, return_bn_stats=True)
        true_bn_stats = [(bn_mean.detach(), bn_var.detach()) for bn_mean, bn_var in true_bn_stats_attached]
    else:
        output = net(target_batch)
        true_bn_stats = None
    target_loss = criterion(output, target_batch_labels)
    input_gradient = torch.autograd.grad(target_loss, net.parameters())
    input_gradient = [grad.detach() for grad in input_gradient]

    # label inversion if wanted
    gt_labels = target_batch_labels.clone().detach()
    if config['invert_labels']:
        label_counts = restore_labels(net=net, input_size=target_batch.size(), gradients=input_gradient,
                                      post_process=True, device=args.device)[1]
        target_batch_labels = torch.zeros(args.batch_size, device=args.device)
        target_batch_labels[:int(label_counts)] = 1.

    # defense if wanted
    if args.dp_defense:
        # defend the gradient
        input_gradient = dp_defense(in_grad=input_gradient, scale=args.dp_scale,
                                    noise_distribution=config['dp_noise_distribution'])
        args.metadata_path = args.metadata_path + f'/scale_{args.dp_scale}'

    if args.brute_force_labels:
        label_counts = restore_labels(net=net, input_size=target_batch.size(), gradients=input_gradient,
                                      post_process=True, device=args.device)[1]
        rounded_ten_percent_of_batchsize = int(np.ceil(args.batch_size * 0.1))
        label_count_lower, label_count_upper = max(0, int(label_counts) - rounded_ten_percent_of_batchsize),\
                                               min(args.batch_size, int(label_counts) + rounded_ten_percent_of_batchsize)
        label_range = label_count_lower + np.arange(label_count_upper - label_count_lower + 1)

        for i, label_count in enumerate(label_range):
            target_batch_labels = torch.zeros(args.batch_size, device=args.device)
            target_batch_labels[:int(label_count)] = 1.

            batch_recon, all_reconstructions, all_reconstruction_losses = invert_grad(
                net=net,
                training_criterion=criterion,
                true_grad=input_gradient,
                true_label=target_batch_labels.long(),
                true_data=target_batch,
                reconstruction_loss=config['reconstruction_loss'],
                initialization_mode=config['initialization_mode'],
                learning_rates=config['learning_rates'],
                alpha=1.,
                priors=config['priors'],
                dataset=dataset,
                max_iterations=config['max_iterations'],
                optimization_mode=config['optimization_mode'],
                refill=config['refill'],
                post_selection=config['post_selection'],
                return_all=True,
                return_all_reconstruction_losses=True,
                sign_trick=config['sign_trick'],
                weight_trick=config['weight_trick'],
                gumbel_softmax_trick=config['gumbel_softmax_trick'],
                softmax_trick=config['softmax_trick'],
                sigmoid_trick=config['sigmoid_trick'],
                temperature_mode=config['temperature_mode'],
                pooling=config['pooling'],
                perfect_pooling=config['perfect_pooling'],
                true_bn_stats=true_bn_stats,
                lr_scheduler=config['lr_scheduler'],
                soteria_defended_layer=config['soteria_defended_layer'],
                verbose=False,
                device=args.device)

            # save all reconstructions and all their losses
            base_path = f'{args.metadata_path}/batch_size_{args.batch_size}/sample_{args.sample}/label_count_{i}'
            os.makedirs(base_path, exist_ok=True)
            target_batch_np, gt_labels_np = target_batch.detach().cpu().numpy(), gt_labels.detach().cpu().numpy()
            all_reconstructions = [r.detach().cpu().numpy() for r in all_reconstructions]
            all_reconstruction_losses = np.array(all_reconstruction_losses)
            np.save(f'{base_path}/ground_truth_{args.batch_size}_{args.sample}_{i}.npy', target_batch_np)
            np.save(f'{base_path}/true_labels_{args.batch_size}_{args.sample}_{i}.npy', gt_labels_np)
            for j, recon in enumerate(all_reconstructions):
                np.save(f'{base_path}/reconstruction_ensemble_{j}_{args.batch_size}_{args.sample}_{i}.npy', recon)
            np.save(f'{base_path}/all_reconstruction_losses_{args.batch_size}_{args.sample}_{i}.npy', all_reconstruction_losses)

    else:
        # try to reconstruct the batch from the true gradient
        batch_recon, all_recons = invert_grad(
            net=net,
            training_criterion=criterion,
            true_grad=input_gradient,
            true_label=target_batch_labels.long(),
            true_data=target_batch,
            reconstruction_loss=config['reconstruction_loss'],
            initialization_mode=config['initialization_mode'],
            learning_rates=config['learning_rates'],
            alpha=1.,
            priors=config['priors'],
            dataset=dataset,
            max_iterations=config['max_iterations'],
            optimization_mode=config['optimization_mode'],
            refill=config['refill'],
            post_selection=config['post_selection'],
            return_all=True,
            sign_trick=config['sign_trick'],
            weight_trick=config['weight_trick'],
            gumbel_softmax_trick=config['gumbel_softmax_trick'],
            softmax_trick=config['softmax_trick'],
            sigmoid_trick=config['sigmoid_trick'],
            temperature_mode=config['temperature_mode'],
            pooling=config['pooling'],
            perfect_pooling=config['perfect_pooling'],
            true_bn_stats=true_bn_stats,
            lr_scheduler=config['lr_scheduler'],
            soteria_defended_layer=config['soteria_defended_layer'],
            verbose=False,
            device=args.device)

        target_batch, batch_recon, labels = target_batch.detach().cpu().numpy(), batch_recon.detach().cpu().numpy(), gt_labels.detach().cpu().numpy()
        np.save(f'{args.metadata_path}/batch_size_{args.batch_size}/ground_truth_{args.batch_size}_{args.sample}.npy', target_batch)
        np.save(f'{args.metadata_path}/batch_size_{args.batch_size}/reconstruction_{args.batch_size}_{args.sample}.npy', batch_recon)
        np.save(f'{args.metadata_path}/batch_size_{args.batch_size}/true_labels_{args.batch_size}_{args.sample}.npy', labels)
        if config['post_selection'] > 1:
            os.makedirs(f'{args.metadata_path}/batch_size_{args.batch_size}/all_reconstructions_{args.sample}', exist_ok=True)
            for h, recon in enumerate(all_recons):
                np.save(f'{args.metadata_path}/batch_size_{args.batch_size}/all_reconstructions_{args.sample}/ensemble_recon_{h}.npy', recon.detach().cpu().numpy())
        if config['invert_labels']:
            np.save(f'{args.metadata_path}/batch_size_{args.batch_size}/rec_labels_{args.batch_size}_{args.sample}.npy', target_batch_labels.detach().cpu().numpy())


if __name__ == '__main__':
    parser = argparse.ArgumentParser('run_inversion_parser')
    parser.add_argument('--metadata_path', type=str, help='Path to intermediate data')
    parser.add_argument('--batch_size', type=int, help='Selected batch size of inversion')
    parser.add_argument('--sample', type=int, help='The sample number of this experiment')
    parser.add_argument('--random_seed', type=int, default=42, help='Set the random state for reproducibility')
    parser.add_argument('--brute_force_labels', action='store_true', help='Toggle to create the data allowing for a '
                                                                          'brute force label postselection')
    parser.add_argument('--dp_defense', action='store_true', help='Toggle to conduct DP defense')
    parser.add_argument('--dp_scale', type=float, help='Scale of the DP')
    parser.add_argument('--device', type=str, default='cpu', help='Select the device to run the program on')
    in_args = parser.parse_args()
    main(in_args)
