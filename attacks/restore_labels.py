import torch
import numpy as np


def get_last_relu_and_out_softmax(net, input_batch):
    """
    A function that return the last relu output and the last softmaxed logits.

    :param net: (models.FullyConnected) An instantiated fully connected network from our model class.
    :param input_batch: (torch.tensor) An input batch wrt. which we want to get the activations and the
        out-probabilities.
    :return: (tuple of torch.tensor) The final activations and the final softmax.
    """
    activations = {}
    output = input_batch.clone().detach()
    for layer in net.layers:
        output = layer(output)
        activations[str(layer).split('(')[0]] = output.clone().detach()
    last_relu = activations['LinReLU'].clone().detach()
    out_softmax = torch.nn.functional.softmax(activations['Linear'], dim=1).clone().detach()
    # out_softmax = activations['Linear'].clone().detach()
    return last_relu.detach(), out_softmax.detach()


def post_process_label_reconstructions(counts, batch_size):
    """
    To avoid rounding errors, this function takes the inexact label counts and returns a whole number count by
    one-by-one assembling the counts for each class. At each step, we look at the inexact counts per label, see which
    class has the highest number, and add one of those class labels to our reconstruction, while deducting one from the
    inexact reconstructions. Once out reconstruction sums up to the batch size, we stop the process.

    :param counts: (np.ndarray) The inexact reconstructions.
    :param batch_size: (int) Size of the input batch.
    :return: (np.ndarray) The reconstructed counts.
    """
    post_processed_counts = np.zeros_like(counts)
    while sum(post_processed_counts) < batch_size:
        max_indx = np.argmax(counts).item()
        post_processed_counts[max_indx] += 1
        counts[max_indx] -= 1
    return post_processed_counts


def _restore_labels(net, input_size, gradients, dummy_in=None, n_samples=1000, device=None):
    """
    Inner function restoring the labels based on the technique described by Geng et al. in
    https://arxiv.org/abs/2110.09074.

    :param net: (models.FullyConnected) An instantiated fully connected network from our model class.
    :param input_size: (tuple) The dimensions of the input batch: (batch_size, n_features).
    :param gradients: (list) The gradients of the loss wrt. to the network parameters evaluated at the point which we
        want to recover.
    :param dummy_in: (None or torch.tensor) If given, the intermediate feature maps are estimated not from random data,
        but from this dummy data.
    :param n_samples: (int) If estimating from dummy data, this many random samples are taken to estimate the feature
        maps.
    :param device: (str) The device on which the tensors are located.
    :return: (np.ndarray) The non-rounded recovered counts for each class of (batch_size, n_classes).
    """
    if device is None:
        device = gradients[0].device
    # run dummy data through and extract the last relu and the softmax outputs
    if dummy_in is None:
        dummy_in = torch.randn(n_samples, *input_size[1:], device=device)
    Os, ps = get_last_relu_and_out_softmax(net, dummy_in)

    # compute means of ps and O, compute sum of derivatives in last layer
    ps = ps.mean(dim=0)
    O = (Os.sum(dim=1)).mean()
    K = input_size[0]
    dW = gradients[-2].sum(dim=1) if len(gradients[-1].size()) == 1 else gradients[-1].sum(dim=1)

    # calculate the counts
    counts = K * ps - K * dW / O

    return counts.numpy()


def restore_labels(net, input_size, gradients, dummy_in=None, n_samples=1000, post_process=False, device=None):
    """
    Wrapper function to restore labels based on the technique described by Geng et al. in
    https://arxiv.org/abs/2110.09074 with optional post-processing.

    :param net: (models.FullyConnected) An instantiated fully connected network from our model class.
    :param input_size: (tuple) The dimensions of the input batch: (batch_size, n_features).
    :param gradients: (list) The gradients of the loss wrt. to the network parameters evaluated at the point which we
        want to recover.
    :param dummy_in: (None or torch.tensor) If given, the intermediate feature maps are estimated not from random data,
        but from this dummy data.
    :param n_samples: (int) If estimating from dummy data, this many random samples are taken to estimate the feature
        maps.
    :param post_process: (bool) Toggle to apply post-processing to the raw inexact label counts.
    :param device: (str) The device on which the tensors are located.
    :return: (np.ndarray)
    """
    counts = _restore_labels(net, input_size, gradients, dummy_in, n_samples, device)
    if post_process:
        counts = post_process_label_reconstructions(counts, input_size[0])
    return counts
