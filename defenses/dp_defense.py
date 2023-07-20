# Implementation of the DP SGD approach as in: https://arxiv.org/pdf/1607.00133.pdf
# not to self: first clip and then add noise
import torch


def dp_defense(in_grad, scale, C=None, noise_distribution='gaussian'):
    """
    Perturbs the given gradient 'in_grad' to  ensure differential privacy. This is done by first clipping the gradient
    to produce norm <C and then adding either Gaussian or Laplacian noise.

    :param in_grad: (list of torch.tensor) The input gradient.
    :param scale: (float) The scaling parameter of the added noise.
    :param C: (float) The gradient clipping constant.
    :param noise_distribution: (str) The type of the noise distribution. Available are 'gaussian' for Gaussian
        noise and 'laplacian' for Laplacian noise.
    :return:
    """
    out_grad = [grad.detach().clone() for grad in in_grad]
    final_out_grad = []

    # clip
    if C is not None:
        for grad in out_grad:
            grad_norm = grad.pow(2).sum().sqrt()
            grad *= 1/torch.max(torch.ones_like(grad_norm), grad_norm/C)

    # add noise
    for grad in out_grad:
        C = 1 if C is None else C
        if noise_distribution == 'gaussian':
            grad = grad + scale * C * torch.normal(torch.zeros_like(grad), torch.ones_like(grad)) # TODO check C
        elif noise_distribution == 'laplacian':
            grad = grad + scale * C * torch.distributions.laplace.Laplace(loc=0., scale=1.).sample(grad.size())
        else:
            raise NotImplementedError('Only gaussian and laplacian DP implemented')
        final_out_grad.append(grad)

    return final_out_grad
