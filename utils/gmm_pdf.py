import torch


def gmm_pdf(x, means, covariances, mixture_weights, nll=False):
    """
    Calculates the probability density of a given Gaussian Mixture Model at point x.

    :param x: (torch.tensor) The point at which the pdf shall be evaluated.
    :param means: (torch.tensor) The means of the Gaussian components.
    :param covariances: (torch.tensor) The covariance matrices of the Gaussian components.
    :param mixture_weights: (torch.Tensor) The weights mixing the Gaussian components.
    :param nll: (bool) Toggle to return the negative log of the pdf value.
    :return: (torch.tensor) The pdf of the given mixture evaluated at x.
    """
    p = torch.as_tensor([0.0], device=x.device)
    for mu, sigma, pi in zip(means, covariances, mixture_weights):
        normalizer = 1. / torch.det(6.28 * sigma).sqrt()
        exponential = torch.exp(-0.5 * torch.matmul(torch.matmul((x - mu).T, torch.inverse(sigma)), (x - mu)))
        p += pi * normalizer * exponential
    if nll:
        return -torch.log(p)
    else:
        return p


def gmm_pdf_batch(x_batch, means, covariances, mixture_weights, nll=False):
    """
    Calculates the average probability density of a given Gaussian Mixture Model at points x_batch.

    :param x_batch: (torch.tensor) The points at which the pdf shall be evaluated.
    :param means: (torch.tensor) The means of the Gaussian components.
    :param covariances: (torch.tensor) The covariance matrices of the Gaussian components.
    :param mixture_weights: (torch.Tensor) The weights mixing the Gaussian components.
    :param nll: (bool) Toggle to return the negative log of the pdf value.
    :return: (torch.tensor) The average pdf of the given mixture evaluated at the points in x_batch.
    """
    if len(x_batch.size()) == 1:
        p = gmm_pdf(x_batch, means, covariances, mixture_weights, nll=nll)
    else:
        p = torch.as_tensor([0.0], device=x_batch.device)
        for x in x_batch:
            p += gmm_pdf(x, means, covariances, mixture_weights, nll=nll)
        p = p / x_batch.size()[0]
    return p
