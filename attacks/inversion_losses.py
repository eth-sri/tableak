import torch


def _squared_error_loss(reconstruct_gradient, true_grad, device, weights=None, alpha=None):
    """
    Implements the squared loss function for retrieving the gradient. Optionally it is weighted by parameter groups
    (optimization trick).

    :param reconstruct_gradient: (list of torch.Tensor) A list containing the reconstruction gradient for each parameter
        group.
    :param true_grad: (list of torch.Tensor) A list containing the true gradient for each parameter group.
    :param device: (str) The name of the device on which the tensors of the process are stored.
    :param weights: (torch.Tensor) An optional argument, if given, the squared error corresponding to each parameter
        group is multiplied by the corresponding weight.
    :return: (torch.Tensor) The extended reconstruction loss 'rec_loss'.
    """
    rec_loss = torch.as_tensor([0.0], device=device)
    if weights is not None:
        for rec_g, in_g, w in zip(reconstruct_gradient, true_grad, weights):
            rec_loss += ((rec_g - in_g) * w).pow(2).mean()
    else:
        for rec_g, in_g in zip(reconstruct_gradient, true_grad):
            rec_loss += (rec_g - in_g).pow(2).mean()
    return rec_loss


def _cosine_similarity_loss(reconstruct_gradient, true_grad, device, weights=None, alpha=None):
    """
    Implements the cosine similarity based loss function.

    :param reconstruct_gradient: (list of torch.Tensor) A list containing the reconstruction gradient for each parameter
        group.
    :param true_grad: (list of torch.Tensor) A list containing the true gradient for each parameter group.
    :param device: (str) The name of the device on which the tensors of the process are stored.
    :param weights: (torch.Tensor) An optional argument, if given, the squared error corresponding to each parameter
        group is multiplied by the corresponding weight.
    :return: (torch.Tensor) The extended reconstruction loss 'rec_loss'.
    """
    scalar_prod = torch.as_tensor([0.0], device=device)
    true_norm = torch.as_tensor([0.0], device=device)
    rec_norm = torch.as_tensor([0.0], device=device)
    if weights is not None:
        for rec_g, in_g, w in zip(reconstruct_gradient, true_grad, weights):
            scalar_prod += (rec_g * in_g * w).sum() * w  # elementwise product and then sum --> euclidean norm
            true_norm += in_g.pow(2).sum() * w
            rec_norm += rec_g.pow(2).sum() * w
    else:
        for rec_g, in_g in zip(reconstruct_gradient, true_grad):
            scalar_prod += (rec_g * in_g).sum()  # elementwise product and then sum --> euclidean norm
            true_norm += in_g.pow(2).sum()
            rec_norm += rec_g.pow(2).sum()
    rec_loss = 1 - scalar_prod / (true_norm.sqrt() * rec_norm.sqrt())
    return rec_loss


def _gradient_norm_weighted_CS_SE_loss(reconstruct_gradient, true_grad, device, weights=None, alpha=1e-5):
    """
    Implements a linear combination of the squared error loss and the cosine similarity loss. The weighting is based
    on the length of the true gradient, where the idea is that for shorter gradients the cosine similarity loss should
    work better. Hence, the combination formula is SE * norm(true_grad) + CS * alpha/norm(true_grad), where alpha is a
    hyperparameter.

    :param reconstruct_gradient: (list of torch.Tensor) A list containing the reconstruction gradient for each parameter
        group.
    :param true_grad: (list of torch.Tensor) A list containing the true gradient for each parameter group.
    :param device: (str) The name of the device on which the tensors of the process are stored.
    :param weights: (torch.Tensor) An optional argument, if given, the squared error corresponding to each parameter
        group is multiplied by the corresponding weight.
    :param alpha: (float) Combination parameter.
    :return: (torch.Tensor) The extended reconstruction loss 'rec_loss'.
    """
    # calculate the norm of the gradient
    norm = torch.as_tensor([0.0], device=device)
    for true_g in true_grad:
        norm += true_g.pow(2).sum().sqrt()
    square_loss = _squared_error_loss(reconstruct_gradient, true_grad, device, weights=weights)
    cs_loss = _cosine_similarity_loss(reconstruct_gradient, true_grad, device)

    rec_loss = norm * square_loss + (alpha / norm) * cs_loss

    return rec_loss


def _weighted_CS_SE_loss(reconstruct_gradient, true_grad, device, weights=None, alpha=1e-5):
    """
    Implements a weighted linear combination of the square loss and the cosine similarity loss. The weighting parameter
    is given by alpha.

    :param reconstruct_gradient: (list of torch.Tensor) A list containing the reconstruction gradient for each parameter
        group.
    :param true_grad: (list of torch.Tensor) A list containing the true gradient for each parameter group.
    :param device: (str) The name of the device on which the tensors of the process are stored.
    :param weights: (torch.Tensor) An optional argument, if given, the squared error corresponding to each parameter
        group is multiplied by the corresponding weight.
    :param alpha: (float) Combination parameter.
    :return: (torch.Tensor) The extended reconstruction loss 'rec_loss'.
    """
    square_loss = _squared_error_loss(reconstruct_gradient, true_grad, device, weights=weights)
    cs_loss = _cosine_similarity_loss(reconstruct_gradient, true_grad, device)

    rec_loss = square_loss + alpha * cs_loss

    return rec_loss
