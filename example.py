import torch
import numpy as np
from attacks import invert_grad
from models import FullyConnected
from datasets import ADULT
from utils import match_reconstruction_ground_truth


print('Instantiate the dataset')
# instantiate and standardize the dataset, and extract the already one-hot encoded data
dataset = ADULT()
dataset.standardize()
Xtrain, ytrain = dataset.get_Xtrain(), dataset.get_ytrain()

# sample a random batch we are going to invert
batch_size = 32
random_indices = np.random.randint(0, len(Xtrain), batch_size)
true_x, true_y = Xtrain[random_indices], ytrain[random_indices]

# now, instantiate a neural network, and calculate its parameter gradient w.r.t. to the above batch
net = FullyConnected(Xtrain.size()[1], [100, 100, 2])
criterion = torch.nn.CrossEntropyLoss()
output = net(true_x)
loss = criterion(output, true_y)
true_grad = [grad.detach() for grad in torch.autograd.grad(loss, net.parameters())]

print('Conduct the attack')
# now we have obtained the true gradient that is shared with the server, and can simulate the attack from the server's side
rec_x = invert_grad(
    net=net, 
    training_criterion=criterion,
    true_grad=true_grad,
    true_label=true_y,  # note that we assume knoweldge of the labels
    true_data=true_x,  # only used for shaping, not used in the actual inversion
    reconstruction_loss='cosine_sim',
    dataset=dataset,
    max_iterations=1500,
    # the following parameter setup below corresponds to TabLeak as in the paper
    post_selection=30,
    softmax_trick=True,
    sigmoid_trick=True,
    pooling='median+softmax',
    verbose=True
)
print('Evaluate')

# rec_x is the reconstruction, but still standardized and one-hot encoded
# to evaluate it, we project both the true data and the reconsutruction back to mixed representation
true_x_mixed, rec_x_mixed = dataset.decode_batch(true_x, standardized=True), dataset.decode_batch(rec_x.detach(), standardized=True)

# now we match the rows of the two batches and obtain an error map, the average of which is the error of the reconstruction
tolerance_map = dataset.create_tolerance_map()
_, error_map, _, _ = match_reconstruction_ground_truth(true_x_mixed, rec_x_mixed, tolerance_map)
reconstruction_accuracy = 100 * (1 - np.mean(error_map))

print(f'Reconstruction accuracy: {reconstruction_accuracy:. 1f}%')