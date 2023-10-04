# TabLeak: Tabular Data Leakage in Federated Learning <a href="https://www.sri.inf.ethz.ch/"><img width="100" alt="portfolio_view" align="right" src="http://safeai.ethz.ch/img/sri-logo.svg"></a>

This repository contains the code for our ICML 2023 paper: **[TabLeak: Tabular Data Leakage in Federated Learning](https://openreview.net/forum?id=mRiDy4qGwB)**. In case of any questions, feel free to contact us, either here, or per email to the corresponding author (Mark Vero, mark.vero@inf.ethz.ch).

Our group has published several other works on attacking federated learning systems, which are listed here:
 - [Bayesian Framework for Gradient Leakage](https://arxiv.org/abs/2111.04706), Balunović et al., ICLR 2022
 - [Lamp: Extracting text from gradients with language model priors](https://arxiv.org/abs/2202.08827), Balunović et al., NeurIPS 2022
 - [Data leakage in federated averaging](https://openreview.net/forum?id=e7A0B99zJf&noteId=6wc4VYeT6N), Dimitrov et al., TMLR 2022
 - [Hiding in Plain Sight: Disguising Data Stealing Attacks in Federated Learning](https://arxiv.org/abs/2306.03013), Garov et al., Arxiv preprint 2023

## Installation Guide and Requirements

### Requirements
Our code runs solely on CPUs, where we require at least one CPU core with AVX support, and recommend 50 for maximum 
parallelization when reproducing our results. Further, our experiments rely on the use of the Linux cpu affinity manager 
`taskset`, which is included by default with most Linux installations, coming with the *util-linux* package. In case the 
command is not found on your system, please, follow this [taskset installation guide](https://www.howtodojo.com/taskset-command-not-found/).
For installing the environment of the repository, we recommend using [conda](https://docs.conda.io/en/latest/) and 
provide a corresponding `.yml` file.

### Installation
We provide a conda environment with the code, called *tableak*, enforcing also 
the python version used, namely python 3.8. The environment can be installed as:<br>
```
conda env create -f tableak.yml
```
Prior to running the scripts below, please, activate the conda environment with the following command:
```
conda activate tableak
```
The provided repository is self-contained, in particular, it also contains the datasets required for running. Hence, 
once the environment is set up, the code can be run from the unzipped repository without the need of any further setup 
or downloads.

### Raw Dataset Data 
All raw data files are included in the repository, except for the files of the Health Heritage Prize dataset, as it is over the size limit of GitHub. The required raw data for the Health Heritage dataset can be downloaded from [here](https://files.sri.inf.ethz.ch/tableak/Health_Heritage/). Please, download the files and place them on the path `datasets/Health_Heritage`.

## Usage of TabLeak
Here, we give a minimal example in python on how to mount an attack and calculate its accuracy with our code. The example is ready to be run in `example.py`.
```python
import torch
import numpy as np
from attacks import invert_grad
from models import FullyConnected
from datasets import ADULT
from utils import match_reconstruction_ground_truth


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
    pooling='median+softmax'
)

# rec_x is the reconstruction, but still standardized and one-hot encoded
# to evaluate it, we project both the true data and the reconsutruction back to mixed representation
true_x_mixed, rec_x_mixed = dataset.decode_batch(true_x, standardized=True), dataset.decode_batch(rec_x.detach(), standardized=True)

# now we match the rows of the two batches and obtain an error map, the average of which is the error of the reconstruction
tolerance_map = dataset.create_tolerance_map()
_, error_map, _, _ = match_reconstruction_ground_truth(true_x_mixed, rec_x_mixed, tolerance_map)
reconstruction_accuracy = 100 * (1 - np.mean(error_map))

print(f'Reconstruction accuracy: {reconstruction_accuracy:.1f}%')
```

## Reproducing the Results
Please follow the instructions below to reproduce the results presented in the main paper and in the appendix. 

First we give some general instructions and later give concrete guidance on reproducing the tables and figures in the main body of the manuscript. For details on **reproducing certain experiments in the Appendix** that are not explianed here, please **contact us**. Each experiment is performed on one of the following four datasets (with the corresponding argument to be passed):
- Adult census dataset [1]: `'ADULT'`
- German Credit dataset [1]: `'German'`
- Lawschool Admission dataset [2]: `'Lawschool'`
- Health Heritage datasset [3]: `'HealthHeritage'`

Further, a random seed `random_seed`, a number of samples (monte carlo repetitions, i.e., number of independently 
reconstructed batches) `n_samples`, the number of available 
cpus `max_n_cpus`, and the index of the beginning of the first cpu in the available cpu affinity range `start_cpu` has 
to be passed to each script as an argument. Note that the parallelization happens over the samples, i.e., having 25 cpus 
but only using 25 samples will be equally fast as making 50 cpus available for 50 samples. If less cpus are available 
than samples to be calculated, the program maximally parallelizes over the samples, dividing the workload into chunks 
executed sequentially (i.e., 50 samples but only 25 cpus available --> two chunks of 25 samples, each chunk fully 
parallelized). For the results presented in the paper, we used the random seed 42, collected 50 samples, distributed 
over 50 cores.

### General Results on FedSGD
Below are the instructions to reproduce the data required for all base experiments conducted in the FedSGD 
setting, i.e., the data presented in Tables 1, and 4 and Figure 4 in the main paper, as well as, Tables 17-20, 25-32, 
and Figure 12 in the appendix. The general command to produce the data required for these figures and tables is 
(corresponding to the respective dataset):
```
./fedsgd_attacks.sh <dataset> <n_samples> <random_seed> <max_n_cpus> <start_cpu>
```
The data produced by the execution of this command will be on the path: `experiment_data/large_scale_experiments/<dataset>`.

As an example, to reproduce our presented data on the Adult census dataset, the following command is to be executed:

```
./fedsgd_attacks.sh 'ADULT' 50 42 50 0
```
Where we assumed 50 available cpus. The execution of this command takes on our system around 6 hours in total, at full
parallelization over the samples.

The results can be viewed in the notebook: `fed_sgd.ipynb`.

To reproduce and view the results on the assessment method using the entropy (e.g., Table 4), first you have to have ran the above command.
Then, running the cells in the notebook `assessment_via_entropy.ipynb` will process the entropy data from the saved reconstructions
of the previous execution, and finally display them. Note that the first execution of this notebook might take a few hours, but as the
results are saved, any later execution will load them and take only a few seconds.

### Results on FedAVG
We provide the instructions to reproduce experiments conducted in the FedAvg 
setting, i.e., the data presented in Table 2 in the main paper, and Tables 21-24 in the appendix. The general command 
to produce the data required for these tables is (corresponding to the respective dataset):

```
./fedavg_attacks.sh <dataset> <n_samples> <random_seed> <max_n_cpus> <start_cpu>
```
The data produced by the execution of this command will be on the path: `experiment_data/fedavg_experiments/<dataset>`.

As an example, to reproduce our presented data on the Adult census dataset, the following command is to be executed:

```
./fedavg_attacks.sh 'ADULT' 50 42 50 0
```
Where we assumed 50 available cpus. The execution of this command takes on our system around 2 days in total, at full
parallelization over the samples.

The results can be viewed in the notebook: `fed_avg.ipynb`.

### Reproducing the Experiment: Impact of Network Architecture
Below are the instructions to reproduce the experiment studying the impact of the network architecture on the attack success (Tables 3, & 10-13):

```
./varying_model_architecture.sh <dataset> <n_samples> <random_seed> <max_n_cpus> <start_cpu>
```

As an example, to reproduce our presented data on the Adult census dataset, the following command is to be executed:

```
./varying_model_architecture.sh 'ADULT' 50 42 50 0
```
Where we assumed 50 available cpus.

The results can be viewed in the notebook: `varying_model_architecture.ipynb`.

### Reproducing the Experiment: Attack over Training
Below are the instructions to reproduce the experiment studying the impact of the network training on the attack success (Tables 5 and 15):

```
./attack_during_training.sh <dataset> <n_samples> <random_seed> <max_n_cpus> <start_cpu>
```

As an example, to reproduce our presented data on the Adult census dataset, the following command is to be executed:

```
./attack_during_training.sh 'ADULT' 50 42 50 0
```
Where we assumed 50 available cpus.

The results can be viewed in the notebook: `attack_during_training.ipynb`.

### Reproducing the Experiment: Defending with Noise
Below are the instructions to reproduce the experiment studying the effectiveness of adding noise to the gradients as a defense (Table 6 and Figures 5-8):

```
./noise_defense.sh <dataset> <n_samples> <random_seed> <max_n_cpus> <start_cpu>
```

As an example, to reproduce our presented data on the Adult census dataset, the following command is to be executed:

```
./noise_defense.sh 'ADULT' 50 42 50 0
```
Where we assumed 50 available cpus.

The results can be viewed in the notebook: `noise_defense_evaluation.ipynb`.

## Potential Errors
You might encounter the following error when running some experiments:
```
RuntimeError: a view of a leaf Variable that requires grad is being used in an in-place operation.
```
In this case, please make sure that your pytorch installation is the version 1.7.1. You can check this by opening a 
python console in your environment and running the following two commands:
```python
import torch
print(torch.__version__)
```
In case your version is not 1.7.1, please downgrade to it by running the following command in your terminal with the 
conda environment activated:
```
conda install -c pytorch pytorch=1.7.1
```

## Citation
```
@misc{vero2023tableak,
      title={TabLeak: Tabular Data Leakage in Federated Learning}, 
      author={Mark Vero and Mislav Balunović and Dimitar I. Dimitrov and Martin Vechev},
      year={2023},
      eprint={2210.01785},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## References
[1] D. Dua and C. Graff, “UCI machine learning repository,” 2017. [Online]. Available: http://archive.ics.uci.edu/m <br>
[2] F. L. Wightman, “LSAC national longitudinal bar passage study,” 2017. <br>
[3] Health Heritage, https://www.kaggle.com/c/hhp <br>
