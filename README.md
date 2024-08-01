# MSPIDNA-ABC-SMC-RF : Deep Learning and Approximate Bayesian Computation for Finite sites model multivariate inference

## Run [specific_experiment](specific_experiment.py) to run the overall process

### Advices before running the code
```bash
# Create a virtual environment 
python -m venv myenv

# Activate the virtual environment
   # Windows
myenv\Scripts\activate
   # macOS and Linux
source myenv/bin/activate

# Load required python libraries
pip install -r requirements.txt
#NB : the library torch may need to be installed separately depending on your system and if you're using CPU/GPU
```
## How to test the inference method quickly

In  the [results_analysis](results_analysis.ipynb) notebook, we give a function that can be used to run new inference experiments with already trained MSPIDNA networks. We also give an implementation of an idea developed in [Deep Learning and Approximate Bayesian Computation for Finite sites model multivariate inference](Deep_Learning_and_Approximate_Bayesian_Computation_for_Finite_sites_model_multivariate_inference.pdf) to assess the efficiency of the implemented method as well as functions to plot the results of the inference (AD & 2D posteriors). 

## Overall presentation of the repository
This repository gives an implementation of a process made to infer parameters of a stochastic finite sites model whose output data are DNA sequences sampled in a population with a known common ancestor. The process is motivated and explained in [Deep Learning and Approximate Bayesian Computation for Finite sites model multivariate inference](Deep_Learning_and_Approximate_Bayesian_Computation_for_Finite_sites_model_multivariate_inference.pdf).

The data will be generated using the library [msprime](https://tskit.dev/msprime/docs/latest/intro.html), the code for simulating data is in [SMC_DRF](SMC_DRF.py), in the DataSimulator class.

The prossess consists in two parts :
- The first part involves calculating various summary statistics from the DNA sequences sampled in the population using the [MSPIDNA](MSPIDNA.py) neural network. These statistics provide a condensed representation of the data.
  
- The second part consists in running an approximate bayesian computation (ABC) algorithm on the summary statistics to infer the stochastic finite sites model's parameters posterior distribution. The different parts of the adaptive sequential Monte-Carlo ABC with random forests (ABC algorithm used) are implemented in [SMC_DRF](SMC_DRF.py) and [abc_drf](abc_drf.r).
  
The overall process is implemented in [MSPIDNA_SMC_ABC_RF](MSPIDNA_SMC_ABC_RF.py), it enables to gain insights into the evolutionary processes that have shaped the DNA sequences and make inferences about the underlying genetic mechanisms. An example of a specific experiment taking a prior on the parameters, training a MSPIDNA neural network to generate summary-statistics and trying to infer the parameters of a specific (simulated or real) data using SMC-ABC-RF is provided in [specific_experiment](specific_experiment.py).





