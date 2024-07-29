# MSPIDNA-ABC-SMC-RF : Deep Learning and Approximate Bayesian Computation for Finite sites model multivariate inference

## Run [specific_experiment](specific_experiment.py) to run the overall process


This repository gives an implementation of a process made to infer parameters of a stochastic finite sites model whose output data are DNA sequences sampled in a population with a known common ancestor. The process is motivated and explained in [Deep Learning and Approximate Bayesian Computation for Finite sites model multivariate inference](Deep_Learning_and_Approximate_Bayesian_Computation_for_Finite_sites_model_multivariate_inference.pdf).

The data will be generated using the library [msprime](https://tskit.dev/msprime/docs/latest/intro.html), the code for simulating data is in [SMC_DRF](SMC_DRF.py), in tha DataSimulator class.

The prossess consist in two parts :
- The first part involves calculating various summary statistics from the DNA sequences sampled in the population using the [MSPIDNA](MSPIDNA.py) neural network. These statistics provide a condensed representation of the data.
  
- The second part consists in running an approximate bayesian computation (ABC) algorithm on the summary statistics to infer the parameters posterior distribution of the stochastic finite sites model from them. The different parts of the ABC algorithm we use which is a adaptive sequential Monte-Carlo ABC with random forests are implemented in [SMC_DRF](SMC_DRF.py) and [abc_drf](abc_drf.r).
  
The overall process is implemented in [MSPIDNA_SMC_ABC_RF](MSPIDNA_SMC_ABC_RF.py), it enables to gain insights into the evolutionary processes that have shaped the DNA sequences and make inferences about the underlying genetic mechanisms. An example of a specific experiment taking a prior on the parameters, training a MSPIDNA neural network to generate summary-statistics and try to infer the parameters of a specific (simulated or real) data using SMC-ABC-RF is provided in [specific_experiment](specific_experiment.py).

Finally, in [results_analysis](results_analysis.ipynb), we provide some functions and ideas to state the efficiency of the implemented method. We also provide an implementation of SMC-ABC-RF functions.



