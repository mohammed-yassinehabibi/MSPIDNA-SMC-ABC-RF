from MSPIDNA_SMC_ABC_RF import Experiment

'''
Parameters of the experiment:
- nb_samples_mspidna_training: number of data points to generate for the training of MSPIDNA (summary-statistics generator)
- nb_samples_abc_step: number of data points to generate for an ABC step (posterior inference)
- sample_size: number of individual sampled in each data point
- population_size: total size of the population
- sequence_length: length of the sequences (in base pairs)
- len_seq_final: length of the final sequences (number of base pairs kept)
- mr_min: minimum mutation rate for the prior
- mr_max: maximum mutation rate for the prior
- root_distribution: nucleotides distribution in the ancestral sequence
- null_diag: boolean to have a null diagonal in the transition matrix
- full_seq: boolean to generate with the "full-sequence" approach or "SNPS-only" approach
- nb_iter: number of iterations T 
- idx_start, idx_end : |end index - start and + 1| is the number of points to infer
- nb_iter_start: start iteration for the ABC-SMC-RF (usually 1 if we didn't already run it for the specified idx)
- nb_iter_end: end iteration for the ABC-SMC-RF 
- real: boolean to use real data or generated data (empirically root_distribution=[0.32,0.23,0.29,0.16] for mtDNA)
- model_custom_exp_name: prior-specific name for the experiment (for the file names saved)
- custom_exp_name: custom experiment name (for the file names saved)
- mspidna_training: boolean to train MSPIDNA or not (not necessary if already trained for the specified prior)
Outputs:
- The data folder containing the data points : exp.data_folder
- The results folder containing the summary statistics : exp.results_folder
- The DRF folder containing the DRF weights and posterior samples for each step : exp.DRF_folder
- The DRF subfolder contining the figures : exp.DRF_folder + f'figures/'
'''


exp_null_diag1 = Experiment(nb_samples_mspidna_training=5000, nb_samples_abc_step=5000, sample_size=100, population_size=1000, sequence_length=4000, 
                 len_seq_final=4000, mr_min=1e-7, mr_max=1e-5, root_distribution=[0.32,0.23,0.29,0.16], null_diag=False,
                 full_seq=True, nb_iter=1, idx_start=0, idx_end=0, nb_iter_start=1, nb_iter_end=5, real=True, 
                 model_custom_exp_name="low_mr", custom_exp_name="", mspidna_training=True)

exp_null_diag2 = Experiment(nb_samples_mspidna_training=5000, nb_samples_abc_step=5000, sample_size=100, population_size=1000, sequence_length=4000, 
                 len_seq_final=4000, mr_min=1e-6, mr_max=1e-4, root_distribution=[0.32,0.23,0.29,0.16], null_diag=False,
                 full_seq=True, nb_iter=1, idx_start=0, idx_end=0, nb_iter_start=1, nb_iter_end=5, real=True, 
                 model_custom_exp_name="high_mr", custom_exp_name="", mspidna_training=True)

exp_null_diag3 = Experiment(nb_samples_mspidna_training=5000, nb_samples_abc_step=5000, sample_size=100, population_size=1000, sequence_length=4000, 
                 len_seq_final=4000, mr_min=1e-5, mr_max=1e-3, root_distribution=[0.32,0.23,0.29,0.16], null_diag=False,
                 full_seq=True, nb_iter=1, idx_start=0, idx_end=0, nb_iter_start=1, nb_iter_end=5, real=True, 
                 model_custom_exp_name="vh_mr", custom_exp_name="", mspidna_training=True)

exp_null_diag1.run()

exp_null_diag2.run()

exp_null_diag3.run()