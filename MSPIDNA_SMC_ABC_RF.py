import os
from MSPIDNA import MSPIDNASimulation, GenerateSummaryStatistics
from SMC_DRF import DataSimulator
from SMC_DRF import run_abc_smc_rf

class Experiment:
    '''
    Run a MSPIDNA_SMC_ABC_RF experiment
    Parameters:
    - nb_samples_mspidna_training: number of data points to generate for the training of MSPIDNA (summary-statistics generator)
    - nb_samples_abc_step: number of data points to generate for an ABC step (posterior inference)
    - sample_size: number of individual sampled in each data point
    - population_size: total size of the population
    - sequence_length: length of the sequences (in base pairs)
    - len_seq_final: length of the final sequences (number of base pairs kept)
    - mr_min: minimum mutation rate for the prior
    - mr_max: maximum mutation rate for the prior
    - root_distribution: nucleotides distribution in the ancestral sequence
    - full_seq: boolean to generate with the "full-sequence" approach or "SNPS-only" approach
    - nb_iter: number of iterations T
    - idx_start, idx_end : |end index - start and + 1| is the number of points to infer
    - nb_iter_start: starting iteration for the ABC-SMC-RF (we assume that a MSPIDNA is already trained if nb_iter_start > 1 and data available until nb_iter_start - 1)
    - nb_iter_end: end iteration for the ABC-SMC-RF
    - real: boolean to use real data or generated data
    - model_custom_exp_name: prior-specific name for the experiment (for the file names saved)
    - custom_exp_name: custom experiment name (for the file names saved)
    - mspidna_training: boolean to train MSPIDNA or not (not necessary if already trained for the specified prior)
    
    '''
    def __init__(self, nb_samples_mspidna_training=5000, nb_samples_abc_step=5000, sample_size=100, population_size=1000, 
                 sequence_length=4000, len_seq_final=4000, 
                 mr_min=1e-7, mr_max=1e-5, root_distribution=[0.25,0.25,0.25,0.25], 
                 full_seq=False, nb_iter=1, idx_start=0, idx_end=0, nb_iter_start=1, 
                 nb_iter_end=5, real=False, model_custom_exp_name=None, custom_exp_name=None, mspidna_training=True):
        self.custom_exp_name = custom_exp_name if custom_exp_name is not None else ''
        self.model_custom_exp_name = model_custom_exp_name if model_custom_exp_name is not None else ''
        self.nb_samples_mspidna_training = nb_samples_mspidna_training
        self.nb_samples_abc_step = nb_samples_abc_step
        self.sample_size = sample_size
        self.population_size = population_size
        self.sequence_length = sequence_length
        self.len_seq_final = len_seq_final
        self.path_full_seq = 'full_seq_' if full_seq else ""
        self.path_real = 'real_' if real else ""
        self.data_folder = self.path_full_seq + self.path_real + "data/"
        os.makedirs(os.path.dirname(self.data_folder), exist_ok=True)
        self.results_folder = self.path_full_seq + self.path_real + "results/"
        os.makedirs(os.path.dirname(self.results_folder), exist_ok=True)
        self.DRF_folder = self.path_full_seq + self.path_real + 'DRF_exploitation_of_SPIDNA_results/'
        os.makedirs(os.path.dirname(self.DRF_folder), exist_ok=True)
        self.points_to_infere_path = self.path_full_seq + self.path_real + f'results/parameters_and_SS_0_prior_1_goals{custom_exp_name}{model_custom_exp_name}.pt'
        self.mr_min = mr_min
        self.mr_max = mr_max
        self.root_distribution = root_distribution
        self.full_seq = full_seq
        self.nb_iter = nb_iter
        self.idx_start = idx_start
        self.idx_end = idx_end
        self.nb_iter_start = nb_iter_start
        self.nb_iter_end = nb_iter_end
        self.real = real
        self.perturbation = 'adaptative'
        self.mspidna_training = mspidna_training

    def run(self):
        print(f"Running the experiment {self.custom_exp_name} {self.model_custom_exp_name} {self.path_full_seq + self.path_real}")
        if self.nb_iter_start == 1 and self.mspidna_training:
            self.generate_dataset()
            self.train_model()
        #If we don't use real data, we generate a data point (a set of sequence of nucleotides) for each point to infer
        if not self.real and self.nb_iter_start==1 :
            self.generate_dataset(goal=True)
        #Generate summary statistics for the points to infer with MSPIDNA
        if self.nb_iter_start == 1:
            self.generate_goals_summary_statistics()
        #Run the ABC-SMC-RF process to infer the parameters
        self.complete_abc_smc_rf()
    #1. Generate training dataset for MSPIDNA
    ''' cf. Data_Generation_Class to see parameters to generate data points'''
    def generate_dataset(self, goal=False):
        ds = DataSimulator(nb_samples=self.nb_samples_mspidna_training if not goal else (self.idx_end - self.idx_start + 1), sample_size=self.sample_size, 
                           population_size=self.population_size, sequence_length=self.sequence_length, 
                           len_seq_final=self.len_seq_final, mr_min=self.mr_min, 
                           mr_max=self.mr_max, root_distribution=self.root_distribution, 
                           full_seq=self.full_seq, real=self.real, custom_exp_name=self.custom_exp_name,
                           model_custom_exp_name=self.model_custom_exp_name, goal=goal)
        ds.simulate_and_save(self.nb_iter, idx=0)
    #2. Train the model MSPIDNA
    def train_model(self):
        simulation = MSPIDNASimulation(idx=0, nb_iter=1, data_folder = self.data_folder, 
                                      results_folder = self.results_folder, device=None, 
                                      dataset_name=f"dataset_0_prior_1_adaptative{self.custom_exp_name}{self.model_custom_exp_name}.pt",
                                      custom_exp_name=self.custom_exp_name, model_custom_exp_name=self.model_custom_exp_name)
        simulation.run()
    #3. Generate goals summary statistics with MSPIDNA for the points to infer
    def generate_goals_summary_statistics(self):
        #Generate the summary statistics for all our goal points in one time
        gss = GenerateSummaryStatistics(self.nb_iter, idx = 0, data_folder = self.data_folder, 
                                results_folder = self.results_folder, goal=True, 
                                perturbation=self.perturbation, custom_exp_name=self.custom_exp_name, 
                                model_custom_exp_name=self.model_custom_exp_name)
        gss.run()
    #4. Run the ABC-SMC-RF process to infer the parameters
    def complete_abc_smc_rf(self, plot_1D=True, plot_2D=False):
        for nb_iter in range(self.nb_iter_start, self.nb_iter_end + 1):
            run_abc_smc_rf(nb_samples =self.nb_samples_abc_step, nb_iter=nb_iter, idx_start = self.idx_start, idx_end = self.idx_end, real = self.real, 
                            full_sequence = self.full_seq, plot_1D = plot_1D, plot_2D = plot_2D, root_distribution=self.root_distribution,
                            mr_min=self.mr_min, mr_max=self.mr_max, data_folder=self.data_folder, results_folder=self.results_folder, 
                            DRF_folder=self.DRF_folder, points_to_infere_path=self.points_to_infere_path, perturbation=self.perturbation,
                            custom_exp_name=self.custom_exp_name, model_custom_exp_name=self.model_custom_exp_name)


if __name__ == '__main__':
    exp = Experiment(nb_samples=5000, sample_size=100, population_size=1000, sequence_length=4000, 
                     len_seq_final=4000, mr_min=1e-7, mr_max=1e-5, root_distribution=[0.31,0.25,0.31,0.13], 
                     full_seq=False, nb_iter=1, idx_start=0, idx_end=0, nb_iter_start=1, nb_iter_end=5, real=True, 
                     perturbation='adaptative', custom_exp_name=None)
    exp.run()