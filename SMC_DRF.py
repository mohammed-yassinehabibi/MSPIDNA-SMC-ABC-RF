import os
import multiprocessing
from tqdm import tqdm
import istarmap
import itertools
import pandas as pd
import seaborn as sns
import numpy as np
import torch
from scipy.stats import gaussian_kde
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.cm import ScalarMappable
from MSPIDNA import GenerateSummaryStatistics
import rpy2.robjects as robjects
import msprime
from priors import generate_prior
from torch.utils.data import TensorDataset

'''
Functions to run the ABC-DRF algorithm AND plot the results
Called by the Experiment class and the run_abc_smc_rf function
'''



class DataSimulator:
    '''DataSimulator:
    Parameters to generate data points:
    - nb_samples: number of data points to generate
    - sample_size: number of individual sampled in each data point
    - population_size: total size of the population
    - sequence_length: length of the sequences (in base pairs)
    - len_seq_final: length of the final sequences (number of base pairs kept)
    - mr_min: minimum mutation rate for the prior
    - mr_max: maximum mutation rate for the prior
    - root_distribution: nucleotides distribution in the ancestral sequence
    - full_seq: boolean to generate with the "full-sequence" approach or "SNPS-only" approach
    - real: boolean to know if we are in a process with real data implied (only to know on which folder to save the data)
    - goal: boolean to know if we are in a goal-oriented process
    Output:
    - The data points saved in the data folder : self.data_folder
    '''
    def __init__(self, nb_samples=5000, sample_size=100, population_size=1000, 
                 sequence_length=4000, len_seq_final=4000, mr_min = 1e-7, mr_max = 1e-5, 
                 root_distribution=[0.25, 0.25, 0.25, 0.25], null_diag=True, data_folder=None,
                 full_seq=False, real=True, goal=False):
        self.goal = goal
        self.null_diag = null_diag
        self.prior_type = "uniform"
        self.data_folder = data_folder
        os.makedirs(os.path.dirname(self.data_folder), exist_ok=True)
        self.device = 'cpu'
        self.nb_samples = nb_samples
        self.sample_size = sample_size
        self.population_size = population_size
        self.sequence_length = sequence_length
        self.len_seq_final = len_seq_final
        self.full_seq = full_seq
        self.prior_params_mutation_rate = {"a": [mr_min], "b": [mr_max], "nb_points": self.nb_samples}
        self.nb_alleles = 4
        self.nb_classes = self.nb_alleles ** 2
        self.root_distributions = torch.tensor([root_distribution], dtype=torch.float64).repeat(self.nb_samples, 1)
        self.vects = np.eye(self.nb_classes)
        self.alleles = ["A", "T", "C", "G"]
        self.keys = ["AA", "AT", "AC", "AG", "TA", "TT", "TC", "TG", "CA", "CT", "CC", "CG", "GA", "GT", "GC", "GG"]
        self.vects = {key: self.vects[i] for i, key in enumerate(self.keys)}
        self.numbers = {allele: i for i, allele in enumerate(self.alleles)}

    # Simulate data with nucleotides encoded as a 16-dim vector
    def simulate(self, args):
        np.random.seed()
        mutation_rate, transition_matrix, root_distribution = args
        tree_seq = msprime.sim_ancestry(samples=self.sample_size // 2, ploidy=2, population_size=self.population_size,
                                        sequence_length=self.sequence_length, discrete_genome=True)
        model = msprime.MatrixMutationModel(self.alleles, root_distribution=root_distribution, transition_matrix=transition_matrix)
        ts = msprime.sim_mutations(tree_seq, rate=mutation_rate, 
                                   discrete_genome=True, model=model)
        if not self.full_seq:
            snps = np.empty((ts.num_sites, self.sample_size, self.nb_classes), dtype=np.uint8)
            for i, variant in enumerate(ts.variants()):
                snps[i] = np.array([self.vects[variant.alleles[0] + allele] for allele in variant.alleles])[variant.genotypes]
            snps = snps.transpose(1, 0, 2).astype(np.uint8)
            pos = ts.tables.asdict()["sites"]["position"]
            x = np.concatenate([np.expand_dims(pos.repeat(self.nb_classes).reshape(-1, self.nb_classes), axis=0), snps])
            if x.shape[1] < self.len_seq_final:
                zeros_to_add = self.len_seq_final - x.shape[1]
                result = np.pad(x, ((0, 0), (0, zeros_to_add), (0, 0)), mode='constant')
            else:
                result = x[:, :self.len_seq_final]
        if self.full_seq:
            snps = np.empty((self.sequence_length, self.sample_size, self.nb_classes), dtype=np.uint8)
            pos = ts.tables.asdict()["sites"]["position"]
            for i, variant in enumerate(ts.variants()):
                snps[int(variant.site.position)] = np.array([self.vects[variant.alleles[0] + allele] for allele in variant.alleles])[variant.genotypes]
            for i in range(self.sequence_length):
                if (len(pos) > 0 and i != pos[0]) or len(pos) == 0:
                    allele = np.random.choice(4, 1, p = root_distribution)
                    vect_allele = np.zeros(16)
                    vect_allele[5 * allele] = 1
                    snps[i] = np.tile(vect_allele, (self.sample_size, 1))
                else:
                    pos = pos[1:]
            snps = snps.transpose(1, 0, 2).astype(np.uint8)
            pos = np.arange(self.len_seq_final)
            result = np.concatenate([np.expand_dims(pos.repeat(self.nb_classes).reshape(-1, self.nb_classes), axis=0), snps[:, :self.len_seq_final]])
        return result
    
    def simulate_and_save(self, nb_iter, idx):
        if nb_iter == 1:
            mutation_rates = generate_prior(type=self.prior_type, **self.prior_params_mutation_rate)[0]
            transition_matrices = generate_prior(type="dirichlet_transition_matrix", n=self.nb_alleles, nb_points=self.nb_samples, null_diag=self.null_diag)
            Y = torch.cat([mutation_rates.unsqueeze(1), transition_matrices.flatten(1, 2)], dim=1)
            if self.null_diag:
                columns_to_keep = [0, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15]
                Y = Y[:, columns_to_keep]
        else:
            Y = torch.load(f'{self.data_folder}' + f'prior_{nb_iter}/Y_{idx}_prior_{nb_iter}.pt', weights_only=False)
            mutation_rates, transition_matrices = from_Y_to_generation_params(Y, nb_iter, idx, null_diag=self.null_diag)
        X = torch.zeros(self.nb_samples, self.sample_size + 1, self.len_seq_final, self.nb_classes)
        args = [(mutation_rates[rate].numpy(), transition_matrices[rate].numpy(), self.root_distributions[rate].numpy()) for rate in range(self.nb_samples)]
        for j, arg in enumerate(tqdm(args, desc=f"Simulating data {idx}")):
            result = self.simulate(arg)
            X[j] = torch.from_numpy(result)
        dataset = TensorDataset(X, Y if isinstance(Y, torch.Tensor) else torch.tensor(Y.values))
        if not self.goal:
            torch.save(dataset, f'{self.data_folder}' + f'dataset_{idx}_prior_{nb_iter}.pt')
        else:
            torch.save(dataset, f'{self.data_folder}' + f'dataset_{idx}_prior_{nb_iter}_goals.pt')
        print(f'Saved {idx}')
        return dataset

    # Simulate data with 0,1,2,3 nucleotides
    def simulate2(self, args):
        np.random.seed()
        mutation_rate, transition_matrix, root_distribution = args
        tree_seq = msprime.sim_ancestry(samples=self.sample_size // 2, ploidy=2, population_size=self.population_size,
                                        sequence_length=self.sequence_length, discrete_genome=True)
        model = msprime.MatrixMutationModel(self.alleles, root_distribution=root_distribution, transition_matrix=transition_matrix)
        ts = msprime.sim_mutations(tree_seq, rate=mutation_rate, 
                                   discrete_genome=True, model=model)
        if not self.full_seq:
            snps = np.empty((ts.num_sites, self.sample_size), dtype=np.uint8)
            for i, variant in enumerate(ts.variants()):
                snps[i] = np.array([self.numbers[allele] for allele in variant.alleles])[variant.genotypes]
            snps = snps.transpose(1, 0).astype(np.uint8)
            x = snps
            if x.shape[1] < self.len_seq_final:
                zeros_to_add = self.len_seq_final - x.shape[1]
                result = np.pad(x, ((0, 0), (0, zeros_to_add), (0, 0)), mode='constant')
            else:
                result = x[:, :self.len_seq_final]
        if self.full_seq:
            ancestral = np.empty((self.sequence_length), dtype=np.uint8)
            snps = np.empty((self.sequence_length, self.sample_size), dtype=np.uint8)
            pos = ts.tables.asdict()["sites"]["position"]
            for i, variant in enumerate(ts.variants()):
                snps[int(variant.site.position)] = np.array([self.numbers[allele] for allele in variant.alleles])[variant.genotypes]
                ancestral[int(variant.site.position)] = self.numbers[variant.alleles[0]]
            for i in range(self.sequence_length):
                if (len(pos) > 0 and i != pos[0]) or len(pos) == 0:
                    allele = np.random.choice(4, 1, p = root_distribution)
                    snps[i] = np.tile(allele, (self.sample_size))
                    ancestral[i] = allele
                else:
                    pos = pos[1:]
            snps = snps.transpose(1, 0).astype(np.uint8)
            pos = np.arange(self.len_seq_final)
            result = snps[:, :self.len_seq_final]
        return ancestral,result
    
def create_csv_for_DRF(points_to_infere_path, result_dataset_path, nb_iter, idx, idx_start,
                       results_folder=None):
    subfolder=f'prior_{nb_iter}/'
    if nb_iter==1: #for the first iteration, we need to save the points parameters and statistics
        targets = torch.load(points_to_infere_path, weights_only=False)
        params_targets = targets.iloc[idx - idx_start, :len(targets.columns)//2].to_frame().T
        statistics_targets = targets.iloc[idx - idx_start, len(targets.columns)//2:].to_frame().T
        params_targets.to_csv(results_folder+f'params_targets_{idx}.csv', index=False)
        statistics_targets.to_csv(results_folder+f'statistics_targets_{idx}.csv', index=False)
    os.makedirs(os.path.dirname(results_folder+subfolder), exist_ok=True)
    #merge all the results folder for the same point if needed...
    result_dataset = torch.load(result_dataset_path, weights_only=False)
    Ydrf, Xdrf = result_dataset.iloc[:, :len(result_dataset.columns)//2], result_dataset.iloc[:, len(result_dataset.columns)//2:]
    if nb_iter == 1:
        Ydrf.to_csv(results_folder+subfolder+f'Ydrf_{idx}_prior_{nb_iter}.csv', index=False)
        Xdrf.to_csv(results_folder+subfolder+f'Xdrf_{idx}_prior_{nb_iter}.csv', index=False)
    else:
        Ydrf.to_csv(results_folder+subfolder+f'Ydrf_{idx}_prior_{nb_iter}.csv', index=False)
        Xdrf.to_csv(results_folder+subfolder+f'Xdrf_{idx}_prior_{nb_iter}.csv', index=False)
    print(f'{nb_iter} : CSV for {idx} saved in {results_folder}{subfolder} : Ydrf_{idx}_prior_{nb_iter}.csv and Xdrf_{idx}_prior_{nb_iter}.csv')

def posterior_sampling( mr_min, mr_max, nb_iter, idx, results_folder, subfolder, data_folder, nb_samples=5_000):
    data_folder = data_folder
    Y = pd.read_csv(results_folder+subfolder+f'Ydrf_{idx}_prior_{nb_iter}.csv') if nb_iter>1 else pd.read_csv(results_folder+subfolder+f'Ydrf_{idx}_prior_{nb_iter}.csv')
    W = pd.read_csv(results_folder+subfolder+f'DRF_weights_{idx}_prior_{nb_iter}.csv').values[:,0] if nb_iter>1 else pd.read_csv(results_folder+subfolder+f'DRF_weights_{idx}_prior_{nb_iter}.csv').values[:,0]
    nb_params = Y.shape[1]
    mins, maxs = np.zeros(nb_params), np.ones(nb_params)
    mins[0], maxs[0] = mr_min, mr_max
    #Resampling 10000 samples from the posterior
    indexes = np.random.choice(len(Y), size=int(100*nb_samples), p=W)
    Y_posterior = Y.iloc[indexes]
    ########### Adaptive approximate Bayesian computation
    std_empirical = np.std(Y_posterior, axis=0)
    noise_std = std_empirical*np.sqrt(2)
    noise = np.random.normal(0, noise_std, Y_posterior.shape)
    Y_posterior = Y_posterior+noise
    #keep only values in the range
    Y_posterior = Y_posterior[Y_posterior > mins].dropna()
    Y_posterior = Y_posterior[Y_posterior < maxs].dropna()
    #normalize the values (sum of each row = 1)
    if nb_params == 13:
        for j in range(4):
            Y_posterior.iloc[:, 1+3*j:1+3*(j+1)] = Y_posterior.iloc[:, 1+3*j:1+3*(j+1)].div(Y_posterior.iloc[:, 1+3*j:1+3*(j+1)].sum(axis=1), axis=0)
    elif nb_params == 17:
        for j in range(4):
            Y_posterior.iloc[:, 1+4*j:1+4*(j+1)] = Y_posterior.iloc[:, 1+4*j:1+4*(j+1)].div(Y_posterior.iloc[:, 1+4*j:1+4*(j+1)].sum(axis=1), axis=0)
    if Y_posterior.shape[0] >= nb_samples:
        Y_posterior = Y_posterior.iloc[:nb_samples]
    else :
        raise ValueError('Not enough samples')
    os.makedirs(os.path.dirname(data_folder + f'prior_{nb_iter+1}/'), exist_ok=True)
    torch.save(Y_posterior, data_folder + f'prior_{nb_iter+1}/Y_{idx}_prior_{nb_iter+1}.pt')
    return Y_posterior

def from_Y_to_generation_params(Y, nb_iter, idx, null_diag=True):
    mutation_rates = torch.tensor(Y['mutation_rate'].values)
    transition_matrices = Y.drop(columns='mutation_rate').values
    if null_diag:
        transition_matrices = np.insert(transition_matrices, 0, 0, axis=1)
        transition_matrices = np.insert(transition_matrices, 5, 0, axis=1)
        transition_matrices = np.insert(transition_matrices, 10, 0, axis=1)
        transition_matrices = np.insert(transition_matrices, 15, 0, axis=1)
    transition_matrices = torch.tensor(transition_matrices).reshape(transition_matrices.shape[0],4,4)
    return mutation_rates, transition_matrices

def plot_1D_posteriors(nb_iter, idx, results_folder):
    params_target= pd.read_csv(results_folder+f'params_targets_{idx}.csv')
    sns.set_style("darkgrid")
    # Create a figure with 4 rows and 3 columns of subplots
    nb_col = params_target.shape[1]//4
    nb_row = 4
    fig, axes = plt.subplots(nb_row, nb_col, figsize=(25, 20), constrained_layout=True)
    axes = axes.flatten()  # Flatten the 2D array of axes to a 1D array for easier indexing
    num_curves = nb_iter
    cmap = get_cmap('viridis')
    for i, ax in enumerate(axes):
        for j in range(1, nb_iter+1):
            if j==1:
                Ydrf = pd.read_csv(results_folder + f'prior_{j}/Ydrf_{idx}_prior_{j}.csv')
                W = pd.read_csv(results_folder + f'prior_{j}/DRF_weights_{idx}_prior_{j}.csv').values[:, 0]
            else:
                Ydrf = pd.read_csv(results_folder + f'prior_{j}/Ydrf_{idx}_prior_{j}.csv')
                W = pd.read_csv(results_folder + f'prior_{j}/DRF_weights_{idx}_prior_{j}.csv').values[:, 0]
            if i+1 < params_target.shape[1]:
                col_to_plot = Ydrf.columns[i+1]
                Y = Ydrf[col_to_plot]
                Y_values = np.linspace(min(Y), max(Y), 1000)
                if j==1:
                    prior_kde = gaussian_kde(Y)
                    prior_density = prior_kde(Y_values)
                    #PLOT PRIOR
                    ax.plot(Y_values, prior_density, color=cmap(0/num_curves))
                    ax.fill_between(Y_values, prior_density, color=cmap(0/num_curves), alpha=0.05)
                posterior_kde = gaussian_kde(Y, weights=W)
                posterior_density = posterior_kde(Y_values)
                #PLOT POSTERIOR
                ax.plot(Y_values, posterior_density, color=cmap(j / num_curves))
                ax.fill_between(Y_values, posterior_density, color=cmap(j / num_curves), alpha=0.05)
                if j==nb_iter:
                    cond_exp_est = np.sum(Y.to_numpy() * W)
                    # PLOT COND EXP
                    ax.axvline(cond_exp_est, color='green', linestyle='--', label='conditional expectation')
                    # Plot the true value
                    if not "real" in results_folder:
                        ax.axvline(params_target[col_to_plot].item(), color='red', linestyle='-', label='true value')
                ax.set_title(f'{col_to_plot} parameter', fontsize=20)
                ax.set_xlabel('Values', fontsize=15)
                ax.set_ylabel('Density', fontsize=15)
                ax.tick_params(axis='x', labelsize=15)  
                ax.tick_params(axis='y', labelsize=15) 

    # Add a main title for the figure
    fig.suptitle(f'SPIDNA-ABC-DRF {idx}', fontsize=20)

    # Colorbar
    norm = plt.Normalize(0, nb_iter)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  
    cbar = fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.1, pad=0.04)
    ticks = np.linspace(0, nb_iter, num=nb_iter+1)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([str(int(tick)) for tick in ticks]) 
    cbar.set_label('t', rotation=0, labelpad=30, fontsize=30)  
    cbar.ax.tick_params(labelsize=20) 
    # Add a global legend
    os.makedirs(os.path.dirname(results_folder + f'figures/prior_{nb_iter}/1D_posteriors/'), exist_ok=True)
    plt.savefig(results_folder + f'figures/prior_{nb_iter}/1D_posteriors/ABC_DRF_{idx}_prior_{nb_iter}.png')
    print(f'{nb_iter} : 1D_posteriors for {idx} saved in {results_folder}figures/prior_{nb_iter}/1D_posteriors/ABC_DRF_{idx}_prior_{nb_iter}.png')
    plt.show()
    
def plot_2D_posterior(i, j, nb_iter, idx, results_folder, subfolder):
    params_target = pd.read_csv(results_folder+f'params_targets_{idx}.csv')
    Ydrf = pd.read_csv(results_folder+subfolder+f'Ydrf_{idx}_prior_{nb_iter}.csv') if nb_iter>1 else pd.read_csv(results_folder+subfolder+f'Ydrf_{idx}_prior_{nb_iter}.csv')
    W = pd.read_csv(results_folder+subfolder+f'DRF_weights_{idx}_prior_{nb_iter}.csv').values[:,0] if nb_iter>1 else pd.read_csv(results_folder+subfolder+f'DRF_weights_{idx}_prior_{nb_iter}.csv').values[:,0]
    Y = Ydrf
    columns = Ydrf.columns
    #Resampling 10000 samples from the posterior
    indexes = np.random.choice(len(Y), size=10000, p=W)
    Y_posterior = Y.iloc[indexes]
    x, y = columns[i], columns[j]
    os.makedirs(os.path.dirname(results_folder+'figures/'+subfolder+f"2D_posteriors/{idx}/"), exist_ok=True)
    sns.set_style("darkgrid")
    if i !=j:           
        joint = sns.jointplot(
            data=Y_posterior,
            x=x, y=y, 
            kind="kde",
            fill=True,  
            cmap='plasma'
        )
        joint.ax_joint.set_xlabel(f'{x}')
        joint.ax_joint.set_ylabel(f'{y}')

        true_theta_1 = params_target[x].item()  
        true_theta_2 = params_target[y].item()  
        plt.scatter(true_theta_1, true_theta_2, color='green')
        cond_exp_est_x, cond_exp_est_y  = np.mean(Y_posterior[x]), np.mean(Y_posterior[y])
        plt.scatter(cond_exp_est_x, cond_exp_est_y, color='red')

        ax_main = joint.ax_joint
        ax_marg_x = joint.ax_marg_x
        ax_marg_y = joint.ax_marg_y
        # true_values
        if not "real" in results_folder:
            ax_marg_x.axvline(true_theta_1, color='green', linestyle='--')
            ax_marg_y.axhline(true_theta_2, color='green', linestyle='--')
        # Conditional expectation
        ax_marg_x.axvline(cond_exp_est_x, color='red', linestyle='--')
        ax_marg_y.axhline(cond_exp_est_y, color='red', linestyle='--')
        # Prior density
        Ydrf_prior = pd.read_csv(results_folder + f'prior_1/Ydrf_{idx}_prior_1.csv')
        Y_prior = Ydrf_prior
        sns.histplot(data=Y_prior, x=x, ax=ax_marg_x, color='grey', stat='density')
        sns.histplot(data=Y_prior, y=y, ax=ax_marg_y, color='grey', orientation='horizontal', stat='density')
        joint.savefig(results_folder+'figures/'+subfolder+f'2D_posteriors/{idx}/joint_distribution_{x}_{y}.png')
        plt.close()
    if i==j:
        #PLOT PRIOR
        Ydrf_prior = pd.read_csv(results_folder + f'prior_1/Ydrf_{idx}_prior_1.csv')
        Y_prior = Ydrf_prior[x]
        Y_values = np.linspace(min(Y_prior), max(Y_prior), 1000)
        prior_kde = gaussian_kde(Y_prior)
        prior_density = prior_kde(Y_values)
        cmap = get_cmap('viridis')
        plt.plot(Y_values, prior_density, label='Prior', color=cmap(0))
        plt.fill_between(Y_values, prior_density, color=cmap(0), alpha=0.05)
        Y = Y[x]
        #PLOT POSTERIOR
        posterior_kde = gaussian_kde(Y, weights=W)
        posterior_density = posterior_kde(Y_values)
        plt.plot(Y_values, posterior_density, label=f'Posterior {nb_iter}', color=cmap(1))
        plt.fill_between(Y_values, posterior_density, color=cmap(1), alpha=0.05)
        # Plot the true value
        if not "real" in results_folder:
            plt.axvline(params_target[x].item(), color='green', linestyle='--')
        # PLOT COND EXP
        cond_exp_est = np.sum(Y.to_numpy() * W)
        plt.axvline(cond_exp_est, color='red', linestyle='--', label='conditional expectation')
        plt.savefig(results_folder+'figures/'+subfolder+f'2D_posteriors/{idx}/joint_distribution_{x}_{x}.png')
        plt.close()

def merge_2D_posteriors(idx, results_folder, subfolder):
    current_dir = os.getcwd()
    image_paths = [results_folder+'figures/'+subfolder+f'2D_posteriors/{idx}/'+file for file in os.listdir(results_folder+'figures/'+subfolder+f'2D_posteriors/{idx}/')]
    image_paths.sort()
    fig, axs = plt.subplots(12, 12, figsize=(120, 120))
    for i,j in itertools.product(range(12), range(12)):
        index = i * 12 + j
        if index < len(image_paths):
            img = plt.imread(image_paths[index])  
            axs[i, j].imshow(img)  
            axs[i, j].axis('off')  
        else:
            axs[i, j].axis('off')  

    plt.subplots_adjust(left=0.04, right=0.05, top=0.05, bottom=0.04, wspace=0.1)  # Ajuster l'espacement entre les subplots
    fig.tight_layout()
    plt.savefig(results_folder+'figures/'+subfolder+'2D_posteriors/'+f'complete_joint_distributions_{idx}.png')
    plt.close()

def plot_1D_posterior_custom(idxs_to_plot, nb_iter, idx, results_folder):
    params_target = pd.read_csv(results_folder+f'params_targets_{idx}.csv')
    sns.set_style("darkgrid")
    # Create a figure with 4 rows and 3 columns of subplots
    if len(idxs_to_plot)>=4:
        nb_col = len(idxs_to_plot)//4
        nb_row = 4
    else:
        nb_col = 1
        nb_row = len(idxs_to_plot)
    fig, axes = plt.subplots(nb_row, nb_col, figsize=(5+7*nb_col, 4+4*nb_row))
    axes = axes.flatten() if nb_row>1 else axes # Flatten the 2D array of axes to a 1D array for easier indexing
    num_curves = nb_iter+1
    cmap = get_cmap('viridis')
    if len(idxs_to_plot) == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        for j in range(1, nb_iter+1):
            if j==1:
                Ydrf = pd.read_csv(results_folder + f'prior_{j}/Ydrf_{idx}_prior_{j}.csv')
                W = pd.read_csv(results_folder + f'prior_{j}/DRF_weights_{idx}_prior_{j}.csv').values[:, 0]
            else:
                Ydrf = pd.read_csv(results_folder + f'prior_{j}/Ydrf_{idx}_prior_{j}.csv')
                W = pd.read_csv(results_folder + f'prior_{j}/DRF_weights_{idx}_prior_{j}.csv').values[:, 0]
            if i+1 < params_target.shape[1]:
                col_to_plot = Ydrf.columns[idxs_to_plot[i]]
                Y = Ydrf[col_to_plot]
                Y_values = np.linspace(min(Y), max(Y), 1000)
                if j==1:
                    prior_kde = gaussian_kde(Y)
                    prior_density = prior_kde(Y_values)
                    #PLOT PRIOR
                    ax.plot(Y_values, prior_density, label='Prior', color=cmap(0/num_curves))
                    ax.fill_between(Y_values, prior_density, color=cmap(0/num_curves), alpha=0.05)
                posterior_kde = gaussian_kde(Y, weights=W)
                posterior_density = posterior_kde(Y_values)
                #PLOT POSTERIOR
                ax.plot(Y_values, posterior_density, label=f'Posterior {j}', color=cmap(j / num_curves))
                ax.fill_between(Y_values, posterior_density, color=cmap(j / num_curves), alpha=0.05)
                if j==nb_iter:
                    cond_exp_est = np.sum(Y.to_numpy() * W)
                    # PLOT COND EXP
                    ax.axvline(cond_exp_est, color='green', linestyle='--', label='conditional expectation')
                    # Plot the true value
                    if not "real" in results_folder:
                        ax.axvline(params_target[col_to_plot].item(), color='red', linestyle='--', label='true value')
                ax.set_title(fr'$\theta_1$', fontsize=15)
                ax.set_xlabel('Values', fontsize=15)
                ax.set_ylabel('Density', fontsize=15)
                ax.tick_params(axis='x', labelsize=15)  
                ax.tick_params(axis='y', labelsize=15) 
    
    # Colorbar
    norm = plt.Normalize(0, nb_iter)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  
    cbar = fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.1, pad=0.04)
    ticks = np.linspace(0, nb_iter, num=nb_iter+1)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([str(int(tick)) for tick in ticks]) 
    cbar.set_label('t', rotation=0, labelpad=30, fontsize=15)  
    cbar.ax.tick_params(labelsize=15) 

    # Add a global legend
    os.makedirs(os.path.dirname(results_folder + f'figures/prior_{nb_iter}/1D_posteriors/'), exist_ok=True)
    plt.savefig(results_folder + f'figures/prior_{nb_iter}/1D_posteriors/ABC_DRF_{idx}_prior_{nb_iter}_custom.png')
    print(f'1D_posteriors for {idx} saved in {results_folder}figures/prior_{nb_iter}/1D_posteriors/ABC_DRF_{idx}_prior_{nb_iter}_custom.png')
    plt.close()

#Plot transitions-matrix posteriors for different priors 
def compare_1D_posteriors(nb_iter, idx, results_folders):
    sns.set_style("darkgrid")
    params_target= pd.read_csv(results_folders[0]+f'params_targets_{idx}.csv')
    # Create a figure with 4 rows and 3 columns of subplots
    nb_col = params_target.shape[1]//4
    nb_row = 4
    fig, axes = plt.subplots(nb_row, nb_col, figsize=(25, 20))
    axes = axes.flatten()  # Flatten the 2D array of axes to a 1D array for easier indexing
    num_curves = len(results_folders)
    #cmap from green to red
    colors = [(0, "green"), (0.5, "orange"), (1, "red")]
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
    for i, ax in enumerate(axes):
        for j, results_folder in enumerate(results_folders):
            params_target= pd.read_csv(results_folder+f'params_targets_{idx}.csv')
            Ydrf = pd.read_csv(results_folder + f'prior_{nb_iter}/Ydrf_{idx}_prior_{nb_iter}.csv')
            W = pd.read_csv(results_folder + f'prior_{nb_iter}/DRF_weights_{idx}_prior_{nb_iter}.csv').values[:, 0]
            if i+1 < params_target.shape[1]:
                col_to_plot = Ydrf.columns[i+1]
                Y = Ydrf[col_to_plot]
                Y_values = np.linspace(min(Y), max(Y), 1000)
                posterior_kde = gaussian_kde(Y, weights=W)
                posterior_density = posterior_kde(Y_values)
                #PLOT POSTERIOR
                limit_left = int(-7 + j)
                limit_right = int(-5 + j)
                ax.plot(Y_values, posterior_density, label=fr'$\pi^T\left(\theta_{{i}}|\mathbf{{X}}^{{obs}}_{{mtDNA}}, \theta_1 \in [10^{{{limit_left}}},10^{{{limit_right}}}]\right)_{{2\leq i \leq 13}}$', color=cmap(j / (num_curves-1)))
                ax.fill_between(Y_values, posterior_density, color=cmap(j / (num_curves-1)), alpha=0.05)
                cond_exp_est = np.sum(Y.to_numpy() * W)
                # PLOT COND EXP
                ax.axvline(cond_exp_est, color=cmap(j / (num_curves-1)), linestyle='--', 
                           label=fr'$\mathbb{{E}}_{{\pi^T}}[\theta_{{i}}|\mathbf{{X}}^{{obs}}_{{mtDNA}},\theta_1\in [10^{{{limit_left}}},10^{{{limit_right}}}]]_{{2\leq i \leq 13}}$')
                ax.set_title(fr'$\pi_{{\mathrm{{index}}}}$'.replace('index', str(col_to_plot[-2:])), fontsize = 25)
                ax.set_xlabel('Values', fontsize = 20)
                ax.set_ylabel('Density', fontsize = 20)
    
    # Add a global legend
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', fontsize=25, bbox_to_anchor=(0.5, 0), ncol=3)
    os.makedirs(os.path.dirname(f'Experiments_data/general_figures/'), exist_ok=True)
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    exp_str = ""
    for results_folder in results_folders:
        exp_str += results_folder + "_"
    exp_str = exp_str[:-1]
    plt.savefig(f'Experiments_data/general_figures/ABC_DRF_{idx}_comparison_prior_{nb_iter}_{exp_str}.png')
    print(f'{nb_iter} : 1D_posteriors for {idx} saved in Experiments_data/general_figures/ABC_DRF_{idx}_comparison_prior_{nb_iter}_{exp_str}.png')
    plt.close()


def mse(nb_iter, idx, results_folder, subfolder):
    params_target = pd.read_csv(results_folder+f'params_targets_{idx}.csv')
    Y_posterior = pd.read_csv(results_folder+subfolder+f'Ydrf_{idx}_prior_{nb_iter}.csv') if nb_iter>1 else pd.read_csv(results_folder+subfolder+f'Ydrf_{idx}_prior_{nb_iter}.csv')
    mse =  (Y_posterior.mean(axis=0) - params_target)**2
    result_mse_path = results_folder+f'mse_{nb_iter}.csv'
    if os.path.exists(result_mse_path):
        results_df = pd.read_csv(result_mse_path)
        results_df = pd.concat([results_df, mse], axis=0)
    else:
        results_df = mse
    results_df.to_csv(result_mse_path, index=False)
    return results_df

def run_abc_smc_rf(nb_samples, nb_iter, idx_start=0, idx_end=4, real=False, full_sequence=False, 
                   plot_1D=True, plot_2D=False,root_distribution=[0.25,0.25,0.25,0.25], null_diag=True,
                   mr_min=1e-7, mr_max=1e-5, data_folder=None, model_folder=None, results_folder=None, 
                   points_to_infere_path=None):
    subfolder = f'prior_{nb_iter}/'
    # Simulate and save the data
    ds = DataSimulator(nb_samples=nb_samples, sample_size=100, population_size=1000, sequence_length=4000, 
                    len_seq_final=4000, mr_min = mr_min, mr_max = mr_max, root_distribution=root_distribution, 
                    null_diag=null_diag, data_folder=data_folder,full_seq=full_sequence, real=real)
    for idx in range(idx_start,idx_end+1):
        ds.simulate_and_save(nb_iter, idx)

    # Generate the summary statistics
    for idx in range(idx_start,idx_end+1):
        gss = GenerateSummaryStatistics(nb_iter, idx, data_folder, results_folder, model_folder=model_folder)
        gss.run()

    # Generate Xdrf, Ydrf
    params = [(points_to_infere_path, results_folder + f'parameters_and_SS_{idx}_prior_{nb_iter}.pt', 
               nb_iter, idx, idx_start, results_folder) for idx in range(idx_start, idx_end+1)]
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            for _ in tqdm(pool.istarmap(create_csv_for_DRF, params),total=len(params), desc='Generate Ydrf'):
                pass

    # Run the DRF 
    robjects.r['source']('abc_drf.r')
    generate_DRF_weights = robjects.r['generate_DRF_weights']
    for idx in range(idx_start,idx_end+1):
        generate_DRF_weights(nb_iter, idx, results_folder)

    #Plot the posterior distribution
    #1D
    if plot_1D:
        params = [( nb_iter, idx, results_folder) for idx in range(idx_start,idx_end+1)]
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                for _ in tqdm(pool.istarmap(plot_1D_posteriors, params),total=len(params), desc='Plotting 1D posteriors'):
                    pass
    #2D
    if plot_2D:
        params = [(i,j, nb_iter, idx, results_folder, subfolder) 
                  for idx, i, j in itertools.product(range(idx_start,idx_end+1), range(1,13), range(1,13))]
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            for _ in tqdm(pool.istarmap(plot_2D_posterior, params), total = len(params), desc='Plotting 2D posteriors'):
                pass
        params = [(idx, results_folder, subfolder) for idx in range(idx_start,idx_end+1)]
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            for _ in tqdm(pool.istarmap(merge_2D_posteriors, params),total=len(params), desc='Merging 2D posteriors'):
                pass
    
    #mse
    for idx in range(idx_start,idx_end+1):
        mse(nb_iter, idx, results_folder, subfolder)

    #SMC Prior drawing
    params = [(mr_min, mr_max, nb_iter, idx, results_folder, subfolder, data_folder, nb_samples) for idx in range(idx_start,idx_end+1)]
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            for _ in tqdm(pool.istarmap(posterior_sampling, params),total=len(params), desc='Posterior Sampling'):
                pass