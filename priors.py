#Functions to generate priors
import numpy as np
import torch

def generate_gaussian(mu=[0],sigma=[1],nb_points=1_000):
    res = np.zeros((len(mu),nb_points))
    for i in range(len(mu)):
        res[i] = np.random.normal(mu[i], sigma[i], nb_points)
    return torch.from_numpy(res)

def generate_uniform(a=[0],b=[1],nb_points=1_000):
    res = np.zeros((len(a),nb_points))
    for i in range(len(a)):
        res[i] = np.random.uniform(a[i], b[i], nb_points)
    return torch.from_numpy(res)

def generate_discrete_uniform(points=[list(range(10))], nb_points=1_000):
    res = np.zeros((len(points),nb_points))
    for i in range(len(points)):
        res[i] = np.random.choice(points[i], nb_points)
    return torch.from_numpy(res)

def generate_uniform_transition_matrix(n=4, nb_points=1_000):
    transition_matrix = np.random.rand(nb_points, n, n)
    for matrix in transition_matrix:
        np.fill_diagonal(matrix, 0)
    transition_matrix = transition_matrix / transition_matrix.sum(axis=2, keepdims=True)
    return torch.from_numpy(transition_matrix)

def generate_uniform_root_distribution(n=4, nb_points=1_000):
    root_distribution = np.random.rand(nb_points, n)
    root_distribution = root_distribution / root_distribution.sum(axis=1, keepdims=True)
    return torch.from_numpy(root_distribution)

def generate_dirichlet_transition_matrix(n=4, nb_points=1_000, null_diag=True):
    transition_matrix = np.random.dirichlet(np.ones(n), size=(nb_points,n))
    if null_diag:
        for i in range(transition_matrix.shape[0]):
            np.fill_diagonal(transition_matrix[i], 0)
        # Normalize each row to sum to 1
        transition_matrix = transition_matrix / np.sum(transition_matrix, axis=2, keepdims=True)
    return torch.from_numpy(transition_matrix)

def generate_prior(type="uniform",mu=[0],sigma=[1],a=[0],b=[1],n=4,null_diag=True, points=list(range(10)),nb_points=1_000):
    if type == "gaussian":
        return generate_gaussian(mu,sigma,nb_points)
    elif type == "uniform":
        return generate_uniform(a,b,nb_points)
    elif type == "discrete_uniform":
        return generate_discrete_uniform(points,nb_points)
    elif type == "uniform_transition_matrix":
        return generate_uniform_transition_matrix(n=n,nb_points=nb_points)
    elif type == "uniform_root_distribution":
        return generate_uniform_root_distribution(n=n,nb_points=nb_points)
    elif type == "dirichlet_transition_matrix":
        return generate_dirichlet_transition_matrix(n=n,nb_points=nb_points, null_diag=null_diag)
    else:
        raise ValueError("Invalid type for prior")