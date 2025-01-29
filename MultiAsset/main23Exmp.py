# -*- coding: utf-8 -*-
# Created on Tue Jan 28 2025
# Author: epeterson

from PathDepOption05 import ArthmAsianCall, EuropeanCall
from BSModel02 import BSModel, Gauss, Vector, SamplePath, Matrix
import numpy as np
import matplotlib.pyplot as plt

"""
main32.py, BSModel02.py, PathDepOption05.py
Multi-Asset MC Pricer. 
Can be initialised with non-normalised spots however, 
for real pricing data, with Cholesky decomp of Cov matrix,
use S0 = np.ones(d), K = 1.0 and not option.price() / sum(S0)

"""

d: int = 3

# S0: Vector = np.array([229.86, 118.42, 397.15])
weights = np.ones(d) / d
 
r: float = 0.042
rho = np.array([
    [1, 0.6242, 0.5953],
    [0.6242, 1, 0.4998],
    [0.5953, 0.4998, 1]
])

sigma = np.array([0.2429, 0.5082, 0.5670])

# Cov_ij = rho_ij * sigma_i * sigma_j
Cov = rho * np.outer(sigma, sigma)

# Cholesky decomposition of Cov == Correlation structure of Weiner dynamics
# Z(t) = C ⋅ W(t), where W(t) = [W1(t), W2(t), W3(t)]^T
C = np.linalg.cholesky(Cov)

print(f'Sigma == sqrt(sum(C**2, axis=1)): {sigma == np.sqrt(np.sum(C**2, axis=1))}')

print("Cholesky decomposition (C):")
print(C)

# Verify the result: C @ C.T should equal the original covariance matrix
print("Reconstructed covariance matrix:")
print(C @ C.T)


model: BSModel = BSModel(np.ones(d), r, sigma=sigma, rho=rho)
T: float = 1.0
K: float = 1.0
m: int = 252 
gearing: float = 1.5
Option: EuropeanCall = EuropeanCall(K, T, m, Gearing=gearing)
N: int = 10000
price = Option.PriceByMC(model, N, weights=weights)

print(f'European Basket Call Price = {price}')

S : SamplePath = np.zeros((m, d))
model.GenerateSamplePath(T, m, S)
