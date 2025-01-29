# -*- coding: utf-8 -*-
# Created on Tue Jan 28 2025
# Author: epeterson

from PathDepOption05 import ArthmAsianCall, EuropeanCall
from BSModel02 import BSModel, Gauss, Vector, SamplePath, Matrix
import numpy as np
import matplotlib.pyplot as plt

d: int = 3
S0: Vector = np.array([40.0, 60.0, 100.0])
    
r: float = 0.03
C: Matrix = np.array([
                [0.1, -0.1, 0.0],
                [-0.1, 0.2, 0.0],
                [0.0,  0.0, 0.3]
            ])

model: BSModel = BSModel(S0, r, C)
T: float = 1.0 / 12.0
K: float = 200.0
m: int = 30
Option: ArthmAsianCall = ArthmAsianCall(T=T, m=m, K=K)
N: int = 10000

print(f'Arithmetic Basket Call Price = {Option.PriceByMC(model, N)}')
 
r: float = 0.042
weights = np.ones(d) / d
rho = np.array([
    [1, 0.6242, 0.5953],
    [0.6242, 1, 0.4998],
    [0.5953, 0.4998, 1]
])
sigma = np.array([0.2429, 0.5082, 0.5670])

model: BSModel = BSModel(S0=np.ones(d), r=r, sigma=sigma, rho=rho)
T: float = 1.0
K: float = 1.0
m: int = 252 
gearing: float = 1.5
Option: EuropeanCall = EuropeanCall(K, T, m, Gearing=gearing)
N: int = 10000
print(f'European Basket Call Price = {Option.PriceByMC(model, N, weights=weights)}')
