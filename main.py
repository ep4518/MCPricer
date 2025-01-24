# -*- coding: utf-8 -*-
# Created on Thu Jan 23 2025
# Author: epeterson

from BSModel01 import BSModel, MJDModel
from PathDepOption01 import BSModel, ArthmAsianCall, CallUpOut
import matplotlib.pyplot as plt

S0: float = 1.0
r: float = 0.042
sigma: float = 0.4826
mu: float = 0.05         # Drift / Expected return y-y
lambda_: int = 10       # Poisson Intensity
delta: float = 0.1      # Standard deviation of jumps
Nbpaths: int = 10000

epsilon: float = 0.01

model1: BSModel = BSModel(S0, r, sigma)
model2: MJDModel = MJDModel(S0=S0, r=r, sigma=sigma, mu=mu, lambda_=10, mean=0, delta=.1)

T: float = 1
K: float = 1
m: int = 252

option: CallUpOut = CallUpOut(T=T, K=K, B=1.80, R=0.05, m=m)
priceBS = option.PriceByMC(Model=model1, N=Nbpaths, epsilon=epsilon)
fig, ax = option.plotMC()
fig.savefig('BS.png')
print(f'Call Up and Out Price = {priceBS}')
# priceMJD = option.PriceByMC(Model=model2, N=Nbpaths)
# fig, ax = option.plotMC()
# fig.savefig('MJD.png')
# print(f'Call Up and Out Price = {priceMJD}')