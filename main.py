# -*- coding: utf-8 -*-
# Created on Thu Jan 23 2025
# Author: epeterson

from BSModel01 import BSModel, MJDModel
from PathDepOption01 import ArthmAsianCall, EuropeanCall, CallUpOut
from Meta01 import Vega, Theta, Rho, MetaScriptRegistry
import matplotlib.pyplot as plt
import os
import logging

# import logging.config
# logging.config.fileConfig('logging.conf')
logging.basicConfig(filename='main.log', filemode='w', encoding='utf-8', level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.info('Started')

scale = 100

T: float = 1 
K: float = 1.0 * scale
m: int = int(252 * T)

S0: float = 1.0 * scale
r: float = 0.042
sigma: float = 0.4825
mu: float = 0.05            # Drift / Expected return y-y
lambda_: int = 10           # Poisson Intensity
poisson_mean: float = 0
delta: float = 0.1          # Standard deviation of jumps
Nbpaths: int = 10000
epsilon: float = 0.001

B = 1.8 * scale
R = .05 * scale

model1: BSModel = BSModel(S0, r, sigma)
model2: MJDModel = MJDModel(S0=S0, r=r, sigma=sigma, mu=mu, lambda_=lambda_, mean=poisson_mean, delta=delta)

MetaScriptRegistry.register('vega', Vega)
MetaScriptRegistry.register('theta', Theta)

option: CallUpOut = CallUpOut(T=T, K=K, B=180, R=5, m=m)
priceBS = option.PriceByMC(Engine=model1, N=Nbpaths, epsilon=epsilon)
fig, ax = option.plotMC()
fig.savefig('BS.png')
print(f'Call Up and Out Price = {priceBS}')
print(f'Option Delta: {option.Delta}')
print(f'Option Gamma: {option.Gamma}')
print(f'Option Vega: {option.Vega}')
print(f'Option Theta: {option.Theta}')
print(f'Option Rho: {option.Rho}')
print(f'Price Error = {option.PricingError}')


# option.apply(Vega(model1, Nbpaths, epsilon))
# vega = option.apply(('vega', (), {"Engine":model1, 
#                                   "N": Nbpaths, 
#                                   "epsilon": epsilon}))
# print(f'Option Vega: {vega}')
# option.apply(
#     "vega",
#     "theta",
#     Rho
# )

# priceMJD = option.PriceByMC(Engine=model2, N=Nbpaths, epsilon=epsilon)
# fig, ax = option.plotMC()
# fig.savefig('MJD.png')
# print(f'Call Up and Out Price = {priceMJD}')
