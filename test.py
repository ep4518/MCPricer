# -*- coding: utf-8 -*-
# Created on Thu Jan 23 2025
# Author: epeterson

from BSModel01 import BSModel
from PathDepOption01 import EuropeanCall

K: float = 100
S0: float = 100
Nbpaths: int = 30000

r=0.03
sigma=0.2
T=0.5
m = 30
epsilon=0.001
model1: BSModel = BSModel(S0, r, sigma)

option: EuropeanCall = EuropeanCall(T=T, K=K, m=m)
priceBS = option.PriceByMC(Engine=model1, N=Nbpaths, epsilon=epsilon)
print(f'Call Up and Out Price = {priceBS}')
print(f'Option Delta: {option.Delta}')
print(f'Option Gamma: {option.Gamma}')
print(f'Option Vega: {option.Vega} THIS IS WRONG')
print(f'Option Theta: {option.Theta} THIS IS WRONG')
print(f'Option Rho: {option.Rho} THIS IS WRONG')
print(f'Price Error = {option.PricingError}')