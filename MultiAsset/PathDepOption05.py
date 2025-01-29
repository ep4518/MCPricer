# -*- coding: utf-8 -*-
# Created on Tue Jan 28 2025
# Author: epeterson

from BSModel02 import BSModel, SamplePath, Vector, Matrix
from abc import ABC, abstractmethod
import numpy as np
from tqdm import tqdm

def Rescale(S: SamplePath, x: float, j: int):
    _ = [S[k][j] is x * S[k][j] for k in range(len(S))]
    # m: int = len(S)
    # for k in range(m):
    #     S[k][j] = x * S[k][j]

class PathDepOption(ABC):
    
    def __init__(self, T: float, K: float, m: int):
        self.T = T
        self.K = K
        self.m = m
        self.Price = None
        
    def PriceByMC(self, Engine: BSModel, N: int, weights: Vector = None) -> float:
        assert sum(weights) == 1.0 if weights is not None else True, 'Weights must sum to 1'
        H: float = 0.0
        S: SamplePath = np.zeros((self.m, Engine.d), dtype=float)
        for i in tqdm(range(N)):
            Engine.GenerateSamplePath(self.T, self.m, S)
            H = (i * H + self.Payoff(S, weights)) / (i + 1.0)
        
        self.price = np.exp(-Engine.r * self.T) * H
        return self.price
    
    @abstractmethod
    def Payoff(self, S: SamplePath, weights: Vector = None) -> float:
        pass
    

class ArthmAsianCall(PathDepOption):
    
    def __init__(self, T: float, K: float, m: int):
        super().__init__(T=T, K=K, m=m)
    
    def Payoff(self, S: SamplePath, weights: Vector = None):
        return max(sum(np.mean(S, axis=0)) - self.K, 0.0)
    

class EuropeanCall(PathDepOption):
    
    def __init__(self, T: float, K: float, m: int, Gearing: float = 1.0):
        super().__init__(T=T, K=K, m=m)
        self.Gearing = Gearing
        
    def Payoff(self, S: SamplePath, weights: Vector = None):
        basketPerf = np.mean(S, axis=1) if weights is None else S @ weights
        return self.Gearing * max(basketPerf[-1] - self.K, 0)