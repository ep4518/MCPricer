# -*- coding: utf-8 -*-
# Created on Thu Jan 23 2025
# Author: epeterson

from abc import ABC, abstractmethod
from BSModel01 import BSModel, SamplePath
from numpy import exp, sqrt
from tqdm import tqdm

from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np

class PathDepOption(ABC):
    
    def __init__(self, T: float, K: float, m: int):
        self.T = T
        self.K = K
        self.m = m
        self.Paths = []
        self.PricingError = None
        self.Price = None
        self.delta, self.gamma, self.theta, self.rho, self.vega = None, None, None, None, None
    
    def Rescale(self, S: SamplePath, x: float):
        return [x * y for y in S] 

    def PriceByMC(self, Model: BSModel, N: float, epsilon: float):
        H: float = 0.0
        Hsq: float = 0.0
        HepsPos: float = 0.0
        HepsNeg: float = 0.0
        S: SamplePath = [0.0] * self.m
        
        for i in tqdm(range(N)):
            
            Model.GenerateSamplePath(self.T, self.m, S)
            # pprint(S)
            
            if i < 1000:
                self.Paths.append(S.copy())
                
            H = (i * H + self.Payoff(S)) / (i + 1.0)
            Hsq = (i * Hsq + pow(self.Payoff(S), 2.0)) / (i + 1.0)
            self.Rescale(S, 1.0 + epsilon)
            HepsPos = (i * HepsPos + self.Payoff(S)) / (i + 1.0)
            # self.Rescale(S, (1.0 - epsilon) / (1 + epsilon))
            # HepsNeg = (i * HepsNeg + self.Payoff(S)) / (i + 1.0)
            # pprint(S)
            # break
        
        # print(HepsPos, H) 
        self.Price = exp(-Model.r * self.T) * H
        self.PricingError = exp(-Model.r * self.T) * sqrt(Hsq - H * H) / sqrt(N - 1.0)
        # self.delta = exp(-Model.r * self.T) * (HepsPos-H) / (Model.S0 * epsilon)
        # self.gamma = exp(-Model.r * self.T) * (HepsPos - 2 * H + HepsNeg) / (Model.S0 * epsilon)
        
        return self.Price
    
    
    def plotMC(self):
        
        assert self.Paths is not None, "Price by Monte Carlo first."
        
        fig, ax = plt.subplots(figsize=(14, 7), layout='constrained')
        # ax.set_xlabel('X-axis Label')
        # ax.set_ylabel('Y-axis Label')
        x = np.arange(self.m)
        for _, S in enumerate(self.Paths):
            ax.plot(x, S)
        
        return fig, ax
    
    @abstractmethod
    def Payoff(self, S: SamplePath):
        pass
    
    
class ArthmAsianCall(PathDepOption):
    
    def __init__(self, T: float, K: float, m: int):
        super().__init__(T, K, m) 
    
    def Payoff(self, S):
        avg = 0.0
        for k in range(self.m):
            avg = (k * avg + S[k]) / (k + 1)
        return max(0.0, avg - self.K)    
    
    
class CallUpOut(PathDepOption):
    
    def __init__(self, T: float, K: float, B: float, R: float, m: int):
        super().__init__(T, K, m)
        self.Barrier = B
        self.Rebate = R
        
    def plotMC(self):
        fig, ax = super().plotMC()
        ax.axhline(self.Barrier, color='red')
        ax.axhline(self.Rebate + 1, color='green')
        return fig, ax
        
    def Payoff(self, S):
        indicKI = max(S) > self.Barrier
        return self.Rebate if indicKI else max(S[-1] - self.K, 0)


    
    