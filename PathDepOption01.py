# -*- coding: utf-8 -*-
# Created on Thu Jan 23 2025
# Author: epeterson

from abc import ABC, abstractmethod
from BSModel01 import Model, SamplePath, Vector
from numpy import exp, sqrt, log
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np

from threading import Thread
from multiprocessing import process
from numba import jit

import logging
logger = logging.getLogger('detailedLogger')
logger.info('File name of pdo %', __name__)

class PathDepOption(ABC):
    
    def __init__(self, T: float, K: float, m: int):
        self.T = T
        self.K = K
        self.m = m
        self.Paths = []
        self.PricingError = None
        self.Price = None
        self.Delta, self.Gamma, self.Theta, self.Rho, self.Vega = None, None, None, None, None
    
    # def Rescale(self, S: SamplePath, x: float):
    #     for i in range(len(S)):
    #         S[i] = x * S[i]
            
    def Rescale(self, S: SamplePath, Z: Vector, S0: float, r: float, sigma: float, dt: float):
        S[0] = S0 * exp((r - sigma * sigma / 2.0) * dt + sigma * sqrt(dt) * Z[0])
        for j in range(1, self.m):
            S[j] = S[j-1] * exp((r - sigma * sigma / 2.0) * dt + sigma * sqrt(dt) * Z[j])
            
    
    def GetZ(self, Z: Vector, S: SamplePath, S0: float, r: float, sigma: float, dt: float):
        # Backing out Z_t from:
        # S_t = S_{t-1} \exp\left(\left(r - \frac{\sigma^2}{2}\right)\Delta t + \sigma \sqrt{\Delta t} Z_t\right)
        Z[0] = (log(S[0] / S0) - (r - sigma * sigma / 2.0) * dt) / (sigma * sqrt(dt))
        for j in range(1, self.m):
            Z[j] = (log(S[j] / S[j-1]) - (r - sigma * sigma / 2.0) * dt) / (sigma * sqrt(dt))
            
        logging.debug(f'Z: {[float(a) for a in Z]}')
        
         
    def PriceByMC(self, Engine: Model, N: float, epsilon: float):
        H: float = 0.0
        Hsq: float = 0.0
        Hdelta: float = 0.0
        Hgamma: float = 0.0
        Hrho: float = 0.0
        Hvega: float = 0.0
        Htheta: float = 0.0
        S: SamplePath = np.random.rand(self.m).tolist() 
        Z: Vector = [0.0] * self.m
        dt: float = self.T / self.m
        
        for i in tqdm(range(N)):
            
            Engine.GenerateSamplePath(self.T, self.m, S)
            
            if i < 1000:
                self.Paths.append(S.copy())
                
            H = (i * H + self.Payoff(S)) / (i + 1.0)
            Hsq = (i * Hsq + pow(self.Payoff(S), 2.0)) / (i + 1.0)
            self.GetZ(Z, S, Engine.S0, Engine.r, Engine.sigma, dt)
            
            self.Rescale(S, Z, Engine.S0 * (1.0 + epsilon), Engine.r, Engine.sigma, dt)
            Hdelta = (i * Hdelta + self.Payoff(S)) / (i + 1.0)
            self.Rescale(S, Z, Engine.S0 * (1.0 - epsilon), Engine.r, Engine.sigma, dt)
            Hgamma = (i * Hgamma + self.Payoff(S)) / (i + 1.0)
            self.Rescale(S, Z, Engine.S0, Engine.r * (1.0 + epsilon), Engine.sigma, dt)
            Hrho = (i * Hrho + self.Payoff(S)) / (i + 1.0)
            self.Rescale(S, Z, Engine.S0, Engine.r, Engine.sigma * (1.0 + epsilon), dt)
            Hvega = (i * Hvega + self.Payoff(S)) / (i + 1.0)
            self.Rescale(S, Z, Engine.S0, Engine.r, Engine.sigma, dt * (1.0 + epsilon))
            Htheta = (i * Htheta + self.Payoff(S)) / (i + 1.0)
            
            # self.Rescale(S, 1.0 + epsilon)
            # Hdelta = (i * Hdelta + self.Payoff(S)) / (i + 1.0)
            # self.Rescale(S, (1.0 - epsilon) / (1 + epsilon))
            # Hgamma = (i * Hgamma + self.Payoff(S)) / (i + 1.0)
            # self.Rescale(S, Engine.r * (1.0 + epsilon) / (1.0 - epsilon))
            # Hrho = (i * Hrho + self.Payoff(S)) / (i + 1.0)
            # self.Rescale(S, Engine.sigma / Engine.r)
            # Hvega = (i * Hvega + self.Payoff(S)) / (i + 1.0)
            # self.Rescale(S, dt / Engine.sigma)
            # Htheta = (i * Htheta + self.Payoff(S)) / (i + 1.0)

        
        self.Price = exp(-Engine.r * self.T) * H
        self.PricingError = exp(-Engine.r * self.T) * sqrt(Hsq - H * H) / sqrt(N - 1.0)
        self.Delta = exp(-Engine.r * self.T) * (Hdelta-H) / (Engine.S0 * epsilon)
        self.Gamma = exp(-Engine.r * self.T) * (Hdelta - 2 * H + Hgamma) / pow(Engine.S0 * epsilon, 2)
        self.Vega  = exp(-Engine.r * self.T) * (Hvega - H) / (Engine.sigma * epsilon);
        self.Theta =-(exp(-Engine.r * self.T * (1.0 + epsilon)) * Htheta-exp(-Engine.r * self.T) * H) / (self.T * epsilon);
        self.Rho   = (exp(-Engine.r * (1.0 + epsilon) * self.T) * Hrho - exp(-Engine.r * self.T) * H) / (Engine.r * epsilon);
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
    
    def apply(self, *meta_scripts):
        from Meta01 import MetaScriptRegistry, MetaScript
        for meta_script in meta_scripts:
            if isinstance(meta_script, tuple):  # Tuple for script with arguments
                name, args, kwargs = meta_script[0], meta_script[1] or (), meta_script[2] or {}
                meta_script_class = MetaScriptRegistry.get(name)
                meta_script_instance = meta_script_class(*args, **kwargs)
            elif isinstance(meta_script, str):  # Script by name
                meta_script_class = MetaScriptRegistry.get(meta_script)
                meta_script_instance = meta_script_class()
            elif isinstance(meta_script, MetaScript):  # Script by class
                meta_script_instance = meta_script()
            else:
                raise ValueError("Argument is not of type MetaScript")
            
            return meta_script_instance.run_meta(self)
    
    @abstractmethod
    def Payoff(self, S: SamplePath):
        pass
    
class EuropeanCall(PathDepOption):
    
    def __init__(self, T: float, K: float, m: int):
        super().__init__(T, K, m) 
    
    def Payoff(self, S):
        return max(0.0, S[-1] - self.K)    
    
    
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


    
    