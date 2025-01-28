# -*- coding: utf-8 -*-
# Created on Fri Jan 24 2025
# Author: epeterson

from PathDepOption01 import PathDepOption
from BSModel01 import SamplePath, Model
from abc import ABC, abstractmethod
from tqdm import tqdm

from numpy import exp

import logging
logger = logging.getLogger(__name__)


class MetaScript(ABC):
    
    @abstractmethod
    def run_meta(self, obj: PathDepOption):
        pass

class GreekMetaScript(MetaScript):
    def __init__(self, Engine: Model, N: float, epsilon: float):
        self.Engine = Engine
        self.N = N
        self.epsilon = epsilon

    def monte_carlo_simulation(self, obj):
        """Run Monte Carlo simulation and return base and perturbed payoffs."""
        H = Hsq = Heps = 0.0
        S = [0.0] * obj.m

        try:
            for i in tqdm(range(self.N)):
                self.Engine.GenerateSamplePath(obj.T, obj.m, S)

                payoff = obj.Payoff(S)
                H = (i * H + payoff) / (i + 1.0)
                Hsq = (i * Hsq + payoff ** 2) / (i + 1.0)
                self.param_perturbate(obj)
                modified_payoff = obj.Payoff(S)
                Heps = (i * Heps + modified_payoff) / (i + 1.0)
                self.restore_parameters(obj)
                print(payoff, modified_payoff)
                break

        finally:
            self.restore_parameters(obj)

        print(H, Hsq, Heps)
        return H, Hsq, Heps

    @abstractmethod
    def param_perturbate(self, obj):
        pass

    @abstractmethod
    def restore_parameters(self, obj):
        pass

    @abstractmethod
    def approximate(self, H, Hsq, Heps, obj):
        pass

    def run_meta(self, obj):
        H, Hsq, Heps = self.monte_carlo_simulation(obj)
        return self.approximate(H, Hsq, Heps, obj)


class Vega(GreekMetaScript):
    def param_perturbate(self, obj):
        self.Engine.sigma += self.epsilon

    def restore_parameters(self, obj):
        self.Engine.sigma -= self.epsilon

    def approximate(self, H, Hsq, Heps, obj):
        print(exp(-self.Engine.r * obj.T) * (Heps - H) / self.epsilon, self.Engine.r, obj.T, Heps, H, self.epsilon)
        return exp(-self.Engine.r * obj.T) * (Heps - H) / self.epsilon


class Theta(GreekMetaScript):
    def param_perturbate(self, obj):
        obj.T -= self.epsilon

    def restore_parameters(self, obj):
        obj.T += self.epsilon
        
    def approximate(self, H, Hsq, Heps, obj):
        print(exp(-self.Engine.r * obj.T) * (Heps - H) / self.epsilon, self.Engine.r, obj.T, Heps, H, self.epsilon)
        return exp(-self.Engine.r * obj.T) * (Heps - H) / self.epsilon

class Rho(GreekMetaScript):
    def param_perturbate(self, obj):
        self.Engine.r -= self.epsilon

    def restore_parameters(self, obj):
        self.Engine.r += self.epsilon

    def approximate(self, H, Hsq, Heps, obj):
        return exp(-self.Engine.r * obj.T) * (Heps - H) / self.epsilon

        
# class Delta(GreekMetaScript):
    
#     def __init__(self, Engine: Model, N: float, epsilon: float):
#         super().__init__(Engine=Engine, N=N, epsilon=epsilon)
        
#     def path_pertubate(self, S, obj: PathDepOption):
#         obj.Rescale(S, )
    
#     def approximate(self, H: float, Heps: float, T: float):
#         return exp(-self.Engine.r * T) * (Heps - H) / (self.Engine.S0 * self.epsilon)
    
# class Gamma(GreekMetaScript):
    
#     def __init__(self, Engine: Model, N: float, epsilon: float):
#         super().__init__(Engine=Engine, N=N, epsilon=epsilon)
        
#     def path_pertubate(self, obj):
#         return super().path_pertubate(obj)
    
#     def approximate(self, H: float, HepsPos: float, HepsNeg: float, T: float):
#         return exp(-self.Engine.r * T) * (HepsPos - 2 * H + HepsNeg) / pow(self.Engine.S0 * self.epsilon, 2)
    


class MetaScriptRegistry:
    
    _registry = {}
    
    @classmethod
    def register(cls, name: str, meta_script_class: MetaScript):
        cls._registry[name] = meta_script_class
    
    @classmethod
    def get(cls, name: str):
        # print(f"Available scripts: {cls._registry.keys()}")
        if name not in cls._registry:
            raise ValueError(f"Meta Script {name} is not registered")
        return cls._registry[name]

