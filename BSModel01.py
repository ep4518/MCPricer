# -*- coding: utf-8 -*-
# Created on Thu Jan 23 2025
# Author: epeterson

from typing import List
from numpy import arctan, sqrt, log, cos, exp
from numpy.random import exponential, normal
from random import randint
from abc import ABC, abstractmethod

import logging

logger = logging.getLogger(__name__)

Vector = List[float]
SamplePath = Vector
PI = 4.0 * arctan(1.0)

RAND_MAX = 2147483647


def Gauss():
    U1 = (randint(0, RAND_MAX) + 1.0) / (RAND_MAX + 1.0)
    U2 = (randint(0, RAND_MAX) + 1.0) / (RAND_MAX + 1.0)
    return sqrt(-2.0 * log(U1)) * cos(2.0 * PI * U2)


class Model(ABC):
    def __init__(self, S0: float, r: float, sigma: float):
        self.S0 = S0
        self.r = r
        self.sigma = sigma

    @abstractmethod
    def GenerateSamplePath(self, T: float, m: int, S: SamplePath):
        pass


class BSModel(Model):
    def __init__(self, S0: float, r: float, sigma: float):
        super().__init__(S0=S0, r=r, sigma=sigma)

    def __repr__(self):
        return f"Black & Scholes Engine {round(self.r * 100, 2)} {round(self.sigma * 100, 2)}%"

    def GenerateSamplePath(self, T: float, m: int, S: SamplePath):
        St = self.S0
        for i in range(m):
            S[i] = St * exp(
                (self.r - self.sigma * self.sigma * 0.5) * (T / m)
                + self.sigma * sqrt(T / m) * Gauss()
            )
            St = S[i]


# https://quant-next.com/the-merton-jump-diffusion-model/


class MJDModel(Model):
    def __init__(
        self,
        S0,
        r: float,
        sigma: float,
        mu: float,
        lambda_: float,
        mean: float,
        delta: float,
    ):
        super().__init__(S0=S0, r=r, sigma=sigma)
        self.mu = mu
        self.lambda_ = lambda_
        self.mean = mean
        self.delta = delta

    def normal_compound_poisson_process(self, T):

        t = 0
        jumps = 0
        event_values = [0]
        while t < T:
            t = t + exponential(1 / self.lambda_)
            jumps = jumps + normal(
                self.mean, self.delta
            )  # m + Gauss() * self.delta     # N(m, delta)
            if t < T:
                event_values.append(jumps)

        return event_values

    def GenerateSamplePath(self, T: float, m: int, S: SamplePath):
        dt = T / m
        St = self.S0
        k = exp(self.mean + 0.5 * self.delta**2) - 1

        for i in range(m):
            dW = sqrt(dt) * Gauss()
            jumps = sum(self.normal_compound_poisson_process(dt))
            S[i] = St * exp(
                (self.mu - 0.5 * self.sigma**2 - self.lambda_ * k) * dt
                + self.sigma * dW
                + jumps
            )
            St = S[i]
