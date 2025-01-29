# -*- coding: utf-8 -*-
# Created on Tue Jan 28 2025
# Author: epeterson

import numpy as np
from typing import List
from random import randint
from typing import Optional

PI = 4.0 * np.arctan(1.0)
RAND_MAX = 2147483647

Vector = np.ndarray
SamplePath = np.ndarray
Matrix = np.ndarray

def Gauss(d: int = None):
    if d is None:
        U1 = (randint(0, RAND_MAX) + 1.0) / (RAND_MAX + 1.0)
        U2 = (randint(0, RAND_MAX) + 1.0) / (RAND_MAX + 1.0)
        return np.sqrt(-2.0 * np.log(U1)) * np.cos(2.0 * PI * U2)
    
    Z: Vector = np.ndarray((d,))
    for i in range(d):
        Z[i] = Gauss()
        
    return Z

class BSModel:
    
    def __init__(self, S0: np.ndarray, r: float, 
                 C: Optional[np.ndarray] = None, 
                 sigma: Optional[np.ndarray] = None, 
                 rho: Optional[np.ndarray] = None):
        """
        Initialize the Black-Scholes model.

        Parameters:
        - S0: Vector of initial asset prices.
        - r: Risk-free interest rate.
        - C: Cholesky decomposition matrix (optional).
        - sigma: Volatility vector (optional, used with rho).
        - rho: Correlation matrix (optional, used with sigma).
        """
        self.S0 = S0
        self.r = r
        self.d = len(S0)

        if C is not None:
            self.C = C
            self.sigma = np.sqrt(np.sum(C**2, axis=1))
            self.Cov = C @ C.T
        elif sigma is not None and rho is not None:
            self.sigma = sigma
            self.Cov = rho * np.outer(sigma, sigma)
            self.C = np.linalg.cholesky(self.Cov)
        else:
            raise ValueError("Either 'C' or both 'sigma' and 'rho' must be provided.")

    def GenerateSamplePath(self, T: float, m: int, S: SamplePath):
        St: Vector = self.S0
        d: int = len(self.S0)
        for i in range(m):
            S[i] = St * np.exp((T / m) * (self.r + (-0.5) * self.sigma * self.sigma) + np.sqrt(T / m) * (self.C @ Gauss(d)))
            St = S[i]