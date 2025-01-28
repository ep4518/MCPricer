# -*- coding: utf-8 -*-
# Created on Sat Jan 25 2025
# Author: epeterson

from scipy.stats import norm
from numpy import sqrt, log, exp

N = norm.cdf
n = norm.pdf

sigma = 0.2
T = 0.5
K = 100
r = 0.03
S = 100


class BlackScholesCalc:
    def __init__(self, T, sigma, r, K=1):
        self.T = T
        self.r = r
        self.sigma = sigma
        self.K = K

    def d1(self, S, t):
        return (log(S / self.K) + (self.r + 0.5 * self.sigma**2) * (self.T - t)) / (
            self.sigma * sqrt(self.T - t)
        )

    def d2(self, S, t):
        return self.d1(S, t) - self.sigma * sqrt(self.T - t)

    def C(self, S, t):
        return N(self.d1(S, t)) * S - N(self.d2(S, t)) * self.K * exp(
            -self.r * (self.T - t)
        )

    def P(self, S, t):
        return (
            N(-self.d2(S, t)) * self.K * exp(-self.r * (self.T - t))
            - N(-self.d1(S, t)) * S
        )

    def CDelta(self, S, t):
        return N(self.d1(S, t))

    def PDelta(self, S, t):
        return self.CDelta(S, t) - 1

    def Gamma(self, S, t):
        return n(self.d1(S, t)) / (S * self.sigma * sqrt(self.T - t))

    def Vega(self, S, t):
        return S * n(self.d1(S, t)) * sqrt(self.T - t)

    def CTheta(self, S, t):
        return -(
            (S * n(self.d1(S, t)) * self.sigma) / (2 * sqrt(self.T - t))
        ) - self.r * self.K * exp(-self.r * (self.T - t)) * N(self.d2(S, t))

    def PTheta(self, S, t):
        return -(
            (S * n(self.d1(S, t)) * self.sigma) / (2 * sqrt(self.T - t))
        ) + self.r * self.K * exp(-self.r * (self.T - t)) * N(-self.d2(S, t))

    def CRho(self, S, t):
        return self.K * (self.T - t) * exp(-self.r * (self.T - t)) * N(self.d2(S, t))

    def PRho(self, S, t):
        return -self.K * (self.T - t) * exp(-self.r * (self.T - t)) * N(-self.d2(S, t))


BS = BlackScholesCalc(T, sigma, r, K)
print(f"Call price is: {BS.C(S, 0)}")
print(f"Call Delta is: {BS.CDelta(S, 0)}")
print(f"Gamma is: {BS.Gamma(S, 0)}")
print(f"Vega is: {BS.Vega(S, 0)}")
print(f"Call Theta is: {BS.CTheta(S, 0)}")
print(f"Call Rho is: {BS.CRho(S, 0)}")
print(f"Put price is: {BS.P(S, 0)}")
print(f"Put Delta is: {BS.PDelta(S, 0)}")
print(f"Put Theta is: {BS.PTheta(S, 0)}")
print(f"Put Rho is: {BS.PRho(S, 0)}")
