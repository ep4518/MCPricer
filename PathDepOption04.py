# -*- coding: utf-8 -*-
# Created on Thu Jan 23 2025
# Author: epeterson

from abc import ABC, abstractmethod
from BSModel01 import Model, SamplePath, Vector
from numpy import exp, sqrt, log, prod, atan
from tqdm import tqdm

# Control Variates


def N(x: float):
    gamma = 0.2316419
    a1 = 0.319381530
    a2 = -0.356563782
    a3 = 1.781477937
    a4 = -1.821255978
    a5 = 1.330274429
    pi = 4.0 * atan(1.0)
    k = 1.0 / (1.0 + gamma * x)
    if x >= 0.0:
        return 1.0 - ((((a5 * k + a4) * k + a3) * k + a2) * k + a1) * k * exp(
            -x * x / 2.0
        ) / sqrt(2.0 * pi)
    else:
        return 1.0 - N(-x)


class PathDepOption(ABC):

    def __init__(self, T: float, K: float, m: int):
        self.T = T
        self.K = K
        self.m = m
        self.Paths = []
        self.PricingError = None
        self.Price = None

    def Rescale(
        self, S: SamplePath, Z: Vector, S0: float, r: float, sigma: float, dt: float
    ):
        S[0] = S0 * exp((r - sigma * sigma / 2.0) * dt + sigma * sqrt(dt) * Z[0])
        for j in range(1, self.m):
            S[j] = S[j - 1] * exp(
                (r - sigma * sigma / 2.0) * dt + sigma * sqrt(dt) * Z[j]
            )

    def GetZ(
        self, Z: Vector, S: SamplePath, S0: float, r: float, sigma: float, dt: float
    ):
        Z[0] = (log(S[0] / S0) - (r - sigma * sigma / 2.0) * dt) / (sigma * sqrt(dt))
        for j in range(1, self.m):
            Z[j] = (log(S[j] / S[j - 1]) - (r - sigma * sigma / 2.0) * dt) / (
                sigma * sqrt(dt)
            )

    def PriceByMC(self, Engine: Model, N: float):
        H: float = 0.0
        Hsq: float = 0.0
        S: SamplePath = [0.0] * self.m

        for i in tqdm(range(N)):

            Engine.GenerateSamplePath(self.T, self.m, S)

            H = (i * H + self.Payoff(S)) / (i + 1.0)
            Hsq = (i * Hsq + pow(self.Payoff(S), 2.0)) / (i + 1.0)

        self.Price = exp(-Engine.r * self.T) * H
        self.PricingError = exp(-Engine.r * self.T) * sqrt(Hsq - H * H) / sqrt(N - 1.0)
        return self.Price

    def PriceVarRedMC(self, Engine: Model, N: float, CVOption: "PathDepOption"):
        VarRedOpt = DifferenceOfOptions(self, CVOption, self.T, self.m)
        self.Price = VarRedOpt.PriceByMC(Engine, N) + CVOption.PriceByBSFormula(Engine)
        self.PricingError = VarRedOpt.PricingError
        return self.Price

    def PriceByBSFormula(self, Engine: Model):
        return 0.0

    @abstractmethod
    def Payoff(self, S: SamplePath):
        pass


class DifferenceOfOptions(PathDepOption):
    def __init__(self, O1: PathDepOption, O2: PathDepOption, T: float, m: int):
        self.O1 = O1
        self.O2 = O2
        self.T = T
        self.m = m

    def Payoff(self, S):
        return self.O1.Payoff(S) - self.O2.Payoff(S)


class ArthmAsianCall(PathDepOption):

    def __init__(self, T: float, K: float, m: int):
        super().__init__(T, K, m)

    def Payoff(self, S):
        avg = 0.0
        for i in range(self.m):
            avg = (i * avg + S[i]) / (i + 1)
        return max(0.0, avg - self.K)


class GmtrAsianCall(PathDepOption):

    def __init__(self, T, K, m):
        super().__init__(T, K, m)

    def Payoff(self, S: SamplePath):
        return max(pow(prod(S), 1.0 / self.m) - self.K, 0)

    def PriceByBSFormula(self, Engine):
        a: float = (
            exp(-Engine.r * self.T)
            * Engine.S0
            * exp(
                (self.m + 1.0)
                * self.T
                / (2.0 * self.m)
                * (
                    Engine.r
                    + Engine.sigma
                    * Engine.sigma
                    * ((2.0 * self.m + 1.0) / (3.0 * self.m) - 1.0)
                    / 2.0
                )
            )
        )
        b: float = Engine.sigma * sqrt(
            (self.m + 1.0) * (2.0 * self.m + 1.0) / (6.0 * self.m**2)
        )
        G = EuropeanCall(self.T, self.K, self.m)
        return G.PriceByBSFormula(a, b, Engine.r)


class EuropeanCall(PathDepOption):

    def __init__(self, T: float, K: float, m: int):
        super().__init__(T, K, m)

    def Payoff(self, S):
        return max(0.0, S[-1] - self.K)

    def d_plus(self, S0: float, sigma: float, r: float):
        return (log(S0 / self.K) + (r + 0.5 * pow(sigma, 2.0)) * self.T) / (
            sigma * sqrt(self.T)
        )

    def d_minus(self, S0: float, sigma: float, r: float):
        return self.d_plus(S0, sigma, r) - sigma * sqrt(self.T)

    def PriceByBSFormula(self, S0: float, sigma: float, r: float):
        return S0 * N(self.d_plus(S0, sigma, r)) - self.K * exp(-r * self.T) * N(
            self.d_minus(S0, sigma, r)
        )


class CallUpOut(PathDepOption):

    def __init__(self, T: float, K: float, B: float, R: float, m: int):
        super().__init__(T, K, m)
        self.Barrier = B
        self.Rebate = R

    def plotMC(self):
        fig, ax = super().plotMC()
        ax.axhline(self.Barrier, color="red")
        ax.axhline(self.Rebate + 1, color="green")
        return fig, ax

    def Payoff(self, S):
        indicKI = max(S) > self.Barrier
        return self.Rebate if indicKI else max(S[-1] - self.K, 0)


if __name__ == "__main__":
    from BSModel01 import BSModel

    S0 = 100
    r = 0.03
    sigma = 0.2
    model = BSModel(S0, r, sigma)
    T = 1.0 / 12
    K = 100
    m = 30

    Option = ArthmAsianCall(T, K, m)
    CVOption = GmtrAsianCall(T, K, m)

    Nb = 10000

    Option.PriceVarRedMC(model, Nb, CVOption)
    print(f"Artithmetic call price = {Option.Price}\nError = {Option.PricingError}")
    Option.PriceByMC(model, Nb)
    print(f"Price by direct MC = {Option.Price}\nMC Error = {Option.PricingError}")
