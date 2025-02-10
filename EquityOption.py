# -*- coding: utf-8 -*-
# Created on Sun Feb 09 2025
# Author: epeterson

import QuantLib as ql
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List

# ql.Instrument.VanillaOption
# ql.math.Integrals.TanhsinhIntegral
# ql.PricingEngines.Vanilla.AnalyticEuropeanEngine
# ql.PricingEngines.Vanilla.AnalyticEuropeanVasicekEngine
# ql.PricingEngines.Vanilla.AnalyticHestonEngine
# ql.PricingEngines.Vanilla.BaroneAdesiWhaleyEngine
# ql.PricingEngines.Vanilla.BatesEngine
# ql.PricingEngines.Vanilla.BinomialEngine
# ql.PricingEngines.Vanilla.BjerksundStenslandEngine
# ql.PricingEngines.Vanilla.FdBlackScholesVanillaEngine
# ql.PricingEngines.Vanilla.IntegralEngine
# ql.PricingEngines.Vanilla.MCAmericanEngine
# ql.PricingEngines.Vanilla.MCEuropeanEngine
# ql.PricingEngines.Vanilla.QdFpAmericanEngine
# ql.Time.Calendars.Target
# ql.Utilities.DataFormatters


def rate(rate: float) -> str:
    return f"{rate*100:.2f}%"

def EquityOption():

    try:
        # set up dates
        calendar: ql.Calendar = ql.TARGET();
        todaysDate: ql.Date = ql.Date(15, ql.May, 1998);
        settlementDate: ql.Date = ql.Date(17, ql.May, 1998);
        ql.Settings.instance().evaluationDate = todaysDate;

        # our options
        type: ql.Option = ql.Option.Put
        underlying: ql.Real = 36;
        strike: ql.Real = 40;
        dividendYield: ql.Spread= 0.00;
        riskFreeRate: ql.Rate = 0.06;
        volatility: ql.Volatility = 0.20;
        maturity: ql.Date = ql.Date(17, ql.May, 1999);
        dayCounter: ql.DayCounter = ql.Actual365Fixed();

        print(f"Option type = {type}")
        print(f"Maturity = {maturity}")
        print(f"Underlying price = {underlying}")
        print(f"Strike = {strike}")
        print(f"Risk-free interest rate = {rate(riskFreeRate)}")
        print(f"Dividend yield = {rate(dividendYield)}")
        print(f"Volatility = {rate(volatility)})")
        print()
        print()
        

        widths = [35, 14, 14, 14]

        print(
            "Method".ljust(widths[0]) +
            "European".ljust(widths[1]) +
            "Bermudan".ljust(widths[2]) +
            "American".ljust(widths[3])
        )

        exerciseDates: List[ql.Date] = [];
        for i in range(4):
            exerciseDates.append(settlementDate + 3*i*ql.Months);

        europeanExercise = ql.EuropeanExercise(maturity)
        bermudanExercise = ql.BermudanExercise(exerciseDates)
        americanExercise = ql.AmericanExercise(settlementDate, maturity)
        underlyingH = ql.makeQuoteHandle(underlying)

        # bootstrap the yield/dividend/vol curves
        flatTermStructure = ql.YieldTermStructureHandle(
            ql.FlatForward(settlementDate, riskFreeRate, dayCounter)
        )
        flatDividendTS = ql.YieldTermStructureHandle(
            ql.FlatForward(settlementDate, dividendYield, dayCounter)
        )
        flatVolTS = ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(settlementDate, calendar, volatility, dayCounter)
        )
        payoff = ql.PlainVanillaPayoff(type, strike)
        bsmProcess = ql.BlackScholesMertonProcess(
            underlyingH, flatDividendTS, flatTermStructure, flatVolTS
        )

        # options
        europeanOption = ql.VanillaOption(payoff, europeanExercise)
        bermudanOption = ql.VanillaOption(payoff, bermudanExercise)
        americanOption = ql.VanillaOption(payoff, americanExercise)
        
        # Analytic formulas:

        # Black-Scholes for European
        method = "Black-Scholes"
        europeanOption.setPricingEngine(ql.AnalyticEuropeanEngine(bsmProcess))
        print(method.ljust(widths[0]) +
                str(round(europeanOption.NPV(), 6)).ljust(widths[1]) +
                "N/A".ljust(widths[2]) +
                "N/A".ljust(widths[3])
            )
        
        # # Vasicek rates model for European
        # method = "Black Vasicek Model";
        # r0: ql.Real = riskFreeRate;
        # a: ql.Real = 0.3;
        # b: ql.Real = 0.3;
        # sigma_r: ql.Real = 0.15;
        # riskPremium: ql.Real = 0.0;
        # correlation: ql.Real = 0.5;
        # vasicekProcess = ql.Vasicek(r0, a, b, sigma_r, riskPremium)
        # europeanOption.setPricingEngine(ql.AnalyticBlackVasicekEngine(bsmProcess, vasicekProcess, correlation))
        # print(method.ljust(widths[0]) +
        #         str(europeanOption.NPV()).ljust(widths[1]) +
        #         "N/A".ljust(widths[2]) +
        #         "N/A".ljust(widths[3])
        #     )

        # semi-analytic Heston for European
        method = "Heston semi-analytic";
        hestonProcess: ql.HestonProcess = ql.HestonProcess(
            flatTermStructure, flatDividendTS,
            underlyingH, volatility*volatility,
            1.0, volatility*volatility, 0.001, 0.0
        )
        hestonModel: ql.HestonModel = ql.HestonModel(hestonProcess)
        europeanOption.setPricingEngine(ql.AnalyticHestonEngine(hestonModel))
        print(method.ljust(widths[0]) +
                str(round(europeanOption.NPV(), 6)).ljust(widths[1]) +
                "N/A".ljust(widths[2]) +
                "N/A".ljust(widths[3])
            )

        # semi-analytic Bates for European
        method = "Bates semi-analytic";
        batesProcess = ql.BatesProcess(
            flatTermStructure, flatDividendTS,
            underlyingH, volatility*volatility,
            1.0, volatility*volatility, 0.001, 0.0,
            1e-14, 1e-14, 1e-14
        )
        batesModel = ql.BatesModel(batesProcess)
        europeanOption.setPricingEngine(ql.BatesEngine(batesModel))
        print(method.ljust(widths[0]) +
                str(round(europeanOption.NPV(), 6)).ljust(widths[1]) +
                "N/A".ljust(widths[2]) +
                "N/A".ljust(widths[3])
            )
        
        # Barone-Adesi and Whaley approximation for American
        method = "Barone-Adesi/Whaley"
        americanOption.setPricingEngine(ql.BaroneAdesiWhaleyApproximationEngine(bsmProcess))
        print(method.ljust(widths[0]) +
                "N/A".ljust(widths[1]) +
                "N/A".ljust(widths[2]) +
                str(round(americanOption.NPV(), 6)).ljust(widths[3])
            )

        # Bjerksund and Stensland approximation for American
        method = "Bjerksund/Stensland"
        americanOption.setPricingEngine(ql.BjerksundStenslandApproximationEngine(bsmProcess))
        print(method.ljust(widths[0]) +
                "N/A".ljust(widths[1]) +
                "N/A".ljust(widths[2]) +
                str(round(americanOption.NPV(), 6)).ljust(widths[3])
            )

        # QD+ fixed-point engine for American
        method = "QD+ fixed-point (fast)"
        americanOption.setPricingEngine(ql.QdFpAmericanEngine(bsmProcess, ql.QdFpAmericanEngine.fastScheme()))
        print(method.ljust(widths[0]) +
                "N/A".ljust(widths[1]) +
                "N/A".ljust(widths[2]) +
                str(round(americanOption.NPV(), 6)).ljust(widths[3])
            )

        method = "QD+ fixed-point (accurate)"
        americanOption.setPricingEngine(ql.QdFpAmericanEngine(bsmProcess, ql.QdFpAmericanEngine.accurateScheme()))
        print(method.ljust(widths[0]) +
                "N/A".ljust(widths[1]) +
                "N/A".ljust(widths[2]) +
                str(round(americanOption.NPV(), 6)).ljust(widths[3])
            )

        method = "QD+ fixed-point (high precision)"
        americanOption.setPricingEngine(ql.QdFpAmericanEngine(bsmProcess, ql.QdFpAmericanEngine.highPrecisionScheme()))
        print(method.ljust(widths[0]) +
                "N/A".ljust(widths[1]) +
                "N/A".ljust(widths[2]) +
                str(round(americanOption.NPV(), 6)).ljust(widths[3])
            )

        # Integral
        method = "Integral"
        europeanOption.setPricingEngine(ql.IntegralEngine(bsmProcess))
        print(method.ljust(widths[0]) +
                str(round(europeanOption.NPV(), 6)).ljust(widths[1]) +
                "N/A".ljust(widths[2]) +
                "N/A".ljust(widths[3])
            )

        # Finite differences
        timeSteps: ql.Size = 801
        method = "Finite differences"
        fdengine = ql.FdBlackScholesVanillaEngine(bsmProcess, timeSteps, timeSteps-1)
        europeanOption.setPricingEngine(fdengine)
        bermudanOption.setPricingEngine(fdengine)
        americanOption.setPricingEngine(fdengine)
        print(method.ljust(widths[0]) +
                str(round(europeanOption.NPV(), 6)).ljust(widths[1]) +
                str(round(bermudanOption.NPV(), 6)).ljust(widths[2]) +
                str(round(americanOption.NPV(), 6)).ljust(widths[3])
            )
        
        # Binomial method: Jarrow-Rudd
        method = "Binomial Jarrow-Rudd"
        jrEngine = ql.BinomialJRVanillaEngine(bsmProcess, timeSteps)
        europeanOption.setPricingEngine(jrEngine)
        bermudanOption.setPricingEngine(jrEngine)
        americanOption.setPricingEngine(jrEngine)
        print(method.ljust(widths[0]) +
                str(round(europeanOption.NPV(), 6)).ljust(widths[1]) +
                str(round(bermudanOption.NPV(), 6)).ljust(widths[2]) +
                str(round(americanOption.NPV(), 6)).ljust(widths[3])
            )

        # Binomial method: Cox-Ross-Rubinstein
        method = "Binomial Cox-Ross-Rubinstein"
        crrEngine = ql.BinomialCRRVanillaEngine(bsmProcess, timeSteps)
        europeanOption.setPricingEngine(crrEngine)
        bermudanOption.setPricingEngine(crrEngine)
        americanOption.setPricingEngine(crrEngine)
        print(method.ljust(widths[0]) +
                str(round(europeanOption.NPV(), 6)).ljust(widths[1]) +
                str(round(bermudanOption.NPV(), 6)).ljust(widths[2]) +
                str(round(americanOption.NPV(), 6)).ljust(widths[3])
            )

        # Binomial method: Additive equiprobabilities
        method = "Additive equiprobabilities"
        aeqpEngine = ql.BinomialEQPVanillaEngine(bsmProcess, timeSteps)
        europeanOption.setPricingEngine(aeqpEngine)
        bermudanOption.setPricingEngine(aeqpEngine)
        americanOption.setPricingEngine(aeqpEngine)
        print(method.ljust(widths[0]) +
                str(round(europeanOption.NPV(), 6)).ljust(widths[1]) +
                str(round(bermudanOption.NPV(), 6)).ljust(widths[2]) +
                str(round(americanOption.NPV(), 6)).ljust(widths[3])
            )

        # Binomial method: Binomial Trigeorgis
        method = "Binomial Trigeorgis"
        trigeorgisEngine = ql.BinomialTrigeorgisVanillaEngine(bsmProcess, timeSteps)
        europeanOption.setPricingEngine(trigeorgisEngine)
        bermudanOption.setPricingEngine(trigeorgisEngine)
        americanOption.setPricingEngine(trigeorgisEngine)
        print(method.ljust(widths[0]) +
                str(round(europeanOption.NPV(), 6)).ljust(widths[1]) +
                str(round(bermudanOption.NPV(), 6)).ljust(widths[2]) +
                str(round(americanOption.NPV(), 6)).ljust(widths[3])
            )

        # Binomial method: Binomial Tian
        method = "Binomial Tian"
        tianEngine = ql.BinomialTianVanillaEngine(bsmProcess, timeSteps)
        europeanOption.setPricingEngine(tianEngine)
        bermudanOption.setPricingEngine(tianEngine)
        americanOption.setPricingEngine(tianEngine)
        print(method.ljust(widths[0]) +
                str(round(europeanOption.NPV(), 6)).ljust(widths[1]) +
                str(round(bermudanOption.NPV(), 6)).ljust(widths[2]) +
                str(round(americanOption.NPV(), 6)).ljust(widths[3])
            )

        # Binomial method: Binomial Leisen-Reimer
        method = "Binomial Leisen-Reimer"
        lrEngine = ql.BinomialLRVanillaEngine(bsmProcess, timeSteps)
        europeanOption.setPricingEngine(lrEngine)
        bermudanOption.setPricingEngine(lrEngine)
        americanOption.setPricingEngine(lrEngine)
        print(method.ljust(widths[0]) +
                str(round(europeanOption.NPV(), 6)).ljust(widths[1]) +
                str(round(bermudanOption.NPV(), 6)).ljust(widths[2]) +
                str(round(americanOption.NPV(), 6)).ljust(widths[3])
            )

        # Binomial method: Binomial Joshi
        method = "Binomial Joshi"
        joshiEngine = ql.BinomialJ4VanillaEngine(bsmProcess, timeSteps)
        europeanOption.setPricingEngine(joshiEngine)
        bermudanOption.setPricingEngine(joshiEngine)
        americanOption.setPricingEngine(joshiEngine)
        print(method.ljust(widths[0]) +
                str(round(europeanOption.NPV(), 6)).ljust(widths[1]) +
                str(round(bermudanOption.NPV(), 6)).ljust(widths[2]) +
                str(round(americanOption.NPV(), 6)).ljust(widths[3])
            )

        # Monte Carlo Method: MC (crude)
        timeSteps = 1
        method = "MC (crude)"
        mcSeed: ql.Size = 42
        mcengine1 = ql.MCEuropeanEngine(bsmProcess, traits="PseudoRandom", timeSteps=timeSteps, requiredTolerance=0.02, seed=mcSeed)
        europeanOption.setPricingEngine(mcengine1)
        # Real errorEstimate = europeanOption.errorEstimate();
        print(method.ljust(widths[0]) +
                str(round(europeanOption.NPV(), 6)).ljust(widths[1]) +
                "N/A".ljust(widths[2]) +
                "N/A".ljust(widths[3])
            )

        # Monte Carlo Method: QMC (Sobol)
        method = "QMC (Sobol)"
        nSamples: ql.Size = 32768  # 2^15

        mcengine2 = ql.MCEuropeanEngine(bsmProcess, traits="lowdiscrepancy", timeSteps=timeSteps, requiredSamples=nSamples)
        europeanOption.setPricingEngine(mcengine2)
        print(method.ljust(widths[0]) +
                str(round(europeanOption.NPV(), 6)).ljust(widths[1]) +
                "N/A".ljust(widths[2]) +
                "N/A".ljust(widths[3])
            )

        # Monte Carlo Method: MC (Longstaff Schwartz)
        method = "MC (Longstaff Schwartz)"
        mcengine3 = ql.MCAmericanEngine(bsmProcess, traits="PseudoRandom", timeSteps=100, antitheticVariate=True, nCalibrationSamples=4096, requiredTolerance=0.02, seed=mcSeed)
        americanOption.setPricingEngine(mcengine3)
        print(method.ljust(widths[0]) +
                "N/A".ljust(widths[1]) +
                "N/A".ljust(widths[2]) +
                str(round(americanOption.NPV(), 6)).ljust(widths[3])
            )

        return 0

    except Exception as e:
        print("Error: ", e)
        return 1
if __name__ == "__main__":
    EquityOption()