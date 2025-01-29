# -*- coding: utf-8 -*-
# Created on Tue Jan 28 2025
# Author: epeterson

from BSModel01 import Gauss
from numpy import cos

def Z():
    return Gauss() * 1 / 2
    
def X():
    return cos(Z())

def Y():
    return 1 - (Z() ** 2) / 2

# E(X) = E(X - Y) + y
# cos(y) ~= 1 - z ** 2 / 2
# y = E(Y) = 1 - E(Z) ** 2 / 2
y = 7 / 8

ExpX, ExpXY, HsqX, HsqXY = 0.0, 0.0, 0.0, 0.0
for i in range(100000):
    ExpXY = (i * ExpXY + X() - Y()) / (1.0 + i)
    HsqXY =(i * HsqXY + (X() - Y()) * (X() - Y()))/(i + 1.0)
    ExpX = (i * ExpX + X()) / (1.0 + i)
    HsqX =(i * HsqX + X() * X())/(i + 1.0)
    
ExpXCV = ExpXY + y

print(f'E(X) = {ExpX}\nWith Error {HsqX}')
print(f'E(X) with CV = {ExpXCV}\nWith Error {HsqXY}')