# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 16:11:26 2024

@author: nachi
"""

from Suite_LQR_MT_sen import suite_sim
import numpy as np
from scipy.optimize import minimize


def objective(x):
    std_sensor_sol, std_magnetometros = x
    acc, psd = suite_sim(std_sensor_sol, std_magnetometros)
    return psd**2  # Minimizar el cuadrado de la exactitud


bnds = ((0.033,0.833),(0.012e-9,3e-9))

# Valores iniciales para las desviaciones estándar
initial_guess = [0.68, 1e-9]  # primero ss y luego magn
current_x0 = np.array(initial_guess)
# result = minimize(objective, current_x0, method='SLSQP', bounds=bnds, options={'disp':True}) #[5.00140157e-02 2.99999989e-09] f= 11.248222129126296  f_Eval = 15
# result = minimize(objective, current_x0, method='Nelder-Mead', bounds=bnds, options={'disp':True}) #[3.3e-02 3.0e-09] f= 11.203883474889103  f_Eval = 21
result = minimize(objective, current_x0, method='L-BFGS-B', bounds=bnds, options={'disp':True,'ftol': 1e-4}) #[3.3e-02 1.2e-11]  f= 11.20625354107872

print(result.x)
print(result.fun)