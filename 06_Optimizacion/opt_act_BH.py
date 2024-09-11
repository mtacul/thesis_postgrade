# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 19:41:16 2024

@author: nachi
"""
from Suite_LQR_MT_act import suite_act
import numpy as np
from scipy.optimize import minimize


def objective(x):
    lim = x
    time_R,time_P,time_Y = suite_act(lim)
    print(f"lim: {lim}, time_R: {time_R}, time_P: {time_P}, time_Y: {time_Y}")
    time = np.linalg.norm(np.array([time_R,time_P,time_Y]))
    print(f"time_norm: {time}")
    return time**2  # Minimizar el cuadrado de la exactitud


bnds = [(0.5, 15)]

# Valores iniciales para las desviaciones estándar
initial_guess = [1]  
current_x0 = np.array(initial_guess)
# result = minimize(objective, current_x0, method='SLSQP', bounds=bnds, options={'disp':True})
# result = minimize(objective, current_x0, method='Nelder-Mead', bounds=bnds, options={'disp':True})
result = minimize(objective, current_x0, method='L-BFGS-B', bounds=bnds, options={'disp':True,'ftol': 1e-8}) #[3.3e-02 1.2e-11]  f= 11.20625354107872

print(result.x)
print(result.fun)
