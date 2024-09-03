# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 15:43:36 2024

@author: nachi
"""
import numpy as np
import control as ctrl
import math
from scipy.optimize import minimize
from sigfig import round
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from sklearn.decomposition import PCA
from scipy.optimize import curve_fit

# Datos proporcionados
sigmas_b = np.array([0.012e-9,  #MAG-3 (AAC Clyde Space)
                     0.012e-9,  #TFM100G2 (Billingsley Aerospace & Defense)
                     0.035e-9,  #TFM65-VQS (Billingsley Aerospace & Defense)
                     0.1e-9,    #MAG-3_2 (AAC Clyde Space)
                     0.150e-9,  #Spacemag-Lite (Bartington Instruments)
                     1.18e-9,   #MM200 (AAC Clyde Space),
                     3e-9,      #DTFM100S (Billingsley Aerospace & Defense)
                     ])   

potencias_b = np.array([0.735, #W MAG-3 (AAC Clyde Space)
                    0.6125,    #TFM100G2 (Billingsley Aerospace & Defense)
                    0.504,     #TFM65-VQS (Billingsley Aerospace & Defense)
                    0.15,      #W MAG-3_2 (AAC Clyde Space)
                    0.175,     #Spacemag-Lite (Bartington Instruments)
                    0.1,       #MM200 (AAC Clyde Space)
                    0.11,      #DTFM100S (Billingsley Aerospace & Defense)
                    ])

precio_b = np.array([0,
                    0, 
                    0,
                    0,
                    0,
                    0,
                    0,
                    0
                    ])
# Definición de la función lineal
def linear_func(x, m, b):
    return m * x + b

# Ajuste de la función lineal
params, _ = curve_fit(linear_func, sigmas_b, potencias_b)
m, b = params

# Predicción
linear_fit = linear_func(sigmas_b, m, b)

# Visualización
plt.figure(figsize=(8, 6))
plt.scatter(sigmas_b, potencias_b, color='blue', label='Datos originales')
plt.plot(sigmas_b, linear_fit, color='red', label=f'Ajuste lineal: y = {m:.2e}x + {b:.2e}')
plt.xlabel('Desviación estándar del campo magnético (T)')
plt.ylabel('Potencia consumida (W)')
plt.title('Ajuste Lineal usando scipy.optimize')
plt.legend()
plt.grid(True)
plt.show()

# Coeficientes del ajuste
print(f'Pendiente (m): {m:.2e}')
print(f'Intersección (b): {b:.2e}')
#%%

# Datos proporcionados
sigmas_ss = np.array([0.033,  #CubeSense (CubeSpace Satellite Systems)
                      0.05,    #FSS (Bradford Space)
                      0.167,  #SSOC-D60 (Solar MEMS Technologies)
                      0.167,  #MSS-01(Space Micro)
                      0.5,    #CoSS (Bradford Space)
                      0.833,  #CSS-01, CSS-02 (Space Micro)
                     ])   

potencia_ss = np.array([
                    ])

precio_ss = np.array([0,
                    0, 
                    0,
                    0,
                    0,
                    0,
                    0,
                    0
                    ])

# Ajuste de una curva de segundo grado (polinomio de grado 2)
coeficientes_ss = np.polyfit(sigmas_ss, potencia_ss, 3)

# Obtener los coeficientes
aa, bb, cc,dd = coeficientes_ss
print(f"Coeficientes: a = {aa}, b = {bb}, c = {cc}, d = {dd}")

# Definir la función del polinomio ajustado
polinomio_ss = np.poly1d(coeficientes_ss)

# Graficar los datos y el ajuste
plt.scatter(sigmas_ss, potencia_ss, label='Datos', color='red')
plt.plot(sigmas_ss, polinomio_ss(sigmas_ss), label='Ajuste cuadrático', color='blue')
plt.xscale('log')
plt.xlabel('sigmas_ss')
plt.ylabel('masas_ss')
plt.legend()
plt.show()

#%%
# Datos proporcionados
sigmas_gyros = np.array([0.0116, #stim202 (Safran)
                         0.033, #NSGY-001 (NewSpace System)
                         0.050, #CRH03 – 010 (Silicon Sensing Systems)
                         0.12, #CRH03 – 200 (Silicon Sensing Systems)
                         

                     ])   

masas_gyros = np.array([0.030,
                    0.375,
                    0.035,
                    0.036, 
                    0.024,
                    0.01,
                    ])

precio_gyros = np.array([0,
                    0, 
                    0,
                    0,
                    0,
                    0,
                    0,
                    0
                    ])

# Ajuste de una curva de segundo grado (polinomio de grado 2)
coeficientes_gyros = np.polyfit(sigmas_gyros, masas_gyros, 2)

# Obtener los coeficientes
aaa, bbb, ccc = coeficientes_gyros
print(f"Coeficientes: a = {aaa}, b = {bbb}, c = {ccc}")

# Definir la función del polinomio ajustado
polinomio_gyros = np.poly1d(coeficientes_gyros)

# Graficar los datos y el ajuste
plt.scatter(sigmas_gyros, masas_gyros, label='Datos', color='red')
plt.plot(sigmas_gyros, polinomio_gyros(sigmas_gyros), label='Ajuste cuadrático', color='blue')
plt.xscale('log')
plt.xlabel('sigmas_gyros')
plt.ylabel('masas_gyros')
plt.legend()
plt.show()