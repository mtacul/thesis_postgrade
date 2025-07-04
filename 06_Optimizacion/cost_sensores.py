# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 15:43:36 2024

@author: nachi
"""
#%% Magnetometro
import numpy as np
import matplotlib.pyplot as plt
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

masas_b = np.array([0.1,       #MAG-3 (AAC Clyde Space)
                    0.1,       #TFM100G2 (Billingsley Aerospace & Defense)
                    0.117,     #TFM65-VQS (Billingsley Aerospace & Defense)
                    0.1  ,     # MAG-3_2 (AAC Clyde Space)
                    0.062,     #Spacemag-Lite (Bartington Instruments)
                    0.012,     #MM200 (AAC Clyde Space)
                    0.1        #DTFM100S (Billingsley Aerospace & Defense)
                    ])

vol_b = np.array([3.51*3.23*8.26,
                  3.51* 3.23* 8.26,
                  3.51*3.23 *8.26,
                  3.51*3.23 *8.26,
                  2*2*2,
                  3.3 * 2.0 *1.13,
                  8.26 *3.51 * 3.23          
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

# Función potencial: y = a * x^b
def func_potencial(x, a, b):
    return a * x**b

# Función logarítmica: y = a * log(x) + b
def func_log(x, a, b):
    return a * np.log(x) + b

def func_lin(x,a,b):
    return a*x + b

# Ajuste Potencial
params_pot, _ = curve_fit(func_potencial, sigmas_b, potencias_b)
a_pot_b, b_pot_b = params_pot

params_mass, _ = curve_fit(func_potencial, sigmas_b, masas_b)
a_mass_b, b_mass_b = params_mass

# params_vol, _ = curve_fit(func_potencial, sigmas_b, vol_b)
params_vol, _ = curve_fit(func_lin, sigmas_b, vol_b)
a_vol_b, b_vol_b = params_vol

# Crear valores de x para la curva ajustada
x_fit_b = np.linspace(min(sigmas_b), max(sigmas_b), 100)

# Calcular los valores ajustados
y_potencial_b = func_potencial(x_fit_b, a_pot_b, b_pot_b)
y_mass_b = func_potencial(x_fit_b, a_mass_b, b_mass_b)
# y_vol = func_potencial(x_fit, a_vol, b_vol)
y_vol_b = func_lin(x_fit_b, a_vol_b, b_vol_b)

# Graficar los datos originales
# plt.scatter(sigmas_b, potencias_b, label='Datos originales potencia')
plt.scatter(sigmas_b, potencias_b, label='Original Power Data')
# plt.plot(x_fit, y_potencial, label=f'Ajuste Potencial: $y = {a_pot:.2e} x^{{{b_pot:.2f}}}$', color='green')
plt.plot(x_fit_b, y_potencial_b, label=f'Potential adjust: $y = {a_pot_b:.2e} x^{{{b_pot_b:.2f}}}$', color='green')
# plt.xlabel('Sigmas_b (T)')
plt.xlabel('Magnetometer Standard deviation (T)')
# plt.ylabel('Potencias_b (W)')
plt.ylabel('Magnetometer Mean Power (W)')
plt.legend()
plt.grid
plt.savefig('potencia_magn.pdf', format='pdf')
plt.show()

# plt.scatter(sigmas_b, masas_b, label='Datos originales masa')
plt.scatter(sigmas_b, masas_b, label='Original Mass Data')
# plt.plot(x_fit, y_mass, label=f'Ajuste Potencial: $y = {a_mass:.2e} x^{{{b_mass:.2f}}}$', color='green')
plt.plot(x_fit_b, y_mass_b, label=f'Potential adjust: $y = {a_mass_b:.2e} x^{{{b_mass_b:.2f}}}$', color='green')
# plt.xlabel('Sigmas_b (T)')
plt.xlabel('Magnetometer Standard deviation (T)')
plt.ylabel('Magnetometer Mass (kg)')
plt.legend()
plt.grid
plt.savefig('mass_magn.pdf', format='pdf')
plt.show()
# plt.scatter(sigmas_b, vol_b, label='Datos originales volumen')
# plt.plot(x_fit, y_vol, label=f'Ajuste Potencial: $y = {a_vol}x+{{{b_vol:.2e}}}$', color='green')
# plt.xlabel('Sigmas_b (T)')
# plt.ylabel('Vol_b (cm3)')
# plt.legend()
# plt.show()

# Graficar ajuste Potencial
# plt.plot(x_fit, y_potencial, label=f'Ajuste Potencial: $y = {a_pot:.2e} x^{{{b_pot:.2f}}}$', color='green')

# Graficar ajuste Logarítmico

# Formato de la gráfica

# plt.show()
#%% Sun sensor
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
# Datos proporcionados
# sigmas_ss = np.array([0.033,  #CubeSense (CubeSpace Satellite Systems)
#                       0.05,    #FSS (Bradford Space)
#                       0.167,  #SSOC-D60 (Solar MEMS Technologies)
#                       0.167,  #MSS-01(Space Micro)
#                       0.5,    #CoSS (Bradford Space)
#                       0.833,  #CSS-01, CSS-02 (Space Micro)
#                      ])   

sigmas_ss = np.array([0.02/2, #SMART SUN SENSOR (Leonardo)
                      0.2/2, #CubeSense (Gen2)
                      0.3/3,  #SSOC-D60 (Solar MEMS Technologies)
                      0.5/3, #NanoSSOC-D60 (Solar MEMS Technologies)
                      0.2/1, #CubeSense Sun (Gen1)
                      1/3, #SS Sun Sensor - Medium (OCE Technology)
                      1/3, #MSS-01(Space Micro)
                      5/3 #CoSS (Bradford Space)
                     ])   

masas_ss = np.array([0.33, #SMART SUN SENSOR (Leonardo)
                      0.015, #CubeSense (Gen2)
                      0.035,  #SSOC-D60 (Solar MEMS Technologies)
                      0.0065, #NanoSSOC-D60 (Solar MEMS Technologies)
                      0.03, #CubeSense Sun (Gen1)
                      0.077, #SS Sun Sensor - Medium (OCE Technology)
                      0.036, #MSS-01(Space Micro)
                      0.01 #CoSS (Bradford Space)
                     ])   

potencias_ss = np.array([0.7, #SMART SUN SENSOR (Leonardo)
                        0.1,   #CubeSense Sun (Gen2)
                        0.35,   #SSOC-D60 (Solar MEMS Technologies)
                        0.0759, #NanoSSOC-D60 (Solar MEMS Technologies)
                        0.1,    #CubeSense Sun (Gen1)
                        0,      #SS Sun Sensor - Medium (OCE Technology)
                        0,      #MSS-01(Space Micro)
                        0       #CoSS (Bradford Space)     
                    ])

vol_ss = np.array([11.2*1.2*4.3, #SMART SUN SENSOR (Leonardo)
                        3.5*2.4*2.2,   #CubeSense Sun (Gen2)
                        5.0*3.0*1.2,   #SSOC-D60 (Solar MEMS Technologies)
                        4.3*1.4 *0.59, #NanoSSOC-D60 (Solar MEMS Technologies)
                        4.17 *1.77 *2.29,    #CubeSense Sun (Gen1)
                        6.0*6.0*2.6,      #SS Sun Sensor - Medium (OCE Technology)
                        2.43 * (3.49/2)**2*np.pi,      #MSS-01(Space Micro)
                        0.9 * (1.27/2)**2*np.pi       #CoSS (Bradford Space)     
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

# Función potencial: y = a * x^b
def func_potencial(x, a, b):
    return a * x**b

# Función logarítmica: y = a * log(x) + b
def func_log(x, a, b):
    return a * np.log(x) + b

# Ajuste Potencial
params_pot, _ = curve_fit(func_potencial, sigmas_ss, potencias_ss)
a_pot_ss, b_pot_ss = params_pot

params_mass, _ = curve_fit(func_potencial, sigmas_ss, masas_ss)
a_mass_ss, b_mass_ss = params_mass

params_vol, _ = curve_fit(func_potencial, sigmas_ss, vol_ss)
# params_vol, _ = curve_fit(func_log, sigmas_ss, vol_ss)
a_vol_ss, b_vol_ss = params_vol

# Crear valores de x para la curva ajustada
x_fit_ss = np.linspace(min(sigmas_ss), max(sigmas_ss), 100)

# Calcular los valores ajustados
y_potencial_ss = func_potencial(x_fit_ss, a_pot_ss, b_pot_ss)
y_mass_ss = func_potencial(x_fit_ss, a_mass_ss, b_mass_ss)
# y_vol = func_log(x_fit, a_vol, b_vol)
y_vol_ss = func_potencial(x_fit_ss, a_vol_ss, b_vol_ss)

# Graficar los datos originales
# plt.scatter(sigmas_ss, potencias_ss, label='Datos originales potencia')
plt.scatter(sigmas_ss, potencias_ss, label='Original Power Data')
# plt.plot(x_fit, y_potencial, label=f'Ajuste Potencial: $y = {a_pot:.2e} x^{{{b_pot:.2f}}}$', color='green')
plt.plot(x_fit_ss, y_potencial_ss, label=f'Potential adjust: $y = {a_pot_ss:.2e} x^{{{b_pot_ss:.2f}}}$', color='green')
# plt.xlabel('Sigmas_ss (°)')
plt.xlabel('Sun sensor Standard deviation (°)')
# plt.ylabel('Potencias_ss (W)')
plt.ylabel('Sun sensor Mean Power (W)')# Graficar los datos originales

# plt.scatter(sigmas_ss, masas_ss, label='Datos originales masa')
plt.scatter(sigmas_ss, masas_ss, label='Original Mass Data')
# plt.plot(x_fit, y_mass, label=f'Ajuste Potencial: $y = {a_mass:.2e} x^{{{b_mass:.2f}}}$', color='green')
plt.plot(x_fit_ss, y_mass_ss, label=f'Potential adjust: $y = {a_mass_ss:.2e} x^{{{b_mass_ss:.2f}}}$', color='green')
# plt.xlabel('Sigmas_ss (°)')
plt.xlabel('Sun sensor Standard deviation (°)')
# plt.ylabel('Masas_ss (kg)')
plt.ylabel('Sun sensor Mass (kg)')
plt.legend()
plt.savefig('masa_ss.pdf', format='pdf')
plt.show()

plt.scatter(sigmas_ss, vol_ss, label='Datos originales volumen')
plt.plot(x_fit_ss, y_vol_ss, label=f'Ajuste Potencial: $y = {a_vol_ss:.2e} x^{{{b_vol_ss:.2f}}}$', color='green')
plt.xlabel('Sigmas_ss (°)')
plt.ylabel('Vol_ss (cm3)')
plt.legend()
plt.show()

#%% Magnetorquer

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
# Datos proporcionados

lim_MT = np.array([0.29,  #NCTR-M003 (NewSpace Systems)
                   1.0, #GMAT-1 (Michigan Aerospace Manufacturers Association)
                   1.6, #NCTR-M016 (NewSpace Systems)
                   10, #Tamam MT 1286-0003 (IAI)
                   10, #MQ10 (OCE Technology)
                   10, #GMAT-10 (Michigan Aerospace Manufacturers Association)
                   15 #Magnetic Torquer (Chang Guang Satellite Technology)
                   # 70 #Tamam MT 1286-0010 (IAI)
                   ])   

masas_MT = np.array([0.03,  #NCTR-M003 (NewSpace Systems)
                     0.05, #GMAT-1 (Michigan Aerospace Manufacturers Association)
                     0.053, #NCTR-M016 (NewSpace Systems)
                     0.26, #Tamam MT 1286-0003 (IAI)
                     0.45, #MQ10 (OCE Technology)
                     0.66, #GMAT-10 (Michigan Aerospace Manufacturers Association)
                     0.67 #Magnetic Torquer (Chang Guang Satellite Technology)
                     # 1.7 #Tamam MT 1286-0010 (IAI)
                     ])   

potencias_MT = np.array([0.25,  #NCTR-M003 (NewSpace Systems) 
                         0.25, #GMAT-1 (Michigan Aerospace Manufacturers Association)
                         1.2, #NCTR-M016 (NewSpace Systems)
                         1.1, #Tamam MT 1286-0003 (IAI)
                         0.4, #MQ10 (OCE Technology)
                         1.4, #GMAT-10 (Michigan Aerospace Manufacturers Association)
                         3 #Magnetic Torquer (Chang Guang Satellite Technology)
                         # 2.5 #Tamam MT 1286-0010 (IAI)
                    ])

vol_MT = np.array([7.2* 1.5*1.3, #NCTR-M003 (NewSpace Systems)    
                   (1.1/2)**2*np.pi*11.5, #GMAT-1 (Michigan Aerospace Manufacturers Association)
                   10.7* 1.5 * 1.3,  #NCTR-M016 (NewSpace Systems)  
                   (1.1/2)**2*np.pi*27, #Tamam MT 1286-0003 (IAI)
                   31.0 * 5.6 * 4.4 , #MQ10 (OCE Technology)
                   24.5*5.0*3.6, #GMAT-10 (Michigan Aerospace Manufacturers Association)
                   28*5.6*5.35 #Magnetic Torquer (Chang Guang Satellite Technology)
                   # (1.1/2)**2*np.pi*60, #Tamam MT 1286-0010 (IAI)
                   ])

precio_MT = np.array([0,
                    0, 
                    0,
                    0,
                    0,
                    0,
                    0,
                    0
                    ])

# Función potencial: y = a * x^b
def func_potencial(x, a, b):
    return a * x**b


# Ajuste Potencial
params_pot, _ = curve_fit(func_potencial, lim_MT, potencias_MT)
a_pot_tau, b_pot_tau = params_pot

params_mass, _ = curve_fit(func_potencial, lim_MT, masas_MT)
a_mass_tau, b_mass_tau = params_mass

params_vol, _ = curve_fit(func_potencial, lim_MT, vol_MT)
# params_vol, _ = curve_fit(func_log, sigmas_ss, vol_ss)
a_vol_tau, b_vol_tau = params_vol

# Crear valores de x para la curva ajustada
x_fit_tau = np.linspace(min(lim_MT), max(lim_MT), 100)

# Calcular los valores ajustados
y_potencial_tau = func_potencial(x_fit_tau, a_pot_tau, b_pot_tau)
y_mass_tau = func_potencial(x_fit_tau, a_mass_tau, b_mass_tau)
# y_vol = func_log(x_fit, a_vol, b_vol)
y_vol_tau = func_potencial(x_fit_tau, a_vol_tau, b_vol_tau)

# Graficar los datos originales
plt.scatter(lim_MT, potencias_MT, label='Datos originales potencia')
plt.plot(x_fit_tau, y_potencial_tau, label=f'Ajuste Potencial: $y = {a_pot_tau:.2e} x^{{{b_pot_tau:.2f}}}$', color='green')
plt.xlabel('lim_MT (Am2)')
plt.ylabel('Potencias_MT (W)')
plt.legend()
plt.savefig('potencia_MT.pdf', format='pdf')
plt.show()

plt.scatter(lim_MT, masas_MT, label='Datos originales masa')
plt.plot(x_fit_tau, y_mass_tau, label=f'Ajuste Potencial: $y = {a_mass_tau:.2e} x^{{{b_mass_tau:.2f}}}$', color='green')
plt.xlabel('lim_MT (Am2)')
plt.ylabel('Masas_MT (kg)')
plt.legend()
plt.savefig('masa_MT.pdf', format='pdf')
plt.show()

plt.scatter(lim_MT, vol_MT, label='Datos originales volumen')
plt.plot(x_fit_tau, y_vol_tau, label=f'Ajuste Potencial: $y = {a_vol_tau:.2e} x^{{{b_vol_tau:.2f}}}$', color='green')
plt.xlabel('lim_MT (Am2)')
plt.ylabel('Vol_MT (cm3)')
plt.legend()
plt.show()

#%% Rueda de reaccion

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
# Datos proporcionados

lim_RW = np.array([
                    0.001,   #CubeWheel(SAVOMID)
                    0.001,   #RW-0.01 (Sinclair Interplanetary)
                    0.0023,  #Gen1: CubeWheel Large (CubeSpace)
                    0.005,   #RW 35 (Astrofein)
                    0.025,   #RW110 (SpaceOmiD)
                    0.025,   #VRM-05 (Vectronic Aerospace)
                    0.033,   #RSI 02-33-30A (Collins Aerospace)
                    0.25     #RW4 (Blue Canyon Technologies) 
                    ])   

masas_RW = np.array([
                      0.08,   #CubeWheel(SAVOMID)
                      0.12,   #RW-0.01 (Sinclair Interplanetary)
                      0.225,  #Gen1: CubeWheel Large (CubeSpace)
                      0.5,   #RW 35 (Astrofein)
                      0.95,   #RW110 (SpaceOmiD)
                      1.3,   #VRM-05 (Vectronic Aerospace)
                      1.75,   #RSI 02-33-30A (Collins Aerospace)
                      3.2    #RW4 (Blue Canyon Technologies) 
                      ])   

potencias_RW = np.array([
                        0.3,   #CubeWheel(SAVOMID)
                        1.05,   #RW-0.01 (Sinclair Interplanetary)
                        0.35,  #Gen1: CubeWheel Large (CubeSpace)
                        4.0 ,   #RW 35 (Astrofein)
                        4.0 ,   #RW110 (SpaceOmiD)
                        3.0,   #VRM-05 (Vectronic Aerospace)
                        10.0,   #RSI 02-33-30A (Collins Aerospace)
                        10.0    #RW4 (Blue Canyon Technologies) 
                    ])

vol_RW = np.array([
                    3.4 * 3.4 * 1.7,   #CubeWheel(SAVOMID)
                    5.0* 5.0* 3.0 ,   #RW-0.01 (Sinclair Interplanetary)
                    5.7*5.7*3.15 ,  #Gen1: CubeWheel Large (CubeSpace)
                    10.2 * 10.2 * 5.8 ,   #RW 35 (Astrofein)
                    11.75*10.3*3.9,   #RW110 (SpaceOmiD)
                    11.5*11.5*7.7,   #VRM-05 (Vectronic Aerospace)
                    (13.55/2)**2*np.pi*11,   #RSI 02-33-30A (Collins Aerospace)
                    17* 17 * 7               #RW4 (Blue Canyon Technologies) 
                    ])

precio_RW = np.array([0,
                    0, 
                    0,
                    0,
                    0,
                    0,
                    0,
                    0
                    ])

# Función potencial: y = a * x^b
def func_potencial(x, a, b):
    return a * x**b


# Ajuste Potencial
params_pot, _ = curve_fit(func_potencial, lim_RW, potencias_RW)
a_pot_tauRW, b_pot_tauRW = params_pot

params_mass, _ = curve_fit(func_potencial, lim_RW, masas_RW)
a_mass_tauRW, b_mass_tauRW = params_mass

params_vol, _ = curve_fit(func_potencial, lim_RW, vol_RW)
# params_vol, _ = curve_fit(func_log, sigmas_ss, vol_ss)
a_vol_tauRW, b_vol_tauRW = params_vol

# Crear valores de x para la curva ajustada
x_fit_tauRW = np.linspace(min(lim_RW), max(lim_RW), 100)

# Calcular los valores ajustados
y_potencial_tauRW  = func_potencial(x_fit_tauRW , a_pot_tauRW, b_pot_tauRW)
y_mass_tauRW = func_potencial(x_fit_tauRW, a_mass_tauRW, b_mass_tauRW)
# y_vol = func_log(x_fit, a_vol, b_vol)
y_vol_tauRW = func_potencial(x_fit_tauRW, a_vol_tauRW, b_vol_tauRW)

# Graficar los datos originales
plt.scatter(lim_RW, potencias_RW, label='Datos originales potencia')
plt.plot(x_fit_tauRW, y_potencial_tauRW, label=f'Ajuste Potencial: $y = {a_pot_tauRW:.2e} x^{{{b_pot_tauRW:.2f}}}$', color='green')
plt.xlabel('lim_RW (Nm)')
plt.ylabel('Potencias_RW (W)')
plt.legend()
plt.savefig('potencia_RW.pdf', format='pdf')
plt.show()

plt.scatter(lim_RW, masas_RW, label='Datos originales masa')
plt.plot(x_fit_tauRW, y_mass_tauRW, label=f'Ajuste Potencial: $y = {a_mass_tauRW:.2e} x^{{{b_mass_tauRW:.2f}}}$', color='green')
plt.xlabel('lim_RW (Nm)')
plt.ylabel('Masas_RW (kg)')
plt.legend()
plt.savefig('masa_RW.pdf', format='pdf')
plt.show()

plt.scatter(lim_RW, vol_RW, label='Datos originales volumen')
plt.plot(x_fit_tauRW, y_vol_tauRW, label=f'Ajuste Potencial: $y = {a_vol_tauRW:.2e} x^{{{b_vol_tauRW:.2f}}}$', color='green')
plt.xlabel('lim_RW (Nm)')
plt.ylabel('Vol_RW (cm3)')
plt.legend()
plt.show()

# 


#%%
fig, axs = plt.subplots(2, 2, figsize=(8, 8))  # 3 filas, 2 columnas

# === 1. Magnetorquers ===
# Power
axs[0, 0].scatter(lim_MT, potencias_MT, label='Original Power Data')
axs[0, 0].plot(x_fit_tau, y_potencial_tau, label=f'Fit: $y = {a_pot_tau:.2e} x^{{{b_pot_tau:.2f}}}$', color='green')
axs[0, 0].set_xlabel('lim_MT (Am²)')
axs[0, 0].set_ylabel('Power (W)')
axs[0, 0].set_xlim([min(lim_MT)-3, max(lim_MT)+3])
axs[0, 0].set_ylim([min(potencias_MT)-0.5, max(potencias_MT)+0.1])
axs[0, 0].legend()
axs[0, 0].grid(True)

# === 2. Reaction Wheels  ===
# Power
axs[0, 1].scatter(lim_RW, potencias_RW, label='Original Power Data')
axs[0, 1].plot(x_fit_tauRW, y_potencial_tauRW, label=f'Fit: $y = {a_pot_tauRW:.2e} x^{{{b_pot_tauRW:.2f}}}$', color='green')
axs[0, 1].set_xlabel('lim_RW (Nm)')
axs[0, 1].set_ylabel('Power (W)')
# axs[0, 1].set_xlim([min(lim_RW)-3, max(lim_RW)+3])
# axs[0, 1].set_ylim([min(potencias_RW)-0.5, max(potencias_RW)+0.1])
axs[0, 1].legend()
axs[0, 1].grid(True)

# === 3. Sun Sensors ===
# Power
axs[1, 0].scatter(sigmas_ss, potencias_ss, label='Original Power Data')
axs[1, 0].plot(x_fit_ss, y_potencial_ss, label=f'Fit: $y = {a_pot_ss:.2e} x^{{{b_pot_ss:.2f}}}$', color='green')
axs[1, 0].set_xlabel('Sun sensor σ (°)')
axs[1, 0].set_ylabel('Power (W)')
axs[1, 0].set_xlim([min(sigmas_ss)-0.1, max(sigmas_ss)+0.1])
axs[1, 0].set_ylim([min(potencias_ss)-0.1, max(potencias_ss)+0.1])
axs[1, 0].legend()
axs[1, 0].grid(True)

# === 4. Magnetometers ===
# Power
axs[1, 1].scatter(sigmas_b, potencias_b, label='Original Power Data')
axs[1, 1].plot(x_fit_b, y_potencial_b, label=f'Fit: $y = {a_pot_b:.2e} x^{{{b_pot_b:.2f}}}$', color='green')
axs[1, 1].set_xlabel('Magnetometer σ (T)')
axs[1, 1].set_ylabel('Power (W)')
axs[1, 1].set_xlim([min(sigmas_b)-0.5e-9, max(sigmas_b)+0.5e-9])
axs[1, 1].set_ylim([min(potencias_b)-0.1, max(potencias_b)+0.1])
axs[1, 1].legend()
axs[1, 1].grid(True)

plt.tight_layout()
plt.savefig('physical_components_Power.pdf', format='pdf')
plt.show()



#%%
fig, axs = plt.subplots(2, 2, figsize=(8, 8))  # 3 filas, 2 columnas

# === 1. Magnetorquers ===
# Mass
axs[0, 0].scatter(lim_MT, masas_MT, label='Original Mass Data')
axs[0, 0].plot(x_fit_tau, y_mass_tau, label=f'Fit: $y = {a_mass_tau:.2e} x^{{{b_mass_tau:.2f}}}$', color='green')
axs[0, 0].set_xlabel('lim_MT (Am²)')
axs[0, 0].set_ylabel('Mass (kg)')
axs[0, 0].set_xlim([min(lim_MT)-3, max(lim_MT)+3])
axs[0, 0].set_ylim([min(masas_MT)-0.1, max(masas_MT)+0.1])
axs[0, 0].legend()
axs[0, 0].grid(True)

# === 2. Reaction Wheels  ===
# Mass
axs[0, 1].scatter(lim_RW, masas_RW, label='Original Power Data')
axs[0, 1].plot(x_fit_tauRW, y_mass_tauRW, label=f'Fit: $y = {a_mass_tauRW:.2e} x^{{{b_mass_tauRW:.2f}}}$', color='green')
axs[0, 1].set_xlabel('lim_RW (Nm)')
axs[0, 1].set_ylabel('Mass (kg)')
# axs[0, 1].set_xlim([min(lim_RW)-3, max(lim_RW)+3])
# axs[0, 1].set_ylim([min(masas_RW)-0.5, max(masas_RW)+0.1])
axs[0, 1].legend()
axs[0, 1].grid(True)

# === 3. Sun Sensors ===
# Mass
axs[1, 0].scatter(sigmas_ss, masas_ss, label='Original Mass Data')
axs[1, 0].plot(x_fit_ss, y_mass_ss, label=f'Fit: $y = {a_mass_ss:.2e} x^{{{b_mass_ss:.2f}}}$', color='green')
axs[1, 0].set_xlabel('Sun sensor σ (°)')
axs[1, 0].set_ylabel('Mass (kg)')
axs[1, 0].set_xlim([min(sigmas_ss)-0.1, max(sigmas_ss)+0.1])
axs[1, 0].set_ylim([min(masas_ss)-0.1, max(masas_ss)+0.1])
axs[1, 0].legend()
axs[1, 0].grid(True)

# === 4. Magnetometers ===
# Mass
axs[1, 1].scatter(sigmas_b, masas_b, label='Original Mass Data')
axs[1, 1].plot(x_fit_b, y_mass_b, label=f'Fit: $y = {a_mass_b:.2e} x^{{{b_mass_b:.2f}}}$', color='green')
axs[1, 1].set_xlabel('Magnetometer σ (T)')
axs[1, 1].set_ylabel('Mass (kg)')
axs[1, 1].set_xlim([min(sigmas_b)-0.5e-9, max(sigmas_b)+0.5e-9])
axs[1, 1].set_ylim([min(masas_b)-0.05, max(masas_b)+0.05])
axs[1, 1].legend()
axs[1, 1].grid(True)

plt.tight_layout()
plt.savefig('physical_components_Mass.pdf', format='pdf')
plt.show()
