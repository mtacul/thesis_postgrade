# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 14:30:38 2024

@author: nachi
# """
#%%
from Suite_LQR import suite_sim
from scipy.optimize import minimize

# Función para escribir en archivo .txt
def save_to_txt(x, acc, psd, time, pot_b, masa_b, vol_b, pot_ss, masa_ss, vol_ss, pot_act, masa_act, vol_act, file_name="resultados_optimizacion.txt"):
    with open(file_name, 'a') as f:
        # Escribir valores de x y resultados de la simulación
        f.write(f"x: {x}\n")
        f.write(f"acc: {acc}, psd: {psd}, time: {time}\n")
        f.write(f"pot_b: {pot_b}, masa_b: {masa_b}, vol_b: {vol_b}\n")
        f.write(f"pot_ss: {pot_ss}, masa_ss: {masa_ss}, vol_ss: {vol_ss}\n")
        f.write(f"pot_act: {pot_act}, masa_act: {masa_act}, vol_act: {vol_act}\n")
        f.write(f"------------------------\n")
        
# Definir la función objetivo con dos argumentos
def objective(x, *args):
    # Desempaquetar los argumentos adicionales
    type_act, S_A_both, type_rend,P_i = args  
    
    # Determinar los valores de std_sensor_sol, std_magnetometros y lim según S_A_both
    if S_A_both == 0:
        std_sensor_sol, std_magnetometros = x
        P_i[9] = 0
        P_i[10] = 0 
        P_i[11] = 0
        if type_act == 0:
            lim = 5
        elif type_act == 1:
            lim = 0.1255
    elif S_A_both == 1:
        lim = x
        std_sensor_sol, std_magnetometros = 0.68, 1e-9  # Valores fijos o iniciales
        for i in range(3, 9):
            P_i[i] = 0
        
    elif S_A_both == 2:
        std_sensor_sol, std_magnetometros, lim = x

    # Invocar la simulación según type_act
    acc, psd, time,pot_b, masa_b, vol_b,pot_ss, masa_ss, vol_ss, pot_act, masa_act, vol_act = suite_sim(std_sensor_sol, std_magnetometros, lim, type_act, S_A_both)
    
    # Guardar los resultados en un archivo .txt
    save_to_txt(x, acc, psd, time, pot_b, masa_b, vol_b, pot_ss, masa_ss, vol_ss, pot_act, masa_act, vol_act)

    # Seleccionar el valor a retornar según type_rend
    if type_rend == 'acc':
        return P_i[0]*acc**2 + P_i[3]*pot_b**2 + P_i[4]*masa_b**2 + P_i[5]*vol_b**2 +  P_i[6]*pot_ss**2 + P_i[7]*masa_ss**2 + P_i[8]*vol_ss**2 + P_i[9]*pot_act**2 + P_i[10]*masa_act**2 + P_i[11]*vol_act**2 
    elif type_rend == 'psd':
        return P_i[1]*psd**2 + P_i[3]*pot_b**2 + P_i[4]*masa_b**2 + P_i[5]*vol_b**2 +  P_i[6]*pot_ss**2 + P_i[7]*masa_ss**2 + P_i[8]*vol_ss**2 + P_i[9]*pot_act**2 + P_i[10]*masa_act**2 + P_i[11]*vol_act**2
    elif type_rend == 'time':
        return P_i[2]*time**2 + P_i[3]*pot_b**2 + P_i[4]*masa_b**2 + P_i[5]*vol_b**2 +  P_i[6]*pot_ss**2 + P_i[7]*masa_ss**2 + P_i[8]*vol_ss**2 + P_i[9]*pot_act**2 + P_i[10]*masa_act**2 + P_i[11]*vol_act**2
    elif type_rend =='acc_time':
        return P_i[0]*acc**2 + P_i[2]*time**2 + P_i[3]*pot_b**2 + P_i[4]*masa_b**2 + P_i[5]*vol_b**2 +  P_i[6]*pot_ss**2 + P_i[7]*masa_ss**2 + P_i[8]*vol_ss**2 + P_i[9]*pot_act**2 + P_i[10]*masa_act**2 + P_i[11]*vol_act**2
    elif type_rend =='acc_psd':
        return P_i[0]*acc**2 + P_i[1]*psd**2 + P_i[3]*pot_b**2 + P_i[4]*masa_b**2 + P_i[5]*vol_b**2 +  P_i[6]*pot_ss**2 + P_i[7]*masa_ss**2 + P_i[8]*vol_ss**2 + P_i[9]*pot_act**2 + P_i[10]*masa_act**2 + P_i[11]*vol_act**2
    elif type_rend =='psd_time':
        return P_i[1]*psd**2 + P_i[2]*time**2 + P_i[3]*pot_b**2 + P_i[4]*masa_b**2 + P_i[5]*vol_b**2 +  P_i[6]*pot_ss**2 + P_i[7]*masa_ss**2 + P_i[8]*vol_ss**2 + P_i[9]*pot_act**2 + P_i[10]*masa_act**2 + P_i[11]*vol_act**2
    elif type_rend =='all':
        return P_i[0]*acc**2 + P_i[1]*psd**2 + P_i[2]*time**2 + P_i[3]*pot_b**2 + P_i[4]*masa_b**2 + P_i[5]*vol_b**2 +  P_i[6]*pot_ss**2 + P_i[7]*masa_ss**2 + P_i[8]*vol_ss**2 + P_i[9]*pot_act**2 + P_i[10]*masa_act**2 + P_i[11]*vol_act**2

    
# Definir valores de type_act, S_A_both, type_rend y Pesos P_i
type_act = 0  # 0: magnetorquer, 1: rueda de reacción
S_A_both = 2  # 0: solo sensor, 1: solo actuador, 2: ambos
type_rend = 'acc_time'  # Puede ser 'acc', 'psd', 'time', 'acc_time', 'acc_psd', 'psd_time' y 'all'

P_i = [1,1,1,   #Pesos: 0,1,2 -> acc,psd,time
       1,1,0,   #Pesos: 3,4,5 -> pot, masa y vol magnetometro
       1,1,0,   #Pesos: 6,7,8 -> pot, masa y vol sensor de sol
       1,1,1]   #Pesos: 9,10,11 -> pot, masa y vol actuador


# Definir límites para las variables
if S_A_both == 0:
    bnds = ((0.01, 1.67), (0.012e-9, 3e-9))  # Para std_sensor_sol y std_magnetometros
elif S_A_both == 1:
    if type_act == 0:
        bnds = ((0.29, 70),)  # Solo para lim
    elif type_act == 1:
        bnds = ((0.001, 0.25),)
elif S_A_both == 2:
    if type_act == 0:
        bnds = ((0.01, 1.67), (0.012e-9, 3e-9), (0.29, 70))  # Para std_sensor_sol, std_magnetometros y lim
    elif type_act == 1:
        bnds = ((0.01, 1.67), (0.012e-9, 3e-9), (0.001, 0.25))  # Para std_sensor_sol, std_magnetometros y lim

# Valores iniciales para las desviaciones estándar o lim según el caso
if S_A_both == 0:
    initial_guess = [0.68, 1e-9]  # std_sensor_sol y std_magnetometros
elif S_A_both == 1:
    if type_act == 0:
        initial_guess = [5]  # lim
    elif type_act == 1:
        initial_guess = [0.1255]  # lim
elif S_A_both == 2:
    if type_act == 0:
        initial_guess = [0.68, 1e-9, 5]  # std_sensor_sol, std_magnetometros y lim
    elif type_act == 1:
        initial_guess = [0.68, 1e-9, 0.1255]  # std_sensor_sol, std_magnetometros y lim

# Ejecutar la optimización
result = minimize(objective, initial_guess, args=(type_act, S_A_both, type_rend,P_i), method='L-BFGS-B', bounds=bnds, options={'disp': True, 'ftol': 1e-4})

# Imprimir los resultados de la optimización
print("Optimización completada.")
print("x óptimo:", result.x)
print("Valor mínimo de la función objetivo:", result.fun)


