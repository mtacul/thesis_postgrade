# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 15:26:52 2024

@author: nachi
"""

#%%
from Suite_LQR import suite_sim
from scipy.optimize import minimize

# Función para escribir en archivo .txt
def save_to_txt(x, acc, psd, time, pot_b, masa_b, vol_b, pot_ss, masa_ss, vol_ss, pot_act, masa_act, vol_act, P_i, file_name):
    with open(file_name, 'a') as f:
        # Escribir valores de x y resultados de la simulación
        f.write(f"x: {x}\n")
        f.write(f"acc: {acc}, psd: {psd}, time: {time}\n")
        f.write(f"pot_b: {pot_b}, masa_b: {masa_b}, vol_b: {vol_b}\n")
        f.write(f"pot_ss: {pot_ss}, masa_ss: {masa_ss}, vol_ss: {vol_ss}\n")
        f.write(f"pot_act: {pot_act}, masa_act: {masa_act}, vol_act: {vol_act}\n")
        f.write(f"Pesos: {P_i}\n")
        f.write(f"------------------------\n")
        
# Definir la función objetivo con dos argumentos
def objective(x, *args):
    # Desempaquetar los argumentos adicionales
    type_act, S_A_both, type_rend,P_i,filename = args  
    
    # Determinar los valores de std_sensor_sol, std_magnetometros y lim según S_A_both
    if S_A_both == 0:
        std_sensor_sol, std_magnetometros = x
        P_i = [1,1,1,
               1,1,0, 
               1,1,0,   
               0,0,0]
        if type_act == 0:
            lim = 1
        elif type_act == 1:
            lim = 0.1
    elif S_A_both == 1:
        lim = x
        std_sensor_sol, std_magnetometros = 0.68, 1e-9  # Valores fijos o iniciales
        
        P_i = [1,1,1,
               0,0,0, 
               0,0,0,   
               1,1,0]
        
    elif S_A_both == 2:
        std_sensor_sol, std_magnetometros, lim = x
        P_i = [1,1,1,
               1,1,0, 
               1,1,0,   
               1,1,0]
    # Invocar la simulación según type_act
    acc, psd, time,pot_b, masa_b, vol_b,pot_ss, masa_ss, vol_ss, pot_act, masa_act, vol_act = suite_sim(std_sensor_sol, std_magnetometros, lim, type_act, S_A_both)
    
    # Guardar los resultados en un archivo .txt
    save_to_txt(x, acc, psd, time, pot_b, masa_b, vol_b, pot_ss, masa_ss, vol_ss, pot_act, masa_act, vol_act,P_i,filename)

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


# Iterar sobre diferentes combinaciones de parámetros
type_act_values = [0, 1]  # Por ejemplo: magnetorquer(0) o rueda de reacción(1)
# type_act_values = [0]  
S_A_both_values = [0, 1, 2]  # Diferentes combinaciones de sensor y actuador 0:sensor, 1:actuador, 2: ambos
# S_A_both_values = [2]  
# type_rend_values = ['acc', 'time', 'acc_time', 'acc_psd', 'psd_time','all']  # Diferentes tipos de rendimiento
# type_rend_values = ['acc_psd','psd_time','all'] 
type_rend_values = ['acc', 'time', 'acc_time','all'] 

# hacer 'psd' por separado

# Pesos para la optimización
P_i = [1,1,1,   #Pesos: 0,1,2 -> acc,psd,time
       1,1,0,   #Pesos: 3,4,5 -> pot, masa y vol magnetometro
       1,1,0,   #Pesos: 6,7,8 -> pot, masa y vol sensor de sol
       1,1,0]   #Pesos: 9,10,11 -> pot, masa y vol actuador

# Bucle para probar diferentes configuraciones
for type_act in type_act_values:
    for S_A_both in S_A_both_values:
        for type_rend in type_rend_values:
            # Crear un nombre de archivo único para cada configuración
            filename = f"resultados_typeact{type_act}_saboth{S_A_both}_typerend{type_rend}.txt"

            # Definir límites y condiciones iniciales según los valores actuales
            if S_A_both == 0:
                bnds = ((0.01, 1.67), (0.012e-9, 3e-9))
                initial_guess = [0.5, 1.5e-9]
            elif S_A_both == 1:
                if type_act == 0:
                    bnds = ((0.29, 70),)
                    initial_guess = [1]
                elif type_act == 1:
                    bnds = ((0.001, 0.25),)
                    initial_guess = [0.10]
            elif S_A_both == 2:
                if type_act == 0:
                    bnds = ((0.01, 1.67), (0.012e-9, 3e-9), (0.29, 70))  # Para std_sensor_sol, std_magnetometros y lim
                    initial_guess = [0.68, 1e-9, 1]  # std_sensor_sol, std_magnetometros y lim

                elif type_act == 1:
                    bnds = ((0.01, 1.67), (0.012e-9, 3e-9), (0.001, 0.25))  # Para std_sensor_sol, std_magnetometros y lim
                    initial_guess = [0.68, 1e-9, 0.1]  # std_sensor_sol, std_magnetometros y lim


            # Ejecutar la optimización
            result = minimize(objective, initial_guess, args=(type_act, S_A_both, type_rend, P_i, filename), method='L-BFGS-B', bounds=bnds)

            # Imprimir resultados
            print(f"Optimización completada para type_act={type_act}, S_A_both={S_A_both}, type_rend={type_rend}")
            print(f"x óptimo: {result.x}")
            print(f"Valor mínimo de la función objetivo: {result.fun}\n")
