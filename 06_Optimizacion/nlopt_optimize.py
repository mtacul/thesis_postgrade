# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 20:24:31 2024

@author: nachi
"""

import nlopt
from Suite_LQR import suite_sim

def save_res_txt(x,funi,file_name,text):
    with open(file_name, 'a') as f:
        # Escribir valores de x y resultados de la simulación
        f.write(f"x {text}: {x}\n")
        f.write(f"f {text}: {funi}\n")
        f.write(f"------------------------\n")
        
# Función para escribir en archivo .txt
def save_to_txt(x, acc, psd, time, pot_b, masa_b, vol_b, pot_ss, masa_ss, vol_ss, pot_act, masa_act, vol_act, P_i, funi,file_name):
    with open(file_name, 'a') as f:
        # Escribir valores de x y resultados de la simulación
        f.write(f"x: {x}\n")
        f.write(f"acc: {acc}, psd: {psd}, time: {time}\n")
        f.write(f"pot_b: {pot_b}, masa_b: {masa_b}, vol_b: {vol_b}\n")
        f.write(f"pot_ss: {pot_ss}, masa_ss: {masa_ss}, vol_ss: {vol_ss}\n")
        f.write(f"pot_act: {pot_act}, masa_act: {masa_act}, vol_act: {vol_act}\n")
        f.write(f"Pesos: {P_i}\n")
        f.write(f"funcion de costo: {funi}\n")
        f.write(f"------------------------\n")
        
        
# Configurar la función de objetivo para NLopt
def objective_nlopt(x, grad, *args):
    # Desempaquetar los argumentos adicionales
    type_act, S_A_both, type_rend, P_i, filename = args[:5]  
    
    if S_A_both == 0:
        std_sensor_sol, std_magnetometros = x
        P_i = [P_i[0], P_i[1], P_i[2], P_i[3], P_i[4], 0, P_i[6], P_i[7], 0, 0, 0, 0]
        lim = args[5]
    elif S_A_both == 1:
        lim = x[0]
        std_sensor_sol = args[5]
        std_magnetometros = args[6]
        P_i = [P_i[0], P_i[1], P_i[2], 0, 0, 0, 0, 0, 0, P_i[9], P_i[10], 0]
    elif S_A_both == 2:
        std_sensor_sol, std_magnetometros, lim = x
        P_i = P_i
    
    # Invocar la simulación
    acc, psd, time, pot_b, masa_b, vol_b, pot_ss, masa_ss, vol_ss, pot_act, masa_act, vol_act = suite_sim(
        std_sensor_sol, std_magnetometros, lim, type_act, S_A_both
    )

    # Calcular la función de costo en función de `type_rend`
    if type_rend == 'acc':
        funi = P_i[0] * acc**2 + P_i[3] * pot_b**2 + P_i[4] * masa_b**2 + P_i[5] * vol_b**2 + \
               P_i[6] * pot_ss**2 + P_i[7] * masa_ss**2 + P_i[8] * vol_ss**2 + \
               P_i[9] * pot_act**2 + P_i[10] * masa_act**2 + P_i[11] * vol_act**2
               # Guardar los resultados en un archivo .txt
        save_to_txt(x, acc, psd, time, pot_b, masa_b, vol_b, pot_ss, masa_ss, vol_ss, pot_act, masa_act, vol_act, P_i, funi, filename)
    
    elif type_rend == 'psd':
        funi = P_i[1]*psd**2 + P_i[3]*pot_b**2 + P_i[4]*masa_b**2 + P_i[5]*vol_b**2 +  P_i[6]*pot_ss**2 + P_i[7]*masa_ss**2 + P_i[8]*vol_ss**2 + P_i[9]*pot_act**2 + P_i[10]*masa_act**2 + P_i[11]*vol_act**2
        # Guardar los resultados en un archivo .txt
        save_to_txt(x, acc, psd, time, pot_b, masa_b, vol_b, pot_ss, masa_ss, vol_ss, pot_act, masa_act, vol_act,P_i,funi,filename)
        return funi
      
    elif type_rend == 'time':
        funi = P_i[2]*time**2 + P_i[3]*pot_b**2 + P_i[4]*masa_b**2 + P_i[5]*vol_b**2 +  P_i[6]*pot_ss**2 + P_i[7]*masa_ss**2 + P_i[8]*vol_ss**2 + P_i[9]*pot_act**2 + P_i[10]*masa_act**2 + P_i[11]*vol_act**2
        # Guardar los resultados en un archivo .txt
        save_to_txt(x, acc, psd, time, pot_b, masa_b, vol_b, pot_ss, masa_ss, vol_ss, pot_act, masa_act, vol_act,P_i,funi,filename)
        return funi

    elif type_rend =='acc_time':
        funi = P_i[0]*acc**2 + P_i[2]*time**2 + P_i[3]*pot_b**2 + P_i[4]*masa_b**2 + P_i[5]*vol_b**2 +  P_i[6]*pot_ss**2 + P_i[7]*masa_ss**2 + P_i[8]*vol_ss**2 + P_i[9]*pot_act**2 + P_i[10]*masa_act**2 + P_i[11]*vol_act**2
        # Guardar los resultados en un archivo .txt
        save_to_txt(x, acc, psd, time, pot_b, masa_b, vol_b, pot_ss, masa_ss, vol_ss, pot_act, masa_act, vol_act,P_i,funi,filename)
        return funi

    elif type_rend =='acc_psd':
        funi = P_i[0]*acc**2 + P_i[1]*psd**2 + P_i[3]*pot_b**2 + P_i[4]*masa_b**2 + P_i[5]*vol_b**2 +  P_i[6]*pot_ss**2 + P_i[7]*masa_ss**2 + P_i[8]*vol_ss**2 + P_i[9]*pot_act**2 + P_i[10]*masa_act**2 + P_i[11]*vol_act**2
        # Guardar los resultados en un archivo .txt
        save_to_txt(x, acc, psd, time, pot_b, masa_b, vol_b, pot_ss, masa_ss, vol_ss, pot_act, masa_act, vol_act,P_i,funi,filename)
        return funi        

    elif type_rend =='psd_time':
        funi = P_i[1]*psd**2 + P_i[2]*time**2 + P_i[3]*pot_b**2 + P_i[4]*masa_b**2 + P_i[5]*vol_b**2 +  P_i[6]*pot_ss**2 + P_i[7]*masa_ss**2 + P_i[8]*vol_ss**2 + P_i[9]*pot_act**2 + P_i[10]*masa_act**2 + P_i[11]*vol_act**2
        # Guardar los resultados en un archivo .txt
        save_to_txt(x, acc, psd, time, pot_b, masa_b, vol_b, pot_ss, masa_ss, vol_ss, pot_act, masa_act, vol_act,P_i,funi,filename)
        return funi  

    elif type_rend =='all':
        funi = P_i[0]*acc**2 + P_i[1]*psd**2 + P_i[2]*time**2 + P_i[3]*pot_b**2 + P_i[4]*masa_b**2 + P_i[5]*vol_b**2 +  P_i[6]*pot_ss**2 + P_i[7]*masa_ss**2 + P_i[8]*vol_ss**2 + P_i[9]*pot_act**2 + P_i[10]*masa_act**2 + P_i[11]*vol_act**2
        # Guardar los resultados en un archivo .txt
        save_to_txt(x, acc, psd, time, pot_b, masa_b, vol_b, pot_ss, masa_ss, vol_ss, pot_act, masa_act, vol_act,P_i,funi,filename)
        return funi      
    return funi

# Configurar la optimización NLopt
def optimize_nlopt(type_act, S_A_both, type_rend, P_i, filename):
    # Definir límites y condiciones iniciales
    if S_A_both == 0:
        if type_act == 0:
            lim = float(input("Ingrese el valor de 'lim' de magnetorquer de 0.29 a 70: "))
            bnds = [(0.01, 1.67), (0.012e-9, 3e-9)]
        elif type_act == 1:
            lim = float(input("Ingrese el valor de 'lim' de rueda de reacción de 0.001 a 0.25: "))
            bnds = [(0.01, 1.67), (0.012e-9, 3e-9)]
        x0 = [0.5, 1.5e-9]
        args = (type_act, S_A_both, type_rend, P_i, filename, lim)
    elif S_A_both == 1:
        std_sensor_sol = float(input("Ingrese el valor de 'std_sensor_sol': "))
        std_magnetometros = float(input("Ingrese el valor de 'std_magnetometros': "))
        if type_act == 0:
            bnds = [(0.29, 70)]
        elif type_act == 1:
            bnds = [(0.001, 0.25)]
        x0 = [1.0]
        args = (type_act, S_A_both, type_rend, P_i, filename, std_sensor_sol, std_magnetometros)
    elif S_A_both == 2:
        if type_act == 0:
            bnds = ((0.01, 1.67), (0.012e-9, 3e-9), (0.29, 70))  # Para std_sensor_sol, std_magnetometros y lim
            x0 = [0.5, 1.5e-9, 1]  # std_sensor_sol, std_magnetometros y lim
            args=(type_act, S_A_both, type_rend, P_i, filename)
        elif type_act == 1:
            bnds = ((0.01, 1.67), (0.012e-9, 3e-9), (0.001, 0.25))  # Para std_sensor_sol, std_magnetometros y lim
            x0 = [0.5, 1.5e-9, 0.1]  # std_sensor_sol, std_magnetometros y lim
            args=(type_act, S_A_both, type_rend, P_i, filename)
   
    # Crear el optimizador NLopt
    opt = nlopt.opt(nlopt.LN_BOBYQA, len(x0))  # Elegir el algoritmo NLopt, en este caso LN_BOBYQA
    opt.set_min_objective(lambda x, grad: objective_nlopt(x, grad, *args))
    
    # Configurar límites
    lower_bounds, upper_bounds = zip(*bnds)
    opt.set_lower_bounds(lower_bounds)
    opt.set_upper_bounds(upper_bounds)
    
    # Configurar tolerancias
    opt.set_xtol_rel(1e-10)
    
    # Ejecutar la optimización
    result = opt.optimize(x0)
    funi = opt.last_optimum_value()
    
    # Guardar el resultado final en el archivo
    save_res_txt(result, funi, "optimizacion_nlopt.txt", filename)
    print(f"Optimización completada. Resultado: {result}, Costo: {funi}")


file_result = "optimizacion.txt"
   
# Definir valores de type_act, S_A_both, type_rend y Pesos P_i
type_act = 0  # 0: magnetorquer, 1: rueda de reacción
S_A_both = 0  # 0: solo sensor, 1: solo actuador, 2: ambos
type_rend = 'acc'  # Puede ser 'acc', 'psd', 'time', 'acc_time', 'acc_psd', 'psd_time' y 'all'

P_i = [1,1,1,   #Pesos: 0,1,2 -> acc,psd,time
       0,0,0,   #Pesos: 3,4,5 -> pot, masa y vol magnetometro
       0,0,0,   #Pesos: 6,7,8 -> pot, masa y vol sensor de sol
       0,0,0]   #Pesos: 9,10,11 -> pot, masa y vol actuador

filename = f"resultados_typeact{type_act}_saboth{S_A_both}_typerend{type_rend}.txt"

# Ejecutar la función de optimización con los parámetros de entrada
optimize_nlopt(type_act, S_A_both, type_rend, P_i, filename)
