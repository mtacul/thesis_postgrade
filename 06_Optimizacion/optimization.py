# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 14:30:38 2024

@author: nachi
# """

from Suite_LQR import suite_sim
from scipy.optimize import minimize

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
        
def save_res_txt(x,funi,file_name,text):
    with open(file_name, 'a') as f:
        # Escribir valores de x y resultados de la simulación
        f.write(f"x {text}: {x}\n")
        f.write(f"f {text}: {funi}\n")
        f.write(f"------------------------\n")
        
# Definir la función objetivo con dos argumentos
def objective(x, *args):
    # Desempaquetar los argumentos adicionales
    type_act, S_A_both, type_rend,P_i,filename = args[:5]  
    
    # Determinar los valores de std_sensor_sol, std_magnetometros y lim según S_A_both
    if S_A_both == 0:
        std_sensor_sol, std_magnetometros = x
        P_i = [1,1,1,
               1,1,0, 
               1,1,0,   
               0,0,0]
        lim = args[5]
        
    elif S_A_both == 1:
        lim = x
        std_sensor_sol = args[5]
        std_magnetometros = args[6]  # Valores fijos o iniciales
        
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
    

    # Seleccionar el valor a retornar según type_rend
    if type_rend == 'acc':   
        funi = P_i[0]*acc**2 + P_i[3]*pot_b**2 + P_i[4]*masa_b**2 + P_i[5]*vol_b**2 +  P_i[6]*pot_ss**2 + P_i[7]*masa_ss**2 + P_i[8]*vol_ss**2 + P_i[9]*pot_act**2 + P_i[10]*masa_act**2 + P_i[11]*vol_act**2 
        # Guardar los resultados en un archivo .txt
        save_to_txt(x, acc, psd, time, pot_b, masa_b, vol_b, pot_ss, masa_ss, vol_ss, pot_act, masa_act, vol_act,P_i,funi,filename)
        return funi
    
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
    
    
file_result = "optimizacion.txt"
   
# Definir valores de type_act, S_A_both, type_rend y Pesos P_i
type_act = 1  # 0: magnetorquer, 1: rueda de reacción
S_A_both = 0  # 0: solo sensor, 1: solo actuador, 2: ambos
type_rend = 'acc'  # Puede ser 'acc', 'psd', 'time', 'acc_time', 'acc_psd', 'psd_time' y 'all'

P_i = [1,1,1,   #Pesos: 0,1,2 -> acc,psd,time
       1,1,0,   #Pesos: 3,4,5 -> pot, masa y vol magnetometro
       1,1,0,   #Pesos: 6,7,8 -> pot, masa y vol sensor de sol
       1,1,1]   #Pesos: 9,10,11 -> pot, masa y vol actuador

filename = f"resultados_typeact{type_act}_saboth{S_A_both}_typerend{type_rend}.txt"


# Definir límites y condiciones iniciales según los valores actuales
if S_A_both == 0:
    # Ingreso de valor para `lim`
    if type_act == 0:
        lim = float(input("Ingrese el valor de 'lim' de magnetorquer de 0.29 a 70: "))
    elif type_act == 1:
        lim = float(input("Ingrese el valor de 'lim' de rueda de reacción de 0.001 a 0.25: "))
        
    bnds = ((0.01, 1.67), (0.012e-9, 3e-9))
    initial_guess = [0.5, 1.5e-9]   
    # Definición de argumentos para la optimización
    argss = (type_act, S_A_both, type_rend, P_i, filename, lim)  
    # Ejecución de la optimización
    result = minimize(objective, initial_guess, args=argss, method='SLSQP', bounds=bnds)
    # Ejecutar la optimización
    save_res_txt(result.x, result.fun, file_result, filename)
elif S_A_both == 1:
    # Ingreso de valores para `std_sensor_sol` y `std_magnetometros`
    std_sensor_sol = float(input("Ingrese el valor de 'std_sensor_sol': "))
    std_magnetometros = float(input("Ingrese el valor de 'std_magnetometros': "))
    if type_act == 0:
        bnds = ((0.29, 70),)
        initial_guess = [1]
    elif type_act == 1:
        bnds = ((0.001, 0.25),)
        initial_guess = [0.10]
    # Definición de argumentos para la optimización
    argss = (type_act, S_A_both, type_rend, P_i, filename, std_sensor_sol, std_magnetometros) 
    # Ejecución de la optimización
    result = minimize(objective, initial_guess, args=argss, method='SLSQP', bounds=bnds)
    # Ejecutar la optimización
    save_res_txt(result.x, result.fun, file_result, filename)
    
elif S_A_both == 2:
    if type_act == 0:
        bnds = ((0.01, 1.67), (0.012e-9, 3e-9), (0.29, 70))  # Para std_sensor_sol, std_magnetometros y lim
        initial_guess = [0.5, 1.5e-9, 1]  # std_sensor_sol, std_magnetometros y lim

    elif type_act == 1:
        bnds = ((0.01, 1.67), (0.012e-9, 3e-9), (0.001, 0.25))  # Para std_sensor_sol, std_magnetometros y lim
        initial_guess = [0.5, 1.5e-9, 0.1]  # std_sensor_sol, std_magnetometros y lim
    # Ejecución de la optimización
    result = minimize(objective, initial_guess, args=(type_act, S_A_both, type_rend, P_i, filename), method='SLSQP', bounds=bnds)
    # Ejecutar la optimización
    save_res_txt(result.x, result.fun, file_result, filename)


# Imprimir los resultados de la optimización
print("Optimización completada.")
print("x óptimo:", result.x)
print("Valor mínimo de la función objetivo:", result.fun)



