# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 20:14:06 2024

@author: nachi
"""

import pyomo.environ as pyo
from Suite_LQR_pyomo import suite_sim_pyomo
import numpy as np

# Función para escribir en archivo .txt
def save_to_txt(x, acc, psd, time, pot_b, masa_b, vol_b, pot_ss, masa_ss, vol_ss, pot_act, masa_act, vol_act, P_i, funi, file_name):
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

# Guardar resultados en un archivo
def save_res_txt(filename, model):
    with open(filename, 'w') as f:
        f.write(f"Valor óptimo de x: [{model.std_sensor_sol.value}, {model.std_magnetometros.value}, {model.lim.value}]\n")
        f.write(f"Valor de la función objetivo: {model.objective.expr()}\n")

def objective_rule(model):
    # Llamar a la simulación con los valores de las variables
    if model.S_A_both == 0:
        acc, psd, time, pot_b, masa_b, vol_b, pot_ss, masa_ss, vol_ss, pot_act, masa_act, vol_act = suite_sim_pyomo(model)
        # acc, psd, time, pot_b, masa_b, vol_b, pot_ss, masa_ss, vol_ss, pot_act, masa_act, vol_act = suite_sim_pyomo(model)
    elif model.S_A_both == 1:
        acc, psd, time, pot_b, masa_b, vol_b, pot_ss, masa_ss, vol_ss, pot_act, masa_act, vol_act = suite_sim_pyomo(model)
        # acc, psd, time, pot_b, masa_b, vol_b, pot_ss, masa_ss, vol_ss, pot_act, masa_act, vol_act = suite_sim_pyomo(model)
    elif model.S_A_both == 2:
        acc, psd, time, pot_b, masa_b, vol_b, pot_ss, masa_ss, vol_ss, pot_act, masa_act, vol_act = suite_sim_pyomo(model)
        # acc, psd, time, pot_b, masa_b, vol_b, pot_ss, masa_ss, vol_ss, pot_act, masa_act, vol_act = suite_sim_pyomo(model)

    # Selección de tipo de rendimiento
    if model.type_rend == 0:   
        funi = model.P_i[0]*acc**2 + model.P_i[3]*pot_b**2 + model.P_i[4]*masa_b**2 + model.P_i[5]*vol_b**2 +  \
                model.P_i[6]*pot_ss**2 + model.P_i[7]*masa_ss**2 + model.P_i[8]*vol_ss**2 + \
                model.P_i[9]*pot_act**2 + model.P_i[10]*masa_act**2 + model.P_i[11]*vol_act**2 
    elif model.type_rend == 1:
        funi = model.P_i[1]*psd**2 + model.P_i[3]*pot_b**2 + model.P_i[4]*masa_b**2 + model.P_i[5]*vol_b**2 +  \
                model.P_i[6]*pot_ss**2 + model.P_i[7]*masa_ss**2 + model.P_i[8]*vol_ss**2 + \
                model.P_i[9]*pot_act**2 + model.P_i[10]*masa_act**2 + model.P_i[11]*vol_act**2
    elif model.type_rend == 2:
        funi = model.P_i[2]*time**2 + model.P_i[3]*pot_b**2 + model.P_i[4]*masa_b**2 + model.P_i[5]*vol_b**2 +  \
                model.P_i[6]*pot_ss**2 + model.P_i[7]*masa_ss**2 + model.P_i[8]*vol_ss**2 + \
                model.P_i[9]*pot_act**2 + model.P_i[10]*masa_act**2 + model.P_i[11]*vol_act**2
    elif model.type_rend == 3:
        funi = model.P_i[0]*acc**2 + model.P_i[2]*time**2 + model.P_i[3]*pot_b**2 + model.P_i[4]*masa_b**2 + \
                model.P_i[5]*vol_b**2 + model.P_i[6]*pot_ss**2 + model.P_i[7]*masa_ss**2 + \
                model.P_i[8]*vol_ss**2 + model.P_i[9]*pot_act**2 + model.P_i[10]*masa_act**2 + model.P_i[11]*vol_act**2
    elif model.type_rend == 4:
        funi = model.P_i[0]*acc**2 + model.P_i[1]*psd**2 + model.P_i[3]*pot_b**2 + model.P_i[4]*masa_b**2 + \
                model.P_i[5]*vol_b**2 + model.P_i[6]*pot_ss**2 + model.P_i[7]*masa_ss**2 + \
                model.P_i[8]*vol_ss**2 + model.P_i[9]*pot_act**2 + model.P_i[10]*masa_act**2 + model.P_i[11]*vol_act**2
    elif model.type_rend == 5:
        funi = model.P_i[1]*psd**2 + model.P_i[2]*time**2 + model.P_i[3]*pot_b**2 + model.P_i[4]*masa_b**2 + \
                model.P_i[5]*vol_b**2 + model.P_i[6]*pot_ss**2 + model.P_i[7]*masa_ss**2 + \
                model.P_i[8]*vol_ss**2 + model.P_i[9]*pot_act**2 + model.P_i[10]*masa_act**2 + model.P_i[11]*vol_act**2
    elif model.type_rend == 6:
        funi = model.P_i[0]*acc**2 + model.P_i[1]*psd**2 + model.P_i[2]*time**2 + model.P_i[3]*pot_b**2 + \
                model.P_i[4]*masa_b**2 + model.P_i[5]*vol_b**2 + model.P_i[6]*pot_ss**2 + model.P_i[7]*masa_ss**2 + \
                model.P_i[8]*vol_ss**2 + model.P_i[9]*pot_act**2 + model.P_i[10]*masa_act**2 + model.P_i[11]*vol_act**2

    # Guardar los resultados en un archivo .txt
    save_to_txt([model.std_sensor_sol.value, model.std_magnetometros.value, model.lim.value], acc, psd, time, pot_b, masa_b, vol_b, pot_ss, masa_ss, vol_ss, pot_act, masa_act, vol_act, model.P_i, funi, "optimizacion.txt")
    
    return funi


# def con_rule_1(model):
#     if model.S_A_both == 0:
#         model.std_sensor_sol
#     elif model.S_A_both == 1:
#         acc, psd, time, pot_b, masa_b, vol_b, pot_ss, masa_ss, vol_ss, pot_act, masa_act, vol_act = suite_sim_pyomo(
#             model.std_sensor_sol, model.std_magnetometros, model.lim.value, model.type_act, model.S_A_both)
#     elif model.S_A_both == 2:
#         acc, psd, time, pot_b, masa_b, vol_b, pot_ss, masa_ss, vol_ss, pot_act, masa_act, vol_act = suite_sim_pyomo(
#             model.std_sensor_sol.value, model.std_magnetometros.value, model.lim.value, model.type_act, model.S_A_both)
    

model = pyo.ConcreteModel()

# Definir valores de type_act, S_A_both, type_rend y Pesos P_i
type_act = 0  # 0: magnetorquer, 1: rueda de reacción
S_A_both = 0  # 0: solo sensor, 1: solo actuador, 2: ambos
type_rend = 0 # Puede ser 0:'acc', 1:'psd', 2:'time', 3:'acc_time', 4:'acc_psd', 5:'psd_time' y 6:'all'

model.indices_P_i = pyo.Set(initialize=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])


# Definir las variables de decisión según tu problema
if S_A_both == 0:
    model.std_sensor_sol = pyo.Var(within=pyo.NonNegativeReals, bounds=(0.01, 1.67), initialize=0.6)
    model.std_magnetometros = pyo.Var(within=pyo.NonNegativeReals, bounds=(0.012e-9, 3e-9), initialize=2e-9)
    if type_act == 0:
        model.lim = pyo.Param(initialize=5, mutable=False)
    if type_act == 1:
        model.lim = pyo.Param(initialize = 0.1, mutable=False)
    # P_i = [1, 1, 1,  # Pesos: 0,1,2 -> acc, psd, time
    #         1, 1, 0,  # Pesos: 3,4,5 -> pot, masa y vol magnetómetro
    #         1, 1, 0,  # Pesos: 6,7,8 -> pot, masa y vol sensor de sol
    #         0, 0, 0]  # Pesos: 9,10,11 -> pot, masa y vol actuador
    model.P_i = pyo.Param(model.indices_P_i, initialize={0: 1, 1: 1, 2: 1,
                                                     3: 1, 4: 1, 5: 0,
                                                     6: 1, 7: 1, 8: 0,
                                                     9: 0, 10: 0, 11: 0}, mutable=True)    
    
elif S_A_both == 1:
    model.std_sensor_sol = pyo.Param(initialize=0.5, mutable=False)
    model.std_magnetometros = pyo.Param(initialize=1.5e-9, mutable=False)
    if type_act ==0:
        model.lim = pyo.Var(within=pyo.NonNegativeReals, bounds=(0.29, 70), initialize=5)
    elif type_act == 1:
        model.lim = pyo.Var(within=pyo.NonNegativeReals, bounds=(0.001, 0.25), initialize=0.10)
    # P_i = [1, 1, 1,  # Pesos: 0,1,2 -> acc, psd, time
    #         0, 0, 0,  # Pesos: 3,4,5 -> pot, masa y vol magnetómetro
    #         0, 0, 0,  # Pesos: 6,7,8 -> pot, masa y vol sensor de sol
    #         1, 1, 0]  # Pesos: 9,10,11 -> pot, masa y vol actuador
    model.P_i = pyo.Param(model.indices_P_i, initialize={0: 1, 1: 1, 2: 1,
                                                     3: 0, 4: 0, 5: 0,
                                                     6: 0, 7: 0, 8: 0,
                                                     9: 1, 10: 1, 11: 0}, mutable=True)
elif S_A_both == 2:
    model.std_sensor_sol = pyo.Var(within=pyo.NonNegativeReals, bounds=(0.01, 1.67), initialize=0.5)
    model.std_magnetometros = pyo.Var(within=pyo.NonNegativeReals, bounds=(0.012e-9, 3e-9), initialize=1.5e-9)
    if type_act ==0:
        model.lim = pyo.Var(within=pyo.NonNegativeReals, bounds=(0.29, 70), initialize=5)
    elif type_act == 1:
        model.lim = pyo.Var(within=pyo.NonNegativeReals, bounds=(0.001, 0.25), initialize=0.10)
    # P_i = [1, 1, 1,  # Pesos: 0,1,2 -> acc, psd, time
    #         1, 1, 0,  # Pesos: 3,4,5 -> pot, masa y vol magnetómetro
    #         1, 1, 0,  # Pesos: 6,7,8 -> pot, masa y vol sensor de sol
    #         1, 1, 0]  # Pesos: 9,10,11 -> pot, masa y vol actuador
    model.P_i = pyo.Param(model.indices_P_i, initialize={0: 1, 1: 1, 2: 1,
                                                     3: 1, 4: 1, 5: 0,
                                                     6: 1, 7: 1, 8: 0,
                                                     9: 1, 10: 1, 11: 0}, mutable=True)

    
model.type_act = pyo.Param(initialize=type_act, mutable=False)
model.S_A_both = pyo.Param(initialize=S_A_both, mutable=False)
model.type_rend = pyo.Param(initialize=type_rend, mutable=False)

# Definición de la función objetivo
model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

# Selección y ejecución del solver
solver = pyo.SolverFactory('baron', executable='C:\\baron\\baron.exe')
# solver.options['NLPSubSolver'] = 'IPOPT'
result = solver.solve(model, tee=True)
print(result)
# print(result.fun)
# Guardar resultados
save_res_txt("resultados.txt", model)
