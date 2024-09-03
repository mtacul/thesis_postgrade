# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 19:36:32 2024

@author: nachi
"""
from Suite_LQR_MT import suite_act
import numpy as np
from scipy.optimize import minimize

# Inserte el accuracy como norma deseado:
input_time = input("Inserte la norma del tiempo deseado:")

# Variables globales para manejar la solución
found_solution = False
optimal_x = None
optimal_time = None  # Nueva variable para almacenar el valor de time óptimo

def objective(x):
    global found_solution, optimal_x, optimal_time

    if found_solution:
        return 0  # Si ya se encontró la solución, devolver 0 para evitar más cálculos
    
    lim = x
    time = suite_act(lim)
    # print(f"Current time: {time}")

    if 0 < time < float(input_time):
        found_solution = True
        optimal_x = x  # Guardar la solución
        optimal_time = time # Guardar el valor de time óptimo
        raise StopIteration("Se encontró una solución con la norma de tiempo deseada")  # Detener la optimización

    return time**2  # Minimizar el cuadrado del tiempo

def constraint_magnetorquer_lower(x):
    lim = x
    # a = 0.033  # Valor mínimo permitido para lim
    a = -15
    return lim - a  # lim debe ser mayor o igual a 'a'

def constraint_magnetorquer_upper(x):
    lim = x
    b = 15 # Valor máximo permitido para lim
    return b - lim  # lim debe ser menor o igual a 'b'


# Definir las restricciones en el formato requerido por 'minimize'
constraints = [
    {'type': 'ineq', 'fun': constraint_magnetorquer_lower},
    {'type': 'ineq', 'fun': constraint_magnetorquer_upper}
]

# Valores iniciales para las desviaciones estándar
initial_guess = [0.5]  # primero ss y luego magn

# Ejecutar la optimización con múltiples intentos para encontrar una solución válida
for i in range(1000):
    if i == 0:
        # En la primera iteración, usar el initial_guess
        current_x0 = np.array(initial_guess)
    else:
        # En iteraciones subsiguientes, usar el current_x0 ajustado
        random_adjustment = np.array([
            np.random.rand()     # Número aleatorio entre 0 y 5 para el primer valor
        ])
        
        current_x0 = current_x0 + random_adjustment
        
        # Limitar los valores para que no sean menores que las restricciones mínimas
        current_x0 = np.array([
            max(current_x0[0], 0)   # lim no puede ser menor que 0
        ])

    try:
        result = minimize(objective, current_x0, method='SLSQP', constraints=constraints)
        # Guardar el mejor resultado para la siguiente iteración
        current_x0 = result.x
        print(f"Resultado: {result.x}, Valor objetivo: {result.fun}")
    except StopIteration as e:
        print(e)
        break  # Detener la iteración si se encuentra una solución válida

if found_solution:
    print("Resultado de la optimización:")
    print("Limite de torque del magnetorquer:", optimal_x[0])
    print("Norma del tiempo alcanzada:", optimal_time)  # Usar el valor almacenado de time
else:
    print("No se encontró una solución que cumpla con los criterios deseados.")