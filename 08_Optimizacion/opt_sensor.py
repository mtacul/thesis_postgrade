# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 19:36:32 2024

@author: nachi
"""
from Suite_LQR_MT import suite_sim
import numpy as np
from scipy.optimize import minimize

# Inserte el accuracy como norma deseado:
input_acc = input("Inserte la norma del accuracy deseado:")

# Variables globales para manejar la solución
found_solution = False
optimal_x = None
optimal_acc = None  # Nueva variable para almacenar el valor de acc óptimo

def objective(x):
    global found_solution, optimal_x, optimal_acc

    if found_solution:
        return 0  # Si ya se encontró la solución, devolver 0 para evitar más cálculos
    
    std_sensor_sol, std_magnetometros = x
    acc = suite_sim(std_sensor_sol, std_magnetometros)
    # print(f"Current accuracy: {acc}")

    if 0 < acc < float(input_acc):
        found_solution = True
        optimal_x = x  # Guardar la solución
        optimal_acc = acc  # Guardar el valor de acc óptimo
        raise StopIteration("Se encontró una solución con la norma de exactitud deseada")  # Detener la optimización

    return acc**2  # Minimizar el cuadrado de la exactitud

def constraint_sensor_sol_lower(x):
    std_sensor_sol, std_magnetometros = x
    # a = 0.033  # Valor mínimo permitido para std_sensor_sol
    a = 0
    return std_sensor_sol - a  # std_sensor_sol debe ser mayor o igual a 'a'

def constraint_sensor_sol_upper(x):
    std_sensor_sol, std_magnetometros = x
    b = 0.833  # Valor máximo permitido para std_sensor_sol
    return b - std_sensor_sol  # std_sensor_sol debe ser menor o igual a 'b'

def constraint_magnetometros_lower(x):
    std_sensor_sol, std_magnetometros = x
    # c = 0.012e-9  # Valor mínimo permitido para std_magnetometros
    c = 0
    return std_magnetometros - c  # std_magnetometros debe ser mayor o igual a 'c'

def constraint_magnetometros_upper(x):
    std_sensor_sol, std_magnetometros = x
    d = 3e-9  # Valor máximo permitido para std_magnetometros
    return d - std_magnetometros  # std_magnetometros debe ser menor o igual a 'd'

# Definir las restricciones en el formato requerido por 'minimize'
constraints = [
    {'type': 'ineq', 'fun': constraint_sensor_sol_lower},
    {'type': 'ineq', 'fun': constraint_sensor_sol_upper},
    {'type': 'ineq', 'fun': constraint_magnetometros_lower},
    {'type': 'ineq', 'fun': constraint_magnetometros_upper}
]

# Valores iniciales para las desviaciones estándar
initial_guess = [0.68, 5e-9]  # primero ss y luego magn

# Ejecutar la optimización con múltiples intentos para encontrar una solución válida
for i in range(1000):
    if i == 0:
        # En la primera iteración, usar el initial_guess
        current_x0 = np.array(initial_guess)
    else:
        # En iteraciones subsiguientes, usar el current_x0 ajustado
        random_adjustment = np.array([
            -np.random.rand() * 0.1,     # Número aleatorio entre 0 y 0.1 para el primer valor
            -np.random.rand() * 1e-10    # Número aleatorio en el orden de 10^-10 para el segundo valor
        ])
        
        current_x0 = current_x0 + random_adjustment
        
        # Limitar los valores para que no sean menores que las restricciones mínimas
        current_x0 = np.array([
            max(current_x0[0], 0),   # std_sensor_sol no puede ser menor que 0
            max(current_x0[1], 0)    # std_magnetometros no puede ser menor que 0
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
    print("Desviación estándar del sensor de sol:", optimal_x[0])
    print("Desviación estándar del magnetómetro:", optimal_x[1])
    print("Norma exactitud alcanzada:", optimal_acc)  # Usar el valor almacenado de acc
else:
    print("No se encontró una solución que cumpla con los criterios deseados.")