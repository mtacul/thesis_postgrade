# -*- coding: utf-8 -*-
"""
Created on Sun May 12 20:11:52 2024

@author: nachi
"""

import numpy as np
from scipy.optimize import minimize

def eigenvalue_constraint(x, A, B):
    K = np.hstack([np.diag(x[:3]), np.diag(x[3:])])  # Asegúrate de que esto tiene sentido según tus matrices A y B.
    A_prim = A - B @ K
    eigenvalues = np.linalg.eigvals(A_prim)
    c = np.real(eigenvalues) + 0.0005
    return c  # Esto debería ser un array numpy de restricciones.


def objective_function(x):
    return np.sqrt(np.sum(x**2))

A = np.array([[0, 0, 0, 0.5, 0, 0],
              [0, 0, 0, 0, 0.5, 0],
              [0, 0, 0, 0, 0, 0.5],
              [1.59414e-08, 0, 0, 0, 0, 0],
              [0, -4.78242e-07, 0, 0, 0, 0.00146229],
              [0, 0, 0, 0, -0.000281447, 0]])

B = np.array([[0, 0, 0],
              [0, 0, 0],
              [0, 0, 0],
              [-0.00126442, 4.18906e-06, 4.34834e-05],
              [4.30543e-06, -0.00128915, 0.000124043],
              [0.000268148, 0.000744258, -8.10068e-05]])

x0 = np.array([-0.0261, 0.0027, -0.6306, -17.2535, -10.6473, 138.2978])


# Modify the constraints to include A and B
constraints = {'type': 'ineq', 'fun': eigenvalue_constraint, 'args': (A, B)}

# Continue with your minimization as before
results = []
num_minimos = 200
for i in range(num_minimos):
    delta_x = np.random.randn(len(x0))
    res = minimize(objective_function, x0 + delta_x, method='SLSQP', constraints=[constraints], options={'disp': False})
    results.append(res.x)
    if i % 10 == 0:
        print(f'Solution {i+1}: {res.x}')


# Calcula y muestra los valores propios para cada solución:
eigenvalues_results = []
for x_opt in results:
    K = np.hstack([np.diag(x_opt[:3]), np.diag(x_opt[3:])])
    A_prim = A - B @ K
    eigenvalues = np.linalg.eigvals(A_prim)
    eigenvalues_results.append(eigenvalues)

for i, eigvals in enumerate(eigenvalues_results):
    print(f'Eigenvalues for solution {i+1}: {eigvals}')
    
#%%

import numpy as np
from scipy.optimize import minimize

# Variables globales para manejar la solución
optimal_x = None
found_solution = False

def eigenvalue_constraint(x, A, B):
    global found_solution, optimal_x
    K = np.hstack([np.diag(x[:3]), np.diag(x[3:])])  # Crear matriz de control K
    A_prim = A - B @ K
    eigenvalues = np.linalg.eigvals(A_prim)
    c = np.real(eigenvalues) + 0.0005

    # Verificar si todos los valores propios son negativos y si es así, indicar que encontramos una solución
    if np.all(np.real(eigenvalues) < 0):
        found_solution = True
        optimal_x = x  # Guardar la solución que produce todos valores propios negativos
        raise StopIteration("Found a solution with all negative eigenvalues.")  # Lanzar una excepción para detener la optimización

    return c

def objective_function(x):
    return np.sqrt(np.sum(x**2))

A = np.array([
    [0, 0, 0, 0.5, 0, 0],
    [0, 0, 0, 0, 0.5, 0],
    [0, 0, 0, 0, 0, 0.5],
    [1.59414e-08, 0, 0, 0, 0, 0],
    [0, -4.78242e-07, 0, 0, 0, 0.00146229],
    [0, 0, 0, 0, -0.000281447, 0]
])

B = np.array([
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [-0.00126442, 4.18906e-06, 4.34834e-05],
    [4.30543e-06, -0.00128915, 0.000124043],
    [0.000268148, 0.000744258, -8.10068e-05]
])

x0 = np.array([-0.0261, 0.0027, -0.6306, -17.2535, -10.6473, 138.2978])
constraints = {'type': 'ineq', 'fun': eigenvalue_constraint, 'args': (A, B)}

try:
    res = minimize(objective_function, x0, method='SLSQP', constraints=[constraints])
except StopIteration as e:
    print(e)

if found_solution:
    print("Optimal solution found with all negative eigenvalues:", optimal_x)
else:
    print("No solution found with all negative eigenvalues.")

