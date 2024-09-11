# -*- coding: utf-8 -*-
"""
Created on Sun May 12 20:11:52 2024

@author: nachi
"""

import numpy as np
from scipy.optimize import minimize
import control as ctrl
# #%%

# def eigenvalue_constraint(x, A, B):
#     K = np.hstack([np.diag(x[:3]), np.diag(x[3:])])  # Asegúrate de que esto tiene sentido según tus matrices A y B.
#     A_prim = A - B @ K
#     eigenvalues = np.linalg.eigvals(A_prim)
#     c = abs(eigenvalues) - 0.98
#     return c  # Esto debería ser un array numpy de restricciones.


# def objective_function(x):
#     return np.sqrt(np.sum(x**2))

# deltat = 2
# h = 0.01

# A = np.array([[0, 0, 0, 0.5, 0, 0],
#               [0, 0, 0, 0, 0.5, 0],
#               [0, 0, 0, 0, 0, 0.5],
#               [1.59414e-08, 0, 0, 0, 0, 0],
#               [0, -4.78242e-07, 0, 0, 0, 0.00146229],
#               [0, 0, 0, 0, -0.000281447, 0]])

# B = np.array([[0, 0, 0],
#               [0, 0, 0],
#               [0, 0, 0],
#               [-0.00126442, 4.18906e-06, 4.34834e-05],
#               [4.30543e-06, -0.00128915, 0.000124043],
#               [0.000268148, 0.000744258, -8.10068e-05]])

# C = np.eye(6)  # Assuming a 6x6 identity matrix for C
# D = np.zeros((6, 3))  # Assuming D has the same number of rows as A and the same number of columns as B

# # Create the continuous state-space model
# sys_continuous = ctrl.StateSpace(A, B, C, D)

# # Discretize the system
# sys_discrete = ctrl.c2d(sys_continuous, deltat*h, method='zoh')

# # Extract the discretized A and B matrices
# A_discrete = sys_discrete.A
# B_discrete = sys_discrete.B

# x0 = np.array([-0.0261, 0.0027, -0.6306, -17.2535, -10.6473, 138.2978])


# # Modify the constraints to include A and B
# constraints = {'type': 'ineq', 'fun': eigenvalue_constraint, 'args': (A_discrete, B_discrete)}

# # Continue with your minimization as before
# results = []
# num_minimos = 200
# for i in range(num_minimos):
#     delta_x = np.random.randn(len(x0))
#     res = minimize(objective_function, x0 + delta_x, method='SLSQP', constraints=[constraints], options={'disp': False})
#     results.append(res.x)
#     if i % 10 == 0:
#         print(f'Solution {i+1}: {res.x}')


# # Calcula y muestra los valores propios para cada solución:
# eigenvalues_results = []
# for x_opt in results:
#     K = np.hstack([np.diag(x_opt[:3]), np.diag(x_opt[3:])])
#     A_prim = A - B @ K
#     eigenvalues = np.linalg.eigvals(A_prim)
#     eigenvalues_results.append(eigenvalues)

# for i, eigvals in enumerate(eigenvalues_results):
#     print(f'Eigenvalues for solution {i+1}: {eigvals}')
    
# #%%

# import numpy as np
# from scipy.optimize import minimize

# # Variables globales para manejar la solución
# optimal_x = None
# found_solution = False

# def eigenvalue_constraint(x, A, B):
#     global found_solution, optimal_x
#     K = np.hstack([np.diag(x[:3]), np.diag(x[3:])])  # Crear matriz de control K
#     A_prim = A - B @ K
#     eigenvalues = np.linalg.eigvals(A_prim)
#     c = abs(eigenvalues) - 0.98

#     # Verificar si todos los valores propios son negativos y si es así, indicar que encontramos una solución
#     if np.all(np.real(eigenvalues) < 0):
#         found_solution = True
#         optimal_x = x  # Guardar la solución que produce todos valores propios negativos
#         raise StopIteration("Found a solution with all negative eigenvalues.")  # Lanzar una excepción para detener la optimización

#     return c

# def objective_function(x):
#     return np.sqrt(np.sum(x**2))
# deltat = 2
# h = 0.01

# A = np.array([
#     [0, 0, 0, 0.5, 0, 0],
#     [0, 0, 0, 0, 0.5, 0],
#     [0, 0, 0, 0, 0, 0.5],
#     [1.59414e-08, 0, 0, 0, 0, 0],
#     [0, -4.78242e-07, 0, 0, 0, 0.00146229],
#     [0, 0, 0, 0, -0.000281447, 0]
# ])

# B = np.array([
#     [0, 0, 0],
#     [0, 0, 0],
#     [0, 0, 0],
#     [-0.00126442, 4.18906e-06, 4.34834e-05],
#     [4.30543e-06, -0.00128915, 0.000124043],
#     [0.000268148, 0.000744258, -8.10068e-05]
# ])

# C = np.eye(6)  # Assuming a 6x6 identity matrix for C
# D = np.zeros((6, 3))  # Assuming D has the same number of rows as A and the same number of columns as B

# # Create the continuous state-space model
# sys_continuous = ctrl.StateSpace(A, B, C, D)

# # Discretize the system
# sys_discrete = ctrl.c2d(sys_continuous, deltat*h, method='zoh')

# # Extract the discretized A and B matrices
# A_discrete = sys_discrete.A
# B_discrete = sys_discrete.B

# x0 = [-241.79, 43.35, -3059, -560, -155, -5161];
# constraints = {'type': 'ineq', 'fun': eigenvalue_constraint, 'args': (A_discrete, B_discrete)}

# try:
#     res = minimize(objective_function, x0, method='SLSQP', constraints=[constraints])
# except StopIteration as e:
#     print(e)

# if found_solution:
#     print("Optimal solution found with all negative eigenvalues:", optimal_x)
# else:
#     print("No solution found with all negative eigenvalues.")


#%%

# Variables globales para manejar la solución
optimal_x = None
found_solution = False
eigss = []

def eigenvalue_constraint(x, A, B):
    global found_solution, optimal_x
    K = np.hstack([np.diag(x[:3]), np.diag(x[3:])])  # Crear matriz de control K
    A_prim = A - B @ K
    eigenvalues = np.linalg.eigvals(A_prim)
    c = np.abs(eigenvalues) - 0.9999999  # Asegurarse de que todos los valores propios son menores que 1 en magnitud

    if np.all(np.abs(eigenvalues) < 0.9999999):
        found_solution = True
        optimal_x = x  # Guardar la solución
        eigss.append(np.abs(eigenvalues))
        raise StopIteration("Found a solution with all eigenvalues having magnitude less than 1.")  # Lanzar una excepción para detener la optimización
    else:
        eigss.append(np.abs(eigenvalues))
        # print(np.abs(eigenvalues))
    return c

def objective_function(x):
    return np.sqrt(np.sum(x**2))

deltat = 2
h = 0.01

# Matrices A y B originales como las definiste antes
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

# Discretización del sistema
sys_continuous = ctrl.StateSpace(A, B, np.eye(6), np.zeros((6, 3)))
sys_discrete = ctrl.c2d(sys_continuous, deltat * h, method='zoh')
A_discrete = sys_discrete.A
B_discrete = sys_discrete.B

x0 = np.array([-241.79, 43.35, -3059, -560, -155, -5161])
constraints = {'type': 'ineq', 'fun': eigenvalue_constraint, 'args': (A_discrete, B_discrete)}

for i in range(1000):
    random_adjustment = np.random.rand(len(x0))*10
    current_x0 = x0 + random_adjustment
    try:
        res = minimize(objective_function, current_x0, method='SLSQP', constraints=[constraints])
    except StopIteration as e:
        print(e)
        break  # Detener la iteración si se encuentra una solución válida

if found_solution:
    print("Optimal solution found with all eigenvalues having magnitude less than 1:", optimal_x)
else:
    print("No solution found with all eigenvalues having magnitude less than 1.")

