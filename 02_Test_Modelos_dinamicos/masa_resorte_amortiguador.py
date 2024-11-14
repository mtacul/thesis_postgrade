# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 09:54:35 2024

@author: nachi
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Parámetros del sistema
m = 1.0  # masa
c = 0.5  # coeficiente de amortiguamiento
k = 2.0  # constante del resorte

# Matriz A del sistema de ecuaciones de primer orden
A = np.array([
    [0, 1],
    [-k/m, -c/m]
])

# Definimos la función para el sistema de ecuaciones diferenciales
def mass_spring_damper(t, x):
    return A @ x

# Condiciones iniciales: posición inicial x(0) = 1, velocidad inicial x'(0) = 0
x0 = [1.0, 0.0]

# Tiempo de simulación
t_span = (0, 20)
t_eval = np.linspace(0, 20, 1000)

# Resolver usando solve_ivp
solution = solve_ivp(mass_spring_damper, t_span, x0, t_eval=t_eval, method='RK45')

# Graficar resultados
plt.plot(solution.t, solution.y[0], label='Posición (x)')
plt.plot(solution.t, solution.y[1], label='Velocidad (x\')')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.title('Sistema masa-resorte-amortiguador usando solve_ivp')
plt.legend()
plt.grid()
plt.show()


#%%

# Definimos la función de dinámica para el método RK4 manual
def dynamics_manual(A, x):
    return A @ x

# Función de un paso de integración RK4
def rk4_step(dynamics, x, A, h):
    k1 = h * dynamics(A, x)
    k2 = h * dynamics(A, x + 0.5 * k1)
    k3 = h * dynamics(A, x + 0.5 * k2)
    k4 = h * dynamics(A, x + k3)
    return x + (k1 + 2*k2 + 2*k3 + k4) / 6

# Configuración de la simulación
h = 0.01  # Paso de tiempo
t_manual = np.arange(0, 20, h)
x_manual = np.zeros((len(t_manual), 2))
x_manual[0] = x0  # Condiciones iniciales

# Integración numérica usando RK4
for i in range(1, len(t_manual)):
    x_manual[i] = rk4_step(dynamics_manual, x_manual[i-1], A, h)

# Graficar resultados
plt.plot(t_manual, x_manual[:, 0], label='Posición (x) - RK4 manual')
plt.plot(t_manual, x_manual[:, 1], label='Velocidad (x\') - RK4 manual')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.title('Sistema masa-resorte-amortiguador usando RK4 manual')
plt.legend()
plt.grid()
plt.show()
