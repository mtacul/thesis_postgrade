# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 20:31:27 2024

@author: nachi
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

w0 = 7.292115e-5 #rad/s
Ix = 1030.17 #kgm^2
Iy = 3015.65 #kgm^2
Iz = 3030.43 #kgm^2

Ixy = Ix - Iy
Ixyz = Ix - Iy -Iz
Iyz = Iy - Iz
Ixz = Ix - Iz
Iyx= Iy - Ix
Iyxz = Iy -Ix - Iz
Iyzx = Iy - Iz- Ix

A = np.array([
    [0, 0,0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1],
    [(-4*Iyz*w0**2)/ Ix, 0, 0, 0, 0, (-Iyzx*w0)/Ix],
    [0, (-3*Ixz*w0**2)/Iy, 0, 0, 0, 0],
    [0, 0, (-Iyx*w0**2)/Iz, (Iyxz*w0**2)/Iz, 0, 0]
])

def system(t, x):
    return A @ x

x0 = np.array([0, 0, 0, 0.001, 0, 0])  

t_span = (0, 100000)
solution = solve_ivp(system, t_span, x0, method='RK45', t_eval=np.linspace(0, 100000, 1000000))

plt.figure(figsize=(10, 6))
for i in range(6):
    plt.plot(solution.t, solution.y[i], label=f'x{i+1}')
plt.xlabel('Tiempo')
plt.ylabel('Valores de x')
plt.title('Solución de $\dot{x} = A \cdot x$')
plt.legend()
plt.grid()
plt.show()