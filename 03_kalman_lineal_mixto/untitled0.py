# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 14:47:19 2024

@author: nachi
"""

import numpy as np
import matplotlib.pyplot as plt

# Parámetros del satélite
I_body = np.diag([0.02, 0.02, 0.01])  # Tensor de inercia del satélite (kg·m²)

# Parámetros de las ruedas de reacción
I_wheel = np.array([0.001, 0.001, 0.001])  # Momentos de inercia de las ruedas (kg·m²)
wheel_speeds = np.array([0.0, 0.0, 0.0])   # Velocidades angulares iniciales de las ruedas (rad/s)

# Velocidad angular inicial del satélite
omega_body = np.array([0.0, 0.0, 0.0])  # rad/s

# Tiempo de simulación
dt = 0.01  # Paso de tiempo (s)
t_total = 10  # Tiempo total de simulación (s)
steps = int(t_total / dt)

# Momentos aplicados a las ruedas (L^w) en Nm
L_wheel = np.array([0.001, 0.0, 0.0])  # Momento aplicado solo en el eje x

# Almacenar datos para graficar
omega_body_history = []
wheel_speeds_history = []

# Simulación de la dinámica
for _ in range(steps):
    # Actualizar el momentum angular de las ruedas: dH^w = L^w * dt
    dH_wheel = L_wheel * dt
    wheel_speeds += dH_wheel / I_wheel

    # Momentum angular de las ruedas H^w
    H_wheel = I_wheel * wheel_speeds

    # Momentum angular total: H = I_body * omega_body + H_wheel
    H_total = I_body @ omega_body + H_wheel

    # Calcular la nueva velocidad angular del cuerpo del satélite: omega_body = I_body^-1 * (H_total - H_wheel)
    omega_body = np.linalg.inv(I_body) @ (H_total - H_wheel)

    # Guardar datos para graficar
    omega_body_history.append(omega_body.copy())
    wheel_speeds_history.append(wheel_speeds.copy())

# Convertir listas a arrays
omega_body_history = np.array(omega_body_history)
wheel_speeds_history = np.array(wheel_speeds_history)

# Graficar resultados
time = np.linspace(0, t_total, steps)

plt.figure(figsize=(12, 6))

# Velocidades angulares del satélite
plt.subplot(2, 1, 1)
plt.plot(time, omega_body_history[:, 0], label='ω_x (rad/s)')
plt.plot(time, omega_body_history[:, 1], label='ω_y (rad/s)')
plt.plot(time, omega_body_history[:, 2], label='ω_z (rad/s)')
plt.title('Velocidades Angulares del Satélite')
plt.xlabel('Tiempo (s)')
plt.ylabel('Velocidad Angular (rad/s)')
plt.legend()
plt.grid()

# Velocidades angulares de las ruedas de reacción
plt.subplot(2, 1, 2)
plt.plot(time, wheel_speeds_history[:, 0], label='ω_wheel_x (rad/s)')
plt.plot(time, wheel_speeds_history[:, 1], label='ω_wheel_y (rad/s)')
plt.plot(time, wheel_speeds_history[:, 2], label='ω_wheel_z (rad/s)')
plt.title('Velocidades Angulares de las Ruedas de Reacción')
plt.xlabel('Tiempo (s)')
plt.ylabel('Velocidad Angular de Ruedas (rad/s)')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
