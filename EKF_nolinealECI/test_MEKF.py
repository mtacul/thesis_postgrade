# -*- coding: utf-8 -*-
"""
Created on Tue May 14 18:07:10 2024

@author: nachi
"""
import pandas as pd
import numpy as np
import functions_nl
import matplotlib.pyplot as plt
# %%
archivo_csv = "Vectores_orbit_ECI.csv"

# Leer el archivo CSV en un DataFrame de pandas
df = pd.read_csv(archivo_csv)

# Convertir el DataFrame a un array de NumPy
array_datos = df.to_numpy()

t_aux = array_datos[:, 0]
Bx_orbit = array_datos[:, 1]
By_orbit = array_datos[:, 2]
Bz_orbit = array_datos[:, 3]
Sx_orbit = array_datos[:, 7]
Sy_orbit = array_datos[:, 8]
Sz_orbit = array_datos[:, 9]

#%%
q = np.array([0,0,0,1])
bias = np.array([0,0,0])  # <0.06°/h en radianes por segundo
wi = np.array([0.00001,0.00001,0.00001])
deltat = 2
limite = 102
t = np.arange(0, limite, deltat)
w0_est = [wi[0]]
w1_est = [wi[1]]
w2_est = [wi[2]]

bias0_est = [bias[0]]
bias1_est = [bias[1]]
bias2_est = [bias[2]]
bias_est = np.array([bias0_est[-1], bias1_est[-1], bias2_est[-1]])

q0_est = [q[0]]
q1_est = [q[1]]
q2_est = [q[2]]
q3_est = [q[3]]
q_est = np.array([q0_est[-1],q1_est[-1],q2_est[-1],q3_est[-1]])

w0_minus = [wi[0]]
w1_minus = [wi[1]]
w2_minus = [wi[2]]

# Generar y guardar 1000 instancias de w
for i in range(len(t)-1):
    w_s = np.random.rand(3)*0.01
    w = functions_nl.simulate_gyros_reading(w_s,0,2e-4)
    w0_minus.append(w[0])
    w1_minus.append(w[1])
    w2_minus.append(w[2])

w_minus = np.vstack((w0_minus,w1_minus,w2_minus))
w_time = w_minus[:,0]

hh =0.01

diagonal_values = np.array([0.5**2, 0.5**2, 0.5**2, 0.1**2, 0.1**2, 0.1**2])

P_ki = np.diag(diagonal_values)

# 1e-6, 0.04
sigma_S = 1e-6
sigma_B = 0.036

#%%
for i in range(len(t)-1):
    print(t[i])

    u = np.array([15,15,15])
    
    b_orbit = [Bx_orbit[i],By_orbit[i],Bz_orbit[i]]
    b_body = functions_nl.rotacion_v(q_est, b_orbit)
    b_sensor = functions_nl.simulate_magnetometer_reading(b_body, sigma_B)
    
    s_orbit = [Sx_orbit[i],Sy_orbit[i],Sz_orbit[i]]
    s_body = functions_nl.rotacion_v(q_est, s_orbit)
    s_sensor = functions_nl.simulate_sunsensor_reading(s_body, sigma_S)
    [q_posteriori, bias_posteriori, P_k_pos,w_plus] = functions_nl.kalman_baroni(q_est, w_time, bias_est, deltat,P_ki, b_sensor, s_sensor, 5e-3, 3e-4, sigma_B,sigma_S)

    q0_est.append(q_posteriori[0])
    q1_est.append(q_posteriori[1])
    q2_est.append(q_posteriori[2])
    q3_est.append(q_posteriori[3])
    bias0_est.append(bias_posteriori[0])
    bias1_est.append(bias_posteriori[1])
    bias2_est.append(bias_posteriori[2])
    
    w0_est.append(w_plus[0])
    w1_est.append(w_plus[1])
    w2_est.append(w_plus[2])
    
    q_est = np.array([q0_est[-1], q1_est[-1], q2_est[-1], q3_est[-1]])
    bias_est = np.array([bias0_est[-1], bias1_est[-1], bias2_est[-1]])
    P_ki = P_k_pos
    w_time = w_minus[:,i+1]

# %%
fig0, axes0 = plt.subplots(nrows=1, ncols=3, figsize=(16, 4))

axes0[0].plot(t, q0_est, label='q0 modelo')
axes0[0].plot(t, q1_est, label='q1 modelo')
axes0[0].plot(t, q2_est, label='q2 modelo')
axes0[0].plot(t, q3_est, label='q3 modelo')
axes0[0].set_xlabel('Tiempo [s]')
axes0[0].set_ylabel('cuaternion [-]')
axes0[0].legend()
axes0[0].set_title('cuaterniones obtenidos por el modelo de control no lineal ECI')
axes0[0].grid()
# axes0[0].set_ylim(-1, 1)  # Ajusta los límites en el eje Y

axes0[1].plot(t, w0_est, label='w0 modelo')
axes0[1].plot(t, w1_est, label='w1 modelo')
axes0[1].plot(t, w2_est, label='w2 modelo')
axes0[1].set_xlabel('Tiempo [s]')
axes0[1].set_ylabel('velocidad angular [rad/s]')
axes0[1].legend()
axes0[1].set_title('velocidad angular obtenidos por el modelo de control no lineal ECI')
axes0[1].grid()

axes0[2].plot(t, bias0_est, label='bias0 modelo')
axes0[2].plot(t, bias1_est, label='bias1 modelo')
axes0[2].plot(t, bias2_est, label='bias2 modelo')
axes0[2].set_xlabel('Tiempo [s]')
axes0[2].set_ylabel('velocidad angular [rad/s]')
axes0[2].legend()
axes0[2].set_title('velocidad angular obtenidos por el modelo de control no lineal ECI')
axes0[2].grid()
plt.tight_layout()
plt.show()

fig0, axes0 = plt.subplots(nrows=1, ncols=2, figsize=(16, 4))

axes0[0].plot(t, w0_est, label='w0 plus')
axes0[0].plot(t, w1_est, label='w1 plus')
axes0[0].plot(t, w2_est, label='w2 plus')
axes0[0].set_xlabel('Tiempo [s]')
axes0[0].set_ylabel('velocidad angular [rad/s]')
axes0[0].legend()
axes0[0].set_title('velocidad angular obtenidos por el modelo de control no lineal ECI')
axes0[0].grid()
# axes0[0].set_ylim(-1, 1)  # Ajusta los límites en el eje Y

axes0[1].plot(t, w0_minus, label='w0 modelo')
axes0[1].plot(t, w1_minus, label='w1 modelo')
axes0[1].plot(t, w2_minus, label='w2 modelo')
axes0[1].set_xlabel('Tiempo [s]')
axes0[1].set_ylabel('velocidad angular [rad/s]')
axes0[1].legend()
axes0[1].set_title('velocidad angular obtenidos por el modelo de control no lineal ECI')
axes0[1].grid()

plt.tight_layout()
plt.show()