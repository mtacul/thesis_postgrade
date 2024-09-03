# -*- coding: utf-8 -*-
"""
Created on Wed May 29 15:45:54 2024

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
# q = np.array([0,0,0,1])
q = np.array([0.7077,0,0,0.7077])
bias = np.array([0,0,0])  # <0.06°/h en radianes por segundo
wi = np.array([0.00001,0.00001,0.00001])

w0_O = 0.00163
I_x = 0.037
I_y = 0.036
I_z = 0.006

deltat = 2
limite = 22
t = np.arange(0, limite, deltat)

bias0_est = [bias[0]]
bias1_est = [bias[1]]
bias2_est = [bias[2]]
bias_est = np.array([bias0_est[-1], bias1_est[-1], bias2_est[-1]])

q0_est = [q[0]]
q1_est = [q[1]]
q2_est = [q[2]]
q3_est = [q[3]]
q_est = np.array([q0_est[-1],q1_est[-1],q2_est[-1],q3_est[-1]])

q0_real = [q[0]]
q1_real = [q[1]]
q2_real = [q[2]]
q3_real = [q[3]]
w0_real = [wi[0]]
w1_real = [wi[1]]
w2_real = [wi[2]]
bias0_real = [bias[0]]
bias1_real = [bias[1]]
bias2_real = [bias[2]]
bias_real = np.array([bias0_est[-1], bias1_est[-1], bias2_est[-1]])

q_real = np.array([q0_real[-1],q1_real[-1],q2_real[-1],q3_real[-1]])
w_body = np.array([w0_real[-1], w1_real[-1], w2_real[-1]])
w_gyros_real = functions_nl.simulate_gyros_reading(w_body, 0, bias_real)
w_gyros_real = w_gyros_real[:,0]
w_gyros_est = functions_nl.simulate_gyros_reading(w_body, 0, bias_est)
w_gyros_est = w_gyros_est[:,0]
x_real = np.hstack((np.transpose(q_real[:3]), np.transpose(w_gyros_real)))

hh =0.01

diagonal_values = np.array([0.5**2, 0.5**2, 0.5**2, 0.1**2, 0.1**2, 0.1**2])

P_ki = np.diag(diagonal_values)

# 1e-6, 0.04
sigma_S = 0.036
sigma_B = 1e-6

#%%
np.random.seed(42)
for i in range(len(t)-1):
    print(t[i])
    print(q_est)
    print(q_real)
    u = np.array([0.15,0.15,0.15])
    
    b_orbit = [Bx_orbit[i],By_orbit[i],Bz_orbit[i]]
    b_body_med = functions_nl.rotacion_v(q_real, b_orbit)
    b_sensor_med = functions_nl.simulate_magnetometer_reading(b_body_med, 0)
    b_body_est = functions_nl.rotacion_v(q_est, b_orbit)
    b_sensor_est = functions_nl.simulate_magnetometer_reading(b_body_est, 0)
    
    s_orbit = [Sx_orbit[i],Sy_orbit[i],Sz_orbit[i]]
    s_body_med = functions_nl.rotacion_v(q_real, s_orbit)
    s_sensor_med = functions_nl.simulate_sunsensor_reading(s_body_med, 0)
    s_body_est = functions_nl.rotacion_v(q_est, s_orbit)
    s_sensor_est = functions_nl.simulate_sunsensor_reading(s_body_est, 0)
    
    [q_posteriori, bias_posteriori, P_k_pos,w_plus] = functions_nl.kalman_baroni(q_est, w_body, bias_est, deltat,P_ki, b_sensor_med, s_sensor_med, b_sensor_est, s_sensor_est, 5e-3, 3e-4, sigma_B,sigma_S)

    q0_est.append(q_posteriori[0])
    q1_est.append(q_posteriori[1])
    q2_est.append(q_posteriori[2])
    q3_est.append(q_posteriori[3])
    
    bias0_est.append(bias_posteriori[0])
    bias1_est.append(bias_posteriori[1])
    bias2_est.append(bias_posteriori[2])
    
    q_est = np.array([q0_est[-1], q1_est[-1], q2_est[-1], q3_est[-1]])
    bias_est = np.array([bias0_est[-1], bias1_est[-1], bias2_est[-1]])
    w_gyros_est = functions_nl.simulate_gyros_reading(w_body, 0, bias_est)
    w_gyros_est = w_gyros_est[:,0]
    
    
    [xx_new_d, qq3_new_d] = functions_nl.mod_nolineal(
        x_real, u, deltat, b_body_med,hh,deltat,I_x,I_y,I_z,w0_O)
    
    q_real = np.array([xx_new_d[0],xx_new_d[1],xx_new_d[2],qq3_new_d])
    x_real = xx_new_d
    bias_real = np.random.normal(0,1e-4,3)

    q0_real.append(xx_new_d[0])
    q1_real.append(xx_new_d[1])
    q2_real.append(xx_new_d[2])
    q3_real.append(qq3_new_d)
    w0_real.append(xx_new_d[3])
    w1_real.append(xx_new_d[4])
    w2_real.append(xx_new_d[5])
    bias0_real.append(bias_real[0])
    bias1_real.append(bias_real[1])
    bias2_real.append(bias_real[2])
    
    P_ki = P_k_pos
    w_body =  np.array([w0_real[-1], w1_real[-1], w2_real[-1]])
    w_gyros_real = functions_nl.simulate_gyros_reading(w_body, 0, bias_real)
    w_gyros_real = w_gyros_real[:,0]


# %%
fig0, axes0 = plt.subplots(nrows=1, ncols=2, figsize=(16, 4))

axes0[0].plot(t, q0_est, label='q0 MEKF')
axes0[0].plot(t, q1_est, label='q1 MEKF')
axes0[0].plot(t, q2_est, label='q2 MEKF')
axes0[0].plot(t, q3_est, label='q3 MEKF')
axes0[0].set_xlabel('Tiempo [s]')
axes0[0].set_ylabel('cuaternion [-]')
axes0[0].legend()
axes0[0].set_title('cuaterniones obtenidos por MEKF')
axes0[0].grid()
# axes0[0].set_ylim(-1, 1)  # Ajusta los límites en el eje Y

axes0[1].plot(t, q0_real, label='q0 modelo')
axes0[1].plot(t, q1_real, label='q1 modelo')
axes0[1].plot(t, q2_real, label='q2 modelo')
axes0[1].plot(t, q3_real, label='q3 modelo')
axes0[1].set_xlabel('Tiempo [s]')
axes0[1].set_ylabel('cuaternion [-]')
axes0[1].legend()
axes0[1].set_title('cuaterniones obtenidos por el modelo de control no lineal orbit')
axes0[1].grid()


plt.tight_layout()
plt.show()

fig0, axes0 = plt.subplots(nrows=1, ncols=2, figsize=(16, 4))

axes0[0].plot(t, bias0_est, label='bias0 est')
axes0[0].plot(t, bias1_est, label='bias1 est')
axes0[0].plot(t, bias2_est, label='bias2 set')
axes0[0].set_xlabel('Tiempo [s]')
axes0[0].set_ylabel('bias[rad/s2]')
axes0[0].legend()
axes0[0].set_title('bias obtenidos por MEKF')
axes0[0].grid()
# axes0[0].set_ylim(-1, 1)  # Ajusta los límites en el eje Y

axes0[1].plot(t, bias0_real, label='bias0 modelo')
axes0[1].plot(t, bias1_real, label='bias1 modelo')
axes0[1].plot(t, bias2_real, label='bias2 modelo')
axes0[1].set_xlabel('Tiempo [s]')
axes0[1].set_ylabel('bias [rad/s2]')
axes0[1].legend()
axes0[1].set_title('bias simulado al azar')
axes0[1].grid()

plt.tight_layout()
plt.show()