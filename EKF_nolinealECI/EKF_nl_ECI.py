# -*- coding: utf-8 -*-
"""
Created on Wed May  8 01:26:50 2024

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
Bx_ECI = array_datos[:, 4]
By_ECI = array_datos[:, 5]
Bz_ECI = array_datos[:, 6]
Sx_ECI = array_datos[:, 10]
Sy_ECI = array_datos[:, 11]
Sz_ECI = array_datos[:, 12]

#%%
# q= np.array([0,0.7071,0,0.7071])
q = np.array([0,0,0,1])
w = np.array([0.0001, 0.0001, 0.0001])
q_est = np.array([0,0,0,1])
# q_est = np.array([0.00985969, 0.70703804, 0.00985969, 0.70703804])
bias = np.array([0.00001,0.00001,0.00001])  # <0.06°/h en radianes por segundo
w_est = w

q0_est = [q_est[0]]
q1_est = [q_est[1]]
q2_est = [q_est[2]]
q3_est = [q_est[3]]
bias0_est = [bias[0]]
bias1_est = [bias[1]]
bias2_est = [bias[2]]
w0_est = [w_est[0]]
w1_est = [w_est[1]]
w2_est = [w_est[2]]
# x_est = np.hstack((q_est[0:3],w))
q_est = np.array([q0_est[-1], q1_est[-1], q2_est[-1], q3_est[-1]])
bias_est = np.array([bias0_est[-1], bias1_est[-1], bias2_est[-1]])
# x_est = np.hstack((q_est[0:3],bias_est)) 

q0_real = [q[0]]
q1_real = [q[1]]
q2_real = [q[2]]
q3_real = [q[3]]
w0_real = [w[0]]
w1_real = [w[1]]
w2_real = [w[2]]
q_real = np.array([q0_real[-1],q1_real[-1],q2_real[-1],q3_real[-1]])
w_body = np.array([w0_real[-1], w1_real[-1], w2_real[-1]])
x_real = np.hstack((np.transpose(q_real[:3]), np.transpose(w_body)))

bi_ECI = [Bx_ECI[0],By_ECI[0],Bz_ECI[0]]
bi_body = functions_nl.rotacion_v(q_real, bi_ECI)

si_ECI = [Sx_ECI[0],Sy_ECI[0],Sz_ECI[0]]
si_body = functions_nl.rotacion_v(q_real, si_ECI)

hh =0.01


diagonal_values = np.array([0.5**2, 0.5**2, 0.5**2, 0.1**2, 0.1**2, 0.1**2])

P_ki = np.diag(diagonal_values)

deltat = 2
limite = 502
t = np.arange(0, limite, deltat)

# 1e-6, 0.04
sigma_S = 1
sigma_B = 1
#%%

for i in range(len(t)-1):
    # print(t[i+1])

    u = np.array([0.15,0.15,0.15])
    
    b_ECI = [Bx_ECI[i],By_ECI[i],Bz_ECI[i]]
    b_body = functions_nl.rotacion_v(q_real, b_ECI)
    b_sensor = functions_nl.simulate_magnetometer_reading(b_body, sigma_B)
    
    s_ECI = [Sx_ECI[i],Sy_ECI[i],Sz_ECI[i]]
    s_body = functions_nl.rotacion_v(q_real, s_ECI)
    s_sensor = functions_nl.simulate_sunsensor_reading(s_body, sigma_S)

    [xx_new_d, qq3_new_d] = functions_nl.mod_nolineal(x_real,u,deltat, b_body,hh)
    
    # x_real = xx_new_d
    q_real = np.array([xx_new_d[0],xx_new_d[1], xx_new_d[2], qq3_new_d])
    w_body = np.array([xx_new_d[3], xx_new_d[4], xx_new_d[5]])
    w_gyros = functions_nl.simulate_gyros_reading(w_body, 0, 0)
    x_real = np.hstack((q_real[0:3],w_gyros))

    q0_real.append(x_real[0])
    q1_real.append(x_real[1])
    q2_real.append(x_real[2])
    q3_real.append(qq3_new_d)
    w0_real.append(x_real[3])
    w1_real.append(x_real[4])
    w2_real.append(x_real[5])

    [q_posteriori, bias_posteriori, P_k_pos,w_plus] = functions_nl.kalman_baroni(q_est, w_body, bias_est, deltat,P_ki, b_sensor, s_sensor, 5e-3, 3e-4, sigma_B,sigma_S)

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
    
# %%
fig0, axes0 = plt.subplots(nrows=1, ncols=2, figsize=(16, 4))

axes0[0].plot(t, q0_real, label='q0 modelo')
axes0[0].plot(t, q1_real, label='q1 modelo')
axes0[0].plot(t, q2_real, label='q2 modelo')
axes0[0].plot(t, q3_real, label='q3 modelo')
axes0[0].set_xlabel('Tiempo [s]')
axes0[0].set_ylabel('cuaternion [-]')
axes0[0].legend()
axes0[0].set_title('cuaterniones obtenidos por el modelo de control no lineal ECI')
axes0[0].grid()
# axes0[0].set_ylim(-1, 1)  # Ajusta los límites en el eje Y

axes0[1].plot(t, w0_real, label='w0 modelo')
axes0[1].plot(t, w1_real, label='w1 modelo')
axes0[1].plot(t, w2_real, label='w2 modelo')
axes0[1].set_xlabel('Tiempo [s]')
axes0[1].set_ylabel('velocidad angular [rad/s]')
axes0[1].legend()
axes0[1].set_title('velocidad angular obtenidos por el modelo de control no lineal ECI')
axes0[1].grid()
plt.tight_layout()
plt.show()

fig0, axes0 = plt.subplots(nrows=1, ncols=2, figsize=(16, 4))

axes0[0].plot(t, q0_est, label='q0 kalman')
axes0[0].plot(t, q1_est, label='q1 kalman')
axes0[0].plot(t, q2_est, label='q2 kalman')
axes0[0].plot(t, q3_est, label='q3 kalman')
axes0[0].set_xlabel('Tiempo [s]')
axes0[0].set_ylabel('cuaternion [-]')
axes0[0].legend()
axes0[0].set_title('cuaterniones obtenidos por EKF')
axes0[0].grid()
# axes0[0].set_ylim(-1, 1)  # Ajusta los límites en el eje Y

axes0[1].plot(t, w0_est, label='w0 kalman')
axes0[1].plot(t, w1_est, label='w1 kalman')
axes0[1].plot(t, w2_est, label='w2 kalman')
axes0[1].set_xlabel('Tiempo [s]')
axes0[1].set_ylabel('velocidad angular [rad/s]')
axes0[1].legend()
axes0[1].set_title('velocidad angular obtenidos por EKF')
axes0[1].grid()
plt.tight_layout()
plt.show()




# #%%

# for i in range(len(t)-1):
#     q_est = np.array([q0_est[-1], q1_est[-1], q2_est[-1], q3_est[-1]])
#     bias_est = np.array([bias0_est[-1], bias1_est[-1], bias2_est[-1]])
#     u = np.array([15,15,15])
    
#     # x_est = np.hstack((q_est[0:3],bias_est))  

#     b_ECI = [Bx_ECI[i+1],By_ECI[i+1],Bz_ECI[i+1]]
#     b_body = functions_nl.rotacion_v(q_est, b_ECI)
#     b_sensor = functions_nl.simulate_magnetometer_reading(b_body, sigma_B)
    
#     s_ECI = [Sx_ECI[i+1],Sy_ECI[i+1],Sz_ECI[i+1]]
#     s_body = functions_nl.rotacion_v(q_est, s_ECI)
#     s_sensor = functions_nl.simulate_sunsensor_reading(s_body, sigma_S)
#     # 1e-6, 0.04
#     [q_posteriori, bias_posteriori, P_k_pos,w_plus] = functions_nl.kalman_baroni(q_est, w_body, bias_est, deltat,P_ki, b_sensor,b_ECI, s_sensor,s_ECI, 5e-3, 3e-4, sigma_B,sigma_S)

#     q0_est.append(q_posteriori[0])
#     q1_est.append(q_posteriori[1])
#     q2_est.append(q_posteriori[2])
#     q3_est.append(q_posteriori[3])
#     bias0_est.append(bias_posteriori[0])
#     bias1_est.append(bias_posteriori[1])
#     bias2_est.append(bias_posteriori[2])

#     w0_est.append(w_plus[0])
#     w1_est.append(w_plus[1])
#     w2_est.append(w_plus[2])
    
#     P_ki = P_k_pos
    
#     [xx_new_d, qq3_new_d] = functions_nl.mod_nolineal(x_real,u,deltat, b_body,hh)
    
#     x_real = xx_new_d

#     q0_real.append(xx_new_d[0])
#     q1_real.append(xx_new_d[1])
#     q2_real.append(xx_new_d[2])
#     q3_real.append(qq3_new_d)
#     w0_real.append(xx_new_d[3])
#     w1_real.append(xx_new_d[4])
#     w2_real.append(xx_new_d[5])

#     q_real = np.array([q0_real[-1], q1_real[-1], q2_real[-1], q3_real[-1]])
#     w_body = np.array([w0_real[-1], w1_real[-1], w2_real[-1]])