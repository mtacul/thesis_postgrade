# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 15:13:44 2024

@author: nachi
"""
import functions_03_rw
import numpy as np
import control as ctrl
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

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
Bx_IGRF = array_datos[:, 4]
By_IGRF = array_datos[:, 5]
Bz_IGRF = array_datos[:, 6]
vx_sun_orbit = array_datos[:, 7]
vy_sun_orbit = array_datos[:, 8]
vz_sun_orbit = array_datos[:, 9]
vsun_x = array_datos[:, 10]
vsun_y = array_datos[:, 11]
vsun_z = array_datos[:, 12]
#%%
w0_O = 0.00163

deltat = 2
limite = 1002
t = np.arange(0, limite, deltat)

w0_eq = 0
w1_eq = 0
w2_eq = 0

sigma_ss = 0.036
sigma_b = 1e-6

#%% rueda de reaccion

I_x = 0.037
I_y = 0.036
I_z = 0.006

m_s0 = 0.06 #kg
m_s1 = m_s0
m_s2 = m_s0

b_0 = 0.05
b_1 = 0.05
b_2 = 0.15

I_s0_x = 0.005
I_s1_x = I_s0_x
I_s2_x = I_s0_x

I_s0_y = 0.005
I_s1_y = I_s0_y
I_s2_y = I_s0_y

I_s0_z = 0.004
I_s1_z = I_s0_z
I_s2_z = I_s0_z	

J_x = I_x + I_s0_x + I_s1_x + I_s2_x + m_s1*b_1**2 + m_s2*b_2**2
J_y = I_y + I_s0_y + I_s1_y + I_s2_y + m_s0*b_0**2 + m_s2*b_2**2
J_z = I_z + I_s0_z + I_s1_z + I_s2_z + m_s0*b_0**2 + m_s1*b_1**2

#%%
# q= np.array([0,0.7071,0,0.7071])
q= np.array([0,0,0,1])
w = np.array([0.0001, 0.0001, 0.0001])
ws = np.array([0.00001, 0.00001, 0.00001])

# q_est = np.array([0.00985969, 0.70703804, 0.00985969, 0.70703804])
q_est= np.array([0.0120039,0.0116517,0.0160542,0.999731])

q0_est = [q_est[0]]
q1_est = [q_est[1]]
q2_est = [q_est[2]]
q3_est = [q_est[3]]
w0_est = [w[0]]
w1_est = [w[1]]
w2_est = [w[2]]
w0s_est = [ws[0]]
w1s_est = [ws[1]]
w2s_est = [ws[2]]
# x_est = np.hstack((q_est[0:3],w))

q0_real = [q[0]]
q1_real = [q[1]]
q2_real = [q[2]]
q3_real = [q[3]]
w0_real = [w[0]]
w1_real = [w[1]]
w2_real = [w[2]]
w0s_real = [ws[0]]
w1s_real = [ws[1]]
w2s_real = [ws[2]]
q_real = np.array([q0_real[-1],q1_real[-1],q2_real[-1],q3_real[-1]])
w_body = np.array([w0_real[-1], w1_real[-1], w2_real[-1]])
w_gyros = functions_03_rw.simulate_gyros_reading(w_body,0,0)
ws_real = np.array([w0s_real[-1], w1s_real[-1], w2s_real[-1]])
x_real = np.hstack((np.transpose(q_real[:3]), np.transpose(w_gyros), np.transpose(ws_real)))
h_real = np.array([x_real[6]/I_s0_x-x_real[3], x_real[7]/I_s1_y-x_real[4], x_real[8]/I_s2_z-x_real[5]])

diagonal_values = np.array([0.5**2, 0.5**2, 0.5**2, 0.1**2, 0.1**2, 0.1**2,0.01**2,0.01**2,0.01**2])
hh =0.01
P_ki = np.diag(diagonal_values)
A_dis = []


#%%
np.random.seed(42)
for i in range(len(t)-1):
    print(t[i+1])
    q_est = np.array([q0_est[-1], q1_est[-1], q2_est[-1], q3_est[-1]])
    w_est = np.array([w0_est[-1], w1_est[-1], w2_est[-1]])
    ws_est = np.array([w0s_est[-1], w1s_est[-1], w2s_est[-1]])

    x_est = np.hstack((np.transpose(q_est[:3]), np.transpose(w_est), np.transpose(ws_est)))
    u_est = np.array([0.015,0.015,0.015])
    h_est = np.array([x_est[6]/I_s0_x-x_est[3], x_est[7]/I_s1_y-x_est[4], x_est[8]/I_s2_z-x_est[5]])

    b_orbit = [Bx_orbit[i],By_orbit[i],Bz_orbit[i]]
    b_body_med = functions_03_rw.rotacion_v(q_real, b_orbit,sigma_b)
    b_body_est = functions_03_rw.rotacion_v(q_est, b_orbit,sigma_b)
    
    s_orbit = [vx_sun_orbit[i],vy_sun_orbit[i],vz_sun_orbit[i]]
    s_body_med = functions_03_rw.rotacion_v(q_real, s_orbit,sigma_ss)
    s_body_est = functions_03_rw.rotacion_v(q_est, s_orbit,sigma_ss)

    # print(x_est)
    # print(q_real)
    # print(w_gyros)
    
    # print(b_body,w_gyros)
    # print(x_est)
    [A,B,C,A_discrete,B_discrete,C_discrete] = functions_03_rw.A_B(I_x,I_y,I_z,w0_O, 0,0,0, I_s0_x, I_s1_y, I_s2_z,0,0,0, J_x, J_y, J_z, deltat, hh,b_orbit, b_body_med, s_body_med)
    [q_posteriori, w_posteriori, P_k_pos,K_k, ws_posteriori] = functions_03_rw.kalman_lineal(A_discrete, B_discrete,C_discrete, x_est, u_est, b_body_med, b_body_est, s_body_med, s_body_est, P_ki, sigma_b, sigma_ss, deltat,hh,h_real,h_est)
    A_dis.append(A_discrete)
    q0_est.append(q_posteriori[0])
    q1_est.append(q_posteriori[1])
    q2_est.append(q_posteriori[2])
    q3_est.append(q_posteriori[3])
    w0_est.append(w_posteriori[0])
    w1_est.append(w_posteriori[1])
    w2_est.append(w_posteriori[2])
    w0s_est.append(ws_posteriori[0])
    w1s_est.append(ws_posteriori[1])
    w2s_est.append(ws_posteriori[2])
    P_ki = P_k_pos
    
    [xx_new_d, qq3_new_d] = functions_03_rw.mod_lineal_disc(
        x_real, u_est, deltat, hh, A_discrete,B_discrete)
    
    x_real = xx_new_d
    w_gyros = functions_03_rw.simulate_gyros_reading(x_real[3:6],0.00057595865,0.0008726646)
    ws_real = x_real[6:9]
    q0_real.append(xx_new_d[0])
    q1_real.append(xx_new_d[1])
    q2_real.append(xx_new_d[2])
    q3_real.append(qq3_new_d)
    w0_real.append(w_gyros[0])
    w1_real.append(w_gyros[1])
    w2_real.append(w_gyros[2])
    w0s_real.append(ws_real[0])
    w1s_real.append(ws_real[1])
    w2s_real.append(ws_real[2])
    q_real = np.array([q0_real[-1],q1_real[-1],q2_real[-1],q3_real[-1]])
    h_real = np.array([x_real[6]/I_s0_x-x_real[3], x_real[7]/I_s1_y-x_real[4], x_real[8]/I_s2_z-x_real[5]])

    # q_real = np.array([q0_real[-1], q1_real[-1], q2_real[-1], q3_real[-1]])
    # www_real = np.array([w0_real[-1], w1_real[-1], w2_real[-1]])
    # ww_real = functions_03.simulate_gyros_reading(www_real, 0,0)

[MSE_cuat, MSE_omega]  = functions_03_rw.cuat_MSE_NL(q0_real, q1_real, q2_real, q3_real, w0_real, w1_real, w2_real, q0_est, q1_est, q2_est, q3_est, w0_est, w1_est, w2_est)   

    
# %%
fig0, axes0 = plt.subplots(nrows=1, ncols=2, figsize=(16, 4))

axes0[0].plot(t, q0_real, label='q0 modelo')
axes0[0].plot(t, q1_real, label='q1 modelo')
axes0[0].plot(t, q2_real, label='q2 modelo')
axes0[0].plot(t, q3_real, label='q3 modelo')
axes0[0].set_xlabel('Tiempo [s]')
axes0[0].set_ylabel('cuaternion [-]')
axes0[0].legend()
axes0[0].set_title('cuaterniones obtenidos por el modelo de control lineal discreto')
axes0[0].grid()
# axes0[0].set_ylim(-1, 1)  # Ajusta los límites en el eje Y

axes0[1].plot(t, q0_est, label='q0 kalman')
axes0[1].plot(t, q1_est, label='q1 kalman')
axes0[1].plot(t, q2_est, label='q2 kalman')
axes0[1].plot(t, q3_est, label='q3 kalman')
axes0[1].set_xlabel('Tiempo [s]')
axes0[1].set_ylabel('cuaternion [-]')
axes0[1].legend()
axes0[1].set_title('cuaterniones estimados por el filtro de kalman lineal discreto')
axes0[1].grid()
plt.tight_layout()
plt.show()

fig0, axes0 = plt.subplots(nrows=1, ncols=2, figsize=(16, 4))

axes0[0].plot(t, w0_real, label='w0 modelo')
axes0[0].plot(t, w1_real, label='w1 modelo')
axes0[0].plot(t, w2_real, label='w2 modelo')
axes0[0].set_xlabel('Tiempo [s]')
axes0[0].set_ylabel('velocidad angular [rad/s]')
axes0[0].legend()
axes0[0].set_title('velocidades angulares obtenidos por el modelo de control lineal discreto')
axes0[0].grid()
# axes0[0].set_ylim(-1, 1)  # Ajusta los límites en el eje Y

axes0[1].plot(t, w0_est, label='w0 kalman')
axes0[1].plot(t, w1_est, label='w1 kalman')
axes0[1].plot(t, w2_est, label='w2 kalman')
axes0[1].set_xlabel('Tiempo [s]')
axes0[1].set_ylabel('velocidad angular [rad/s]')
axes0[1].legend()
axes0[1].set_title('velocidades angulares estimados por el filtro de kalman lineal discreto')
axes0[1].grid()
plt.tight_layout()
plt.show()

quats = np.array([0,1,2,3])
plt.figure(figsize=(12, 6))
plt.scatter(quats[0], MSE_cuat[0], label='mse q0', color='r',marker='*')
plt.scatter(quats[1], MSE_cuat[1], label='mse q1', color='b',marker='*')
plt.scatter(quats[2], MSE_cuat[2], label='mse q2', color='k',marker='*')
plt.scatter(quats[3], MSE_cuat[3], label='mse q3', color='g',marker='*')
plt.xlabel('Cuaterniones')
plt.ylabel('Mean Square Error [-]')
plt.legend()
plt.title('MSE de cada cuaternion entre lineal discreto y kalman lineal discreto')
# plt.xlim(20000,100000)
# plt.ylim(-0.005,0.005)
plt.grid()
plt.show()

vels = np.array([0,1,2])
plt.figure(figsize=(12, 6))
plt.scatter(vels[0], MSE_omega[0], label='mse w0', color='r',marker='*')
plt.scatter(vels[1], MSE_omega[1], label='mse w1', color='b',marker='*')
plt.scatter(vels[2], MSE_omega[2], label='mse w2', color='k',marker='*')
plt.xlabel('Velocidades angulares')
plt.ylabel('Mean Square Error [rad/s]')
plt.legend()
plt.title('MSE de cada velocidad angular entre lineal discreto y kalman lineal discreto')
# plt.xlim(20000,100000)
# plt.ylim(-0.005,0.005)
plt.grid()
plt.show()