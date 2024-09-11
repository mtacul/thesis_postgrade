# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 15:13:44 2024

@author: nachi
"""
import functions_tesis
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
limite = 102
t = np.arange(0, limite, deltat)

I_x = 0.037
I_y = 0.036
I_z = 0.006

w0 = 0
w1 = 0
w2 = 0


#%%
q= np.array([0,0.7071,0,0.7071])
w = np.array([0.0001, 0.0001, 0.0001])
q_est = np.array([0.00985969, 0.70703804, 0.00985969, 0.70703804])
bias = np.array([0,0,0])  # <0.06°/h en radianes por segundo
w_est = w-bias

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

q0_real = [q[0]]
q1_real = [q[1]]
q2_real = [q[2]]
q3_real = [q[3]]
w0_real = [w[0]]
w1_real = [w[1]]
w2_real = [w[2]]
q_real = np.array([q0_real[-1],q1_real[-1],q2_real[-1],q3_real[-1]])
w_body = np.array([w0_real[-1], w1_real[-1], w2_real[-1]])
w_gyros = functions_tesis.simulate_gyros_reading(w_body, 0,0)
x_real = np.hstack((np.transpose(q_real[:3]), np.transpose(w_gyros)))

bi_orbit = [Bx_orbit[0],By_orbit[0],Bz_orbit[0]]
bi_body = functions_tesis.rotacion_v(q_real, bi_orbit, 0)

si_orbit = [vx_sun_orbit[0],vy_sun_orbit[0],vz_sun_orbit[0]]
si_body = functions_tesis.rotacion_v(q_real, si_orbit, 0)
# B_body_ctrl = np.array([1617.3,4488.9,46595.8])*1e-9
# v_body_ctrl = np.array([0.09473943,-0.684664,-0.503806])
hh =0.01

# [A,B,C,A_discrete,B_discrete,C_discrete] = functions_tesis.A_B(I_x, I_y, I_z, w0_O, w0, w1, w2, deltat, hh, bi_orbit,bi_orbit)
[Ab,Bb,Cb,Ab_discrete,Bb_discrete,Cb_discrete] = functions_tesis.A_B_bar(I_x, I_y, I_z, w0_O, w0, w1, w2, deltat, hh, bi_body,si_body[0],si_body[1],si_body[2])


P_ki = np.eye(6)

#%%



for i in range(len(t)-1):
    print(t[i+1])
    q_est = np.array([q0_est[-1], q1_est[-1], q2_est[-1], q3_est[-1]])
    bias_est = np.array([bias0_est[-1], bias1_est[-1], bias2_est[-1]])
    u_est = np.array([15,15,15])
    
    x_est = np.hstack((q_est[0:3],bias_est))  

    b_orbit = [Bx_orbit[i+1],By_orbit[i+1],Bz_orbit[i+1]]
    b_body = functions_tesis.rotacion_v(q_real, b_orbit, 0)

    vsun_orbit = [vx_sun_orbit[i+1],vy_sun_orbit[i+1],vz_sun_orbit[i+1]]
    s_body = functions_tesis.rotacion_v(q_real, vsun_orbit, 0)

    # print(q_est)
    # print(q_real)
    # print(w_gyros)
    # print(w_est)
    print(b_body,s_body)
    # print(x_est)
    
    [q_posteriori, bias_posteriori, P_k_pos] = functions_tesis.kalman_baroni(q_est, w_gyros, bias_est, deltat,P_ki, b_body, s_body, 5e-3, 3e-4, 1e-6, 0.04)
    # asasd

    q0_est.append(q_posteriori[0])
    q1_est.append(q_posteriori[1])
    q2_est.append(q_posteriori[2])
    q3_est.append(q_posteriori[3])
    bias0_est.append(bias_posteriori[0])
    bias1_est.append(bias_posteriori[1])
    bias2_est.append(bias_posteriori[2])
    
    bias_est = np.array([bias0_est[-1],bias1_est[-1],bias2_est[-2]])
    w_est = w_gyros-bias_est
    w0_est.append(w_est[0])
    w1_est.append(w_est[1])
    w2_est.append(w_est[2])
    
    P_ki = P_k_pos
    
    [xx_new_d, qq3_new_d] = functions_tesis.mod_lineal_disc(
        x_real, u_est, deltat, hh, Ab_discrete,Bb_discrete)
    
    x_real = xx_new_d

    q0_real.append(xx_new_d[0])
    q1_real.append(xx_new_d[1])
    q2_real.append(xx_new_d[2])
    q3_real.append(qq3_new_d)
    w0_real.append(xx_new_d[3])
    w1_real.append(xx_new_d[4])
    w2_real.append(xx_new_d[5])

    # q_real = np.array([q0_real[-1], q1_real[-1], q2_real[-1], q3_real[-1]])
    w_body = np.array([w0_real[-1], w1_real[-1], w2_real[-1]])
    w_gyros = functions_tesis.simulate_gyros_reading(w_body, 0,0)

    

    
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
axes0[0].set_title('velocidad angular obtenidos por el modelo de control lineal discreto')
axes0[0].grid()
# axes0[0].set_ylim(-1, 1)  # Ajusta los límites en el eje Y

axes0[1].plot(t, w0_est, label='w0 kalman')
axes0[1].plot(t, w1_est, label='w1 kalman')
axes0[1].plot(t, w2_est, label='w2 kalman')
axes0[1].set_xlabel('Tiempo [s]')
axes0[1].set_ylabel('velocidad angular [rad/s]')
axes0[1].legend()
axes0[1].set_title('velocidad angular estimados por el filtro de kalman lineal discreto')
axes0[1].grid()
plt.tight_layout()
plt.show()