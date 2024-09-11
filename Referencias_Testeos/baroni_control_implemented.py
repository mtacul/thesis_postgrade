# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 23:46:38 2024

@author: nachi
"""

import functions
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

# q = np.array([-0.7071/np.sqrt(3), 0.7071 /
#              np.sqrt(3), -0.7071/np.sqrt(3), 0.7071])

q = np.array([0,0,0,1])
w = np.array([0.0001, 0.0001, 0.0001])
deltat = 2
limite = 1002

# bias = np.array([0.03 / 3600 * np.pi/180,
#                  0.03 / 3600 * np.pi/180,
#                  0.03 / 3600 * np.pi/180])  # <0.06°/h en radianes por segundo
bias = np.array([0,0,0])  # <0.06°/h en radianes por segundo
t = np.arange(0, limite, deltat)

# # Crear un array con los valores deseados para la diagonal
# valores_diagonales = [0.5**2, 0.5**2, 0.5**2, 0.1**2, 0.1**2, 0.1**2]

# P_ki = np.diag(valores_diagonales)
P_ki = np.eye(6)

# q0_disc = [-0.332543]
# q1_disc = [-0.662521]
# q2_disc = [-0.149582]
# q3_disc = [0.654298]
q0_disc = [4.08176e-07]
q1_disc = [-5.25635e-07]
q2_disc = [0.00157263]
q3_disc = [0.999999]
bias0_disc = [bias[0]]
bias1_disc = [bias[1]]
bias2_disc = [bias[2]]


q0_control = [q[0]]
q1_control = [q[1]]
q2_control = [q[2]]
q3_control = [q[3]]
w0_control = [w[0]]
w1_control = [w[1]]
w2_control = [w[2]]

# control continuo memoria (mala constante pero estabiliza)
# Kp_x_disc =  -11.7126815151457
# Kp_y_disc = 0.0215395552140476
# Kp_z_disc = -1.94092971659118
# Kd_x_disc = 2.23100994690840
# Kd_y_disc = -0.0591645752827404
# Kd_z_disc =  -473.824574185555

# control discreto
Kp_x_disc = -259.013200255933
Kp_y_disc = 7.39137008894686
Kp_z_disc = -294.830151281917
Kd_x_disc = -696.450801220821
Kd_y_disc = 138.374206336783
Kd_z_disc = -4509.59878465641

b_orbit = [Bx_orbit[0],By_orbit[0],Bz_orbit[0]]
vsun_orbit = [vx_sun_orbit[0],vy_sun_orbit[0],vz_sun_orbit[0]]

bb = functions.rotacion_v(np.array([q0_disc[-1], q1_disc[-1], q2_disc[-1], q3_disc[-1]]), b_orbit, 1e-6)
ss = functions.rotacion_v(np.array([q0_disc[-1], q1_disc[-1], q2_disc[-1], q3_disc[-1]]), vsun_orbit, 0.04)

K_Ex_app_disc = functions.K(
    Kp_x_disc, Kp_y_disc, Kp_z_disc, Kd_x_disc, Kd_y_disc, Kd_z_disc)

acc_control = [np.dot(K_Ex_app_disc,np.hstack((np.transpose(np.array([q0_disc[-1], q1_disc[-1], q2_disc[-1]])), np.transpose(np.array([w0_control[-1], w1_control[-1], w2_control[-1]])))))]
acc_control_real = [np.dot(K_Ex_app_disc,np.hstack((np.transpose(np.array([q0_control[-1], q1_control[-1], q2_control[-1]])), np.transpose(np.array([w0_control[-1], w1_control[-1], w2_control[-1]])))))]

Bx_bodys = [bb[0]]
By_bodys = [bb[1]]
Bz_bodys = [bb[2]]

vsun_bodys_x = [ss[0]]
vsun_bodys_y = [ss[1]]
vsun_bodys_z = [ss[2]]

#%%
w0_O = 0.00163

I_x = 0.037
I_y = 0.036
I_z = 0.006

w0 = 0
w1 = 0
w2 = 0

# B_body_ctrl = np.array([1617.3,4488.9,46595.8])*1e-9
# v_body_ctrl = np.array([0.09473943,-0.684664,-0.503806])
hh =0.01

# [A,B,C,A_discrete,B_discrete,C_discrete] = functions.A_B(I_x, I_y, I_z, w0_O, w0, w1, w2, deltat, hh, B_body_ctrl)
[Ab,Bb,Cb,Ab_discrete,Bb_discrete,Cb_discrete] = functions.A_B_bar(I_x, I_y, I_z, w0_O, w0, w1, w2, deltat, hh, bb,ss[0],ss[1],ss[2])

#%%

for i in range(len(t)-1):
    # print(t[i])

    qq_disc = np.array([q0_disc[-1], q1_disc[-1], q2_disc[-1], q3_disc[-1]])
    qq_control = np.array([q0_control[-1], q1_control[-1], q2_control[-1], q3_control[-1]])
    
    www_control = np.array([w0_control[-1], w1_control[-1], w2_control[-1]])
    ww_control = functions.simulate_gyros_reading(www_control, 0,0)
    
    bias_disc = np.array([bias0_disc[-1],bias1_disc[-1], bias2_disc[-1]])
    ww_disc = functions.simulate_gyros_reading(www_control, 0,0)

    xx_disc = np.hstack((np.transpose(qq_disc[:3]), np.transpose(ww_disc)))
    uu_disc = np.dot(K_Ex_app_disc, xx_disc)
    
    xx_real = np.hstack((np.transpose(qq_control[:3]), np.transpose(ww_control)))
    uu_real = np.dot(K_Ex_app_disc, xx_real)
    
    uu_disc = functions.torquer(uu_disc,1e9)
    uu_real = functions.torquer(uu_real,1e9)

    acc_control.append(uu_disc)
    acc_control_real.append(uu_real)
    

    # accion de control u = K*x_estimado y x usado en A es el mismo
    [xx_new_d, qq3_new_d] = functions.mod_lineal_disc_bar(
        xx_real, uu_disc, deltat,hh,Ab_discrete,Bb_discrete)
    
    # accion de control u= K*x_estimado y x usado en A es el real del modelo de control
    # [xx_new_d, qq3_new_d] = functions.mod_lineal_cont(
    #     xx_real, uu_disc, deltat,hh,A,B)
    
    # [xx_new_d, qq3_new_d] = functions.mod_nolineal(xx_real,uu_disc,deltat,bb)

    
    q0_control.append(xx_new_d[0])
    q1_control.append(xx_new_d[1])
    q2_control.append(xx_new_d[2])
    q3_control.append(qq3_new_d)
    w0_control.append(xx_new_d[3])
    w1_control.append(xx_new_d[4])
    w2_control.append(xx_new_d[5])

    qq_control = np.array([q0_control[-1], q1_control[-1], q2_control[-1], q3_control[-1]])
    bias_priori = np.array([bias0_disc[-1], bias1_disc[-1], bias2_disc[-1]])

    # bb = [Bx_IGRF[i], By_IGRF[i], Bz_IGRF[i]]
    # ss = [vsun_x[i], vsun_y[i], vsun_z[i]]
    
    b_orbit = [Bx_orbit[i+1],By_orbit[i+1],Bz_orbit[i+1]]
    #error magnetometro de 1e-6
    bb = functions.rotacion_v(qq_control, b_orbit, 0)
    Bx_bodys.append(bb[0])
    By_bodys.append(bb[1])
    Bz_bodys.append(bb[2])
    
    vsun_orbit = [vx_sun_orbit[i+1],vy_sun_orbit[i+1],vz_sun_orbit[i+1]]
    # error sun sensor 0.04
    ss = functions.rotacion_v(qq_control, vsun_orbit, 0)
    vsun_bodys_x.append(ss[0])
    vsun_bodys_y.append(ss[1])
    vsun_bodys_z.append(ss[2])
    
    [q_posteriori, bias_posteriori, P_k_pos] = functions.kalman_baroni(qq_disc, ww_control,
                                                                       bias_priori, deltat,
                                                                       P_ki, bb, ss,
                                                                       5e-3, 3e-4,
                                                                       1e-6, 0.04)
    q0_disc.append(q_posteriori[0])
    q1_disc.append(q_posteriori[1])
    q2_disc.append(q_posteriori[2])
    q3_disc.append(q_posteriori[3])
    bias0_disc.append(bias_posteriori[0])
    bias1_disc.append(bias_posteriori[1])
    bias2_disc.append(bias_posteriori[2])

    P_ki = P_k_pos


q0_disc = np.array(q0_disc)
q1_disc = np.array(q1_disc)
q2_disc = np.array(q2_disc)
q3_disc = np.array(q3_disc)
bias0_disc = np.array(bias0_disc)
bias1_disc = np.array(bias1_disc)
bias2_disc = np.array(bias2_disc)

q0_control = np.array(q0_control)
q1_control = np.array(q1_control)
q2_control = np.array(q2_control)
q3_control = np.array(q3_control)
w0_control = np.array(w0_control)
w1_control = np.array(w1_control)
w2_control = np.array(w2_control)

acc_control = np.array(acc_control)
acc_control_real = np.array(acc_control_real)

# #%% Obtencion del MSE

# [RPY_kalman, RPY_control, mse_roll, mse_pitch, mse_yaw] = functions.RPY_MSE(t, q0_disc, q1_disc, q2_disc, q3_disc,q0_control, q1_control, q2_control, q3_control)

# # Nombre del archivo
# archivo_c = "ss2.csv"

# # Abrir el archivo en modo escritura
# with open(archivo_c, 'w') as f:
#     # Escribir los encabezados
#     f.write("t, Roll_kalman, Pitch_kalman, Yaw_kalman, bias0_kalman, bias1_kalman, bias2_kalman, Roll_control,Pitch_control,Yaw_control,  w0_control, w1_control, w2_control, mse_Roll, mse_Pitch, mse_Yaw, ux, uy, uz \n")

#     # Escribir los datos en filas
#     for i in range(len(t)):
#         f.write("{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n".format( 
#             t[i], RPY_kalman[i,0], RPY_kalman[i,1], RPY_kalman[i,2], bias0_disc[i], bias1_disc[i], 
#             bias2_disc[i], RPY_control[i,0], RPY_control[i,1], RPY_control[i,2], w0_control[i],
#             w1_control[i], w2_control[i], mse_roll, mse_pitch, mse_yaw, acc_control[i,0],
#             acc_control[i,1], acc_control[i,2]
#         ))

# print("Vectores guardados en el archivo:", archivo_c)
# %%
fig0, axes0 = plt.subplots(nrows=1, ncols=2, figsize=(16, 4))

axes0[0].plot(t, q0_control, label='q0 modelo')
axes0[0].plot(t, q1_control, label='q1 modelo')
axes0[0].plot(t, q2_control, label='q2 modelo')
axes0[0].plot(t, q3_control, label='q3 modelo')
axes0[0].set_xlabel('Tiempo [s]')
axes0[0].set_ylabel('cuaternion [-]')
axes0[0].legend()
axes0[0].set_title('cuaterniones obtenidos por el modelo de control lineal discreto')
axes0[0].grid()
# axes0[0].set_ylim(-1, 1)  # Ajusta los límites en el eje Y

axes0[1].plot(t, q0_disc, label='q0 kalman')
axes0[1].plot(t, q1_disc, label='q1 kalman')
axes0[1].plot(t, q2_disc, label='q2 kalman')
axes0[1].plot(t, q3_disc, label='q3 kalman')
axes0[1].set_xlabel('Tiempo [s]')
axes0[1].set_ylabel('cuaternion [-]')
axes0[1].legend()
axes0[1].set_title('cuaterniones estimados por el filtro de kalman lineal discreto')
axes0[1].grid()
plt.tight_layout()
plt.show()

fig0, axes0 = plt.subplots(nrows=1, ncols=2, figsize=(16, 4))

axes0[0].plot(t, w0_control, label='w0 modelo')
axes0[0].plot(t, w1_control, label='w1 modelo')
axes0[0].plot(t, w2_control, label='w2 modelo')
axes0[0].set_xlabel('Tiempo [s]')
axes0[0].set_ylabel('velocidad angular [rad/s]')
axes0[0].legend()
axes0[0].set_title('velocidades angulares obtenidos por el modelo de control lineal discreto')
axes0[0].grid()
# axes0[0].set_ylim(-1, 1)  # Ajusta los límites en el eje Y

axes0[1].plot(t, bias0_disc, label='w0 kalman')
axes0[1].plot(t, bias1_disc, label='w1 kalman')
axes0[1].plot(t, bias2_disc, label='w2 kalman')
axes0[1].set_xlabel('Tiempo [s]')
axes0[1].set_ylabel('velocidad angular [rad/s]')
axes0[1].legend()
axes0[1].set_title('velocidades angulares estimados por el filtro de kalman lineal discreto')
axes0[1].grid()

plt.tight_layout()
plt.show()


# plt.figure(figsize=(12, 6))
# plt.plot(t, RPY_all_id[:, 0], label='Roll')
# plt.plot(t, RPY_all_id[:, 1], label='Pitch')
# plt.plot(t, RPY_all_id[:, 2], label='Yaw')
# plt.xlabel('Tiempo [s]')
# plt.ylabel('Ángulos de Euler [°]')
# plt.legend()
# plt.title('Obtencion de los ángulos de Euler')
# # plt.xlim(0.8e7,1.7e7)
# # plt.ylim(-15,2)
# plt.grid()
# plt.show()
# # plt.set_yli1m(-10, 10)  # Ajusta los límites en el eje Y

fig0, axes0 = plt.subplots(nrows=1, ncols=2, figsize=(16, 4))

axes0[0].plot(t, q0_control, label='q0 modelo')
axes0[0].plot(t, q0_disc, label='q0 kalman')

# axes0[0].plot(t[0:len(t)-1], q1_control, label='q1 modelo')
# axes0[0].plot(t[0:len(t)-1], q2_control, label='q2 modelo')
# axes0[0].plot(t[0:len(t)-1], q3_control, label='q3 modelo')
axes0[0].set_xlabel('Tiempo [s]')
axes0[0].set_ylabel('cuaternion [-]')
axes0[0].legend()
axes0[0].set_title('cuaterniones controlados al equilibrio por modelo y kalman')
axes0[0].grid()
axes0[0].set_ylim(-1, 1)  # Ajusta los límites en el eje Y

# axes0[1].plot(t[0:len(t)-1], q0_disc[0:len(t)-1], label='q0 discreto')
axes0[1].plot(t, q1_control, label='q1 modelo')
axes0[1].plot(t, q1_disc, label='q1 kalman')
# axes0[1].plot(t[0:len(t)-1], q2_disc[0:len(t)-1], label='q2 discreto')
# axes0[1].plot(t[0:len(t)-1], q3_disc[0:len(t)-1], label='q3 discreto')
axes0[1].set_xlabel('Tiempo [s]')
axes0[1].set_ylabel('cuaternion [-]')
axes0[1].legend()
axes0[1].set_title('cuaterniones controlados al equilibrio por modelo y kalman')
axes0[1].grid()
axes0[1].set_ylim(-1, 1)  # Ajusta los límites en el eje Y
plt.tight_layout()
plt.show()


fig0, axes0 = plt.subplots(nrows=1, ncols=2, figsize=(16, 4))

axes0[0].plot(t, q2_control, label='q2 modelo')
axes0[0].plot(t, q2_disc, label='q2 kalman')


axes0[0].set_xlabel('Tiempo [s]')
axes0[0].set_ylabel('cuaternion [-]')
axes0[0].legend()
axes0[0].set_title('cuaterniones controlados al equilibrio por modelo y kalman')
axes0[0].grid()
axes0[0].set_ylim(-1, 1)  # Ajusta los límites en el eje Y

# axes0[1].plot(t[0:len(t)-1], q0_disc[0:len(t)-1], label='q0 discreto')
axes0[1].plot(t, q3_control, label='q3 modelo')
axes0[1].plot(t, q3_disc, label='q3 kalman')
# axes0[1].plot(t[0:len(t)-1], q2_disc[0:len(t)-1], label='q2 discreto')
# axes0[1].plot(t[0:len(t)-1], q3_disc[0:len(t)-1], label='q3 discreto')
axes0[1].set_xlabel('Tiempo [s]')
axes0[1].set_ylabel('cuaternion [-]')
axes0[1].legend()
axes0[1].set_title('cuaterniones controlados al equilibrio por modelo y kalman')
axes0[1].grid()
# axes0[1].set_ylim(-1, 1)  # Ajusta los límites en el eje Y
plt.tight_layout()
plt.show()


fig0, axes0 = plt.subplots(nrows=1, ncols=2, figsize=(16, 4))

axes0[0].plot(t[range(0, len(t))], acc_control[:,0], label='x_est')
axes0[0].plot(t[range(0, len(t))], acc_control[:,1], label='y_est')
axes0[0].plot(t[range(0, len(t))], acc_control[:,2], label='z_est')
axes0[0].set_xlabel('Tiempo [s]')
axes0[0].set_ylabel('accion de control []')
axes0[0].legend()
axes0[0].set_title('accion de control con x estimado')
axes0[0].grid()

axes0[1].plot(t[range(0, len(t))], acc_control_real[:,0], label='x_real')
axes0[1].plot(t[range(0, len(t))], acc_control_real[:,1], label='y_real')
axes0[1].plot(t[range(0, len(t))], acc_control_real[:,2], label='z_real')
axes0[1].set_xlabel('Tiempo [s]')
axes0[1].set_ylabel('accion de control []')
axes0[1].legend()
axes0[1].set_title('accion de control con x real')
axes0[1].grid()
plt.tight_layout()
plt.show()


# quats = np.array([0,1,2,3])
# plt.figure(figsize=(12, 6))
# plt.scatter(quats[0], MSE_cuat[0], label='mse q0', color='r',marker='*')
# plt.scatter(quats[1], MSE_cuat[1], label='mse q1', color='b',marker='*')
# plt.scatter(quats[2], MSE_cuat[2], label='mse q2', color='k',marker='*')
# plt.scatter(quats[3], MSE_cuat[3], label='mse q3', color='g',marker='*')
# plt.xlabel('Cuaterniones')
# plt.ylabel('Mean Square Error [-]')
# plt.legend()
# plt.title('MSE de cada cuaternion entre lineal discreto y kalman lineal discreto')
# # plt.xlim(20000,100000)
# # plt.ylim(-0.005,0.005)
# plt.grid()
# plt.show()

# vels = np.array([0,1,2])
# plt.figure(figsize=(12, 6))
# plt.scatter(vels[0], MSE_omega[0], label='mse w0', color='r',marker='*')
# plt.scatter(vels[1], MSE_omega[1], label='mse w1', color='b',marker='*')
# plt.scatter(vels[2], MSE_omega[2], label='mse w2', color='k',marker='*')
# plt.xlabel('Velocidades angulares')
# plt.ylabel('Mean Square Error [rad/s]')
# plt.legend()
# plt.title('MSE de cada velocidad angular entre lineal discreto y kalman lineal discreto')
# # plt.xlim(20000,100000)
# # plt.ylim(-0.005,0.005)
# plt.grid()
# plt.show()