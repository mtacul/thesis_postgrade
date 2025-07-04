# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 15:13:44 2024

@author: nachi
"""
import functions_03_rw_1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_are
import control as ctrl

# %% Cargar datos del .csv obtenido

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

deltat = 2
# limite =  5762*69
limite =  5762*1
t = np.arange(0, limite, deltat)

#%% Parámetros geométricos y orbitales dados

w0_O = 0.00163

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

#%% seleccion de nivel de sensor

# Definir los valores
sigma_ss_values = {
    1: np.sin(0.833*np.pi/180),
    2: np.sin(0.167*np.pi/180),
    3: np.sin(0.05*np.pi/180),
    4: 0
}

sigma_b_values = {
    1: 1.18e-9,
    2: 0.1e-9,
    3: 0.012e-9,
    4: 0
}

ruido_w_values = {
    1: 0.12 * np.pi / 180,
    2: 0.050 * np.pi / 180,
    3: 0.033 * np.pi / 180,
    4: 0
}

bias_w_values = {
    1: (0.05 * np.pi / 180) / 3600,
    2: (0.03 * np.pi / 180) / 3600,
    3: (0.02 * np.pi / 180) / 3600,
    4: 0
}

# Solicitar al usuario que seleccione una opción
opcion = int(input("Seleccione un nivel de sensor (1: bad, 2: med, 3: good, 4: sin ruido): "))

# Asignar los valores seleccionados
sigma_ss = sigma_ss_values[opcion]
sigma_b = sigma_b_values[opcion]
ruido_w = ruido_w_values[opcion]
bias_w = bias_w_values[opcion]

#%% Seleccion de nivel de actuador

# Definir los valores
lim_tau_values = {
    1: 1e-3,   
    2: 0.025,
    3: 0.250
}

# Solicitar al usuario que seleccione una opción
opcion_tau = int(input("Seleccione un nivel de actuador (1: bad, 2: med, 3: good): "))

# Asignar los valores seleccionados
lim = lim_tau_values[opcion_tau]

#%% Condiciones iniciales reales y estimadas

q = np.array([0.1771,0.3063,0.1771,0.9202])
# q= np.array([0.7071,0,0,0.7071])
# q= np.array([0,0,0,1])
# q = np.array([0.0789,0.0941,0.0789,0.9893])
w = np.array([0.0001, 0.0001, 0.0001])
# ws = np.array([0.00001, 0.00001, 0.00001])
# q_est = np.array([0.70703804,0.00985969, 0.00985969, 0.70703804])
# q_est= np.array([0.0120039,0.0116517,0.0160542,0.999731])
# q_est = np.array([0.0789,0.0941,0.0789,0.9893])
q_est = np.array([0.1771,0.3063,0.1771,0.9202])

q0_est = [q_est[0]]  
q1_est = [q_est[1]]
q2_est = [q_est[2]]
q3_est = [q_est[3]]
w0_est = [w[0]]
w1_est = [w[1]]
w2_est = [w[2]]
# w0s_est = [ws[0]]
# w1s_est = [ws[1]]
# w2s_est = [ws[2]]

q0_real = [q[0]]
q1_real = [q[1]]
q2_real = [q[2]]
q3_real = [q[3]]
w0_real = [w[0]]
w1_real = [w[1]]
w2_real = [w[2]]
# w0s_real = [ws[0]]
# w1s_real = [ws[1]]
# w2s_real = [ws[2]]
q_real = np.array([q0_real[-1],q1_real[-1],q2_real[-1],q3_real[-1]])
w_body = np.array([w0_real[-1], w1_real[-1], w2_real[-1]])
w_gyros = functions_03_rw_1.simulate_gyros_reading(w_body, 0,0)
# ws_real = np.array([w0s_real[-1], w1s_real[-1], w2s_real[-1]])
# x_real = np.hstack((np.transpose(q_real[:3]), np.transpose(w_gyros), np.transpose(ws_real)))
x_real = np.hstack((np.transpose(q_real[:3]), np.transpose(w_gyros)))
# h_real = np.array([x_real[6]/I_s0_x-x_real[3], x_real[7]/I_s1_y-x_real[4], x_real[8]/I_s2_z-x_real[5]])

bi_orbit = [Bx_orbit[0],By_orbit[0],Bz_orbit[0]]
b_body_i = functions_03_rw_1.rotacion_v(q_real, bi_orbit, sigma_b)

si_orbit = [vx_sun_orbit[0],vy_sun_orbit[0],vz_sun_orbit[0]]
s_body_i = functions_03_rw_1.rotacion_v(q_real, si_orbit, sigma_ss)

# _mod de modelo
hh = deltat
hh_mod = 0.2

#%%
[A,B,C,A_discrete,B_discrete,C_discrete] = functions_03_rw_1.A_B(w0_O,0,0,0,J_x, J_y, J_z, deltat, hh, bi_orbit,b_body_i, s_body_i)
[A_mod,B_mod,C_mod,A_discrete_mod,B_discrete_mod,C_discrete_mod] = functions_03_rw_1.A_B(w0_O,0,0,0, J_x, J_y, J_z, deltat, hh_mod, bi_orbit,b_body_i, s_body_i)


# # Índices correspondientes a las filas y columnas
# indices_0 = [0, 3, 6]
# indices_1 = [1, 4, 7]
# indices_2 = [2, 5, 8]

# # Seleccionar la primera columna y las filas deseadas
# B_discrete_3x1_0 = B_discrete[indices_0, 0]
# B_discrete_3x1_1 = B_discrete[indices_1, 1]
# B_discrete_3x1_2 = B_discrete[indices_2, 2]

# # Convertir a una matriz columna (3x1)
# B_discrete_0 = B_discrete_3x1_0.reshape(-1, 1)# Extraer submatriz de 3x3
# B_discrete_1 = B_discrete_3x1_1.reshape(-1, 1)# Extraer submatriz de 3x3
# B_discrete_2 = B_discrete_3x1_2.reshape(-1, 1)# Extraer submatriz de 3x3

# A_discrete_0 = A_discrete[np.ix_(indices_0, indices_0)]
# A_discrete_1 = A_discrete[np.ix_(indices_1, indices_1)]
# A_discrete_2 = A_discrete[np.ix_(indices_2, indices_2)]

# C = np.eye(3)
# D = np.zeros((3, 1))  # Assuming D has the same number of rows as A and the same number of columns as B

#%% Control LQR en cada matriz

# # Definir las matrices Q y R del coste del LQR
# diag_Q1 = np.array([1, 1, 1])*100000
# diag_R1 = np.array([1])
# Q1 = np.diag(diag_Q1)
# R1 = np.diag(diag_R1)

# diag_Q2 = np.array([1, 1, 1])*100000
# diag_R2 = np.array([1])
# Q2 = np.diag(diag_Q2)
# R2 = np.diag(diag_R2)

# diag_Q3 = np.array([1, 1, 1])*100000
# diag_R3 = np.array([10])
# Q3 = np.diag(diag_Q3)
# R3 = np.diag(diag_R3)

# K1, P1, eigenvalues1 = ctrl.dlqr(A_discrete_0, B_discrete_0, Q1, R1)
# K2, P2, eigenvalues2 = ctrl.dlqr(A_discrete_1, B_discrete_1, Q2, R2)
# K3, P3, eigenvalues3 = ctrl.dlqr(A_discrete_2, B_discrete_2, Q3, R3)

# K1_1 = np.array([K1[0][0],K1[0][1],K1[0][2]])
# K2_1 = np.array([K2[0][0],K2[0][1],K2[0][2]])
# K3_1 = np.array([K3[0][0],K3[0][1],K3[0][2]])

# Kk1 = np.array([K1_1[0],0,0,K1_1[1],0,0,K1_1[2],0,0])
# Kk2 = np.array([0,K2_1[0],0,0,K2_1[1],0,0,K2_1[2],0])
# Kk3 = np.array([0,0,K3_1[0],0,0,K3_1[1],0,0,K3_1[2]])
# K = np.vstack((Kk1,Kk2,Kk3))

#%% Control LQR entero

# Definir las matrices Q y R del coste del LQR
diag_Q = np.array([10, 10, 10, 10, 10, 10])*10000
diag_R = np.array([0.1,0.1,0.1])*10

Q = np.diag(diag_Q)
R = np.diag(diag_R)


# Calcular la matriz de retroalimentación K
K, P, asad = ctrl.dlqr(A_discrete, B_discrete, Q, R)
k_place = K

# asad[0] = 0.99999
# asad[1] = 0.8
# asad[3] = 0.15
# asad[4] = 0.85
# asad[5] = 0.995 + 2.625e-5j
# asad[6] = 0.995 - 2.625e-5j

# asad[0] = 0.9999
# asad[7] = 0.99991
# asad[8] = 0.992

# k_place = ctrl.place(A_discrete,B_discrete,asad)
# asad_place,vect_place = np.linalg.eig(A_discrete-B_discrete@k_place)

#%%
# asad,vect = np.linalg.eig(A_discrete-B_discrete@K)
# print(asad)

# asad1,vect1 = np.linalg.eig(A_discrete_0-B_discrete_0@K1)
# asad2,vect2 = np.linalg.eig(A_discrete_1-B_discrete_1@K2)
# asad3,vect3 = np.linalg.eig(A_discrete_2-B_discrete_2@K3)


# ### valores propios entero
# asad[0] = 0.37
# asad[1] = 0.1
# asad[2] = 0.888

# # ### valores propios conjugado
# # asad[1] = 0.08
# # asad[4] = 0.31
# # asad[5] = 0.3296
# # asad[5] = 0.80 #ultimo
# # asad[6] = 0.9990 #ultimo
# # asad[7] = 0.9960 #ultimo
# asad[8] = 0.999999 # este es

# k_place = ctrl.place(A_discrete,B_discrete,asad)

# asad_place,vect_place = np.linalg.eig(A_discrete-B_discrete@k_place)

#%% Simulacion dinamica de actitud
aus = []
# diagonal_values = np.array([0.5**2, 0.5**2, 0.5**2, 0.1**2, 0.1**2, 0.1**2,0.01**2,0.01**2,0.01**2])
diagonal_values = np.array([0.5**2, 0.5**2, 0.5**2, 0.1**2, 0.1**2, 0.1**2])
P_ki = np.diag(diagonal_values)
np.random.seed(42)

for i in range(len(t)-1):
    # print(t[i+1])
    # print(x_real)
    # print(np.dot(-K*10,x_real))
    
    q_est = np.array([q0_est[-1], q1_est[-1], q2_est[-1], q3_est[-1]])
    w_est = np.array([w0_est[-1], w1_est[-1], w2_est[-1]])
    # ws_est = np.array([w0s_est[-1], w1s_est[-1], w2s_est[-1]])

    # x_est = np.hstack((np.transpose(q_est[:3]), np.transpose(w_est), np.transpose(ws_est)))
    x_est = np.hstack((np.transpose(q_est[:3]), np.transpose(w_est)))
    u_est = np.dot(-k_place,x_est)
    # print(u_est)
    u_est = functions_03_rw_1.torquer(u_est,lim)
    aus.append(u_est)

    [xx_new_d, qq3_new_d] = functions_03_rw_1.mod_lineal_disc(
        x_real, u_est, deltat, hh_mod, A_discrete_mod,B_discrete_mod)
    
    # print(u_est)
    # print(xx_new_d)
    # input()
    x_real = xx_new_d
    w_gyros = functions_03_rw_1.simulate_gyros_reading(x_real[3:6],ruido_w,bias_w)
    ws_real = x_real[6:9]
    q0_real.append(xx_new_d[0])
    q1_real.append(xx_new_d[1])
    q2_real.append(xx_new_d[2])
    q3_real.append(qq3_new_d)
    w0_real.append(w_gyros[0])
    w1_real.append(w_gyros[1])
    w2_real.append(w_gyros[2])
    # w0s_real.append(ws_real[0])
    # w1s_real.append(ws_real[1])
    # w2s_real.append(ws_real[2])
    q_real = np.array([q0_real[-1],q1_real[-1],q2_real[-1],q3_real[-1]])
    # h_real = np.array([x_real[6]/I_s0_x-x_real[3], x_real[7]/I_s1_y-x_real[4], x_real[8]/I_s2_z-x_real[5]])

    b_orbit = [Bx_orbit[i+1],By_orbit[i+1],Bz_orbit[i+1]]
    b_body_med = functions_03_rw_1.rotacion_v(q_real, b_orbit,sigma_b)
    
    s_orbit = [vx_sun_orbit[i],vy_sun_orbit[i],vz_sun_orbit[i]]
    s_body_med = functions_03_rw_1.rotacion_v(q_real, s_orbit,sigma_ss)

    # [A,B,C,A_discrete,B_discrete,C_discrete] = functions_03_rw_1.A_B_kalman(I_x,I_y,I_z,w0_O, 0,0,0, I_s0_x, I_s1_y, I_s2_z,0,0,0, J_x, J_y, J_z,  deltat, hh,b_orbit, b_body_med, s_body_med)
    [A,B,C,A_discrete,B_discrete,C_discrete] = functions_03_rw_1.A_B(w0_O,0,0,0,J_x, J_y, J_z, deltat, hh,b_orbit, b_body_med, s_body_med)

    if opcion == 4:
        [q_posteriori, w_posteriori, P_k_pos,K_k] = functions_03_rw_1.kalman_lineal(A_discrete, B_discrete,C_discrete, x_est, u_est, b_orbit,b_body_med, s_orbit, s_body_med, P_ki, sigma_b,sigma_ss, deltat,hh,1,1)
    elif opcion == 1 or opcion == 2 or opcion == 3:
        [q_posteriori, w_posteriori, P_k_pos,K_k] = functions_03_rw_1.kalman_lineal(A_discrete, B_discrete,C_discrete, x_est, u_est, b_orbit,b_body_med, s_orbit, s_body_med, P_ki, sigma_b, sigma_ss, deltat,hh, sigma_b, sigma_ss)
    
    q0_est.append(q_posteriori[0])
    q1_est.append(q_posteriori[1])
    q2_est.append(q_posteriori[2])
    q3_est.append(q_posteriori[3])
    w0_est.append(w_posteriori[0])
    w1_est.append(w_posteriori[1])
    w2_est.append(w_posteriori[2])
    # w0s_est.append(ws_posteriori[0])
    # w1s_est.append(ws_posteriori[1])
    # w2s_est.append(ws_posteriori[2])
    
    # q0_est.append(q0_real[-1])
    # q1_est.append(q1_real[-1])
    # q2_est.append(q2_real[-1])
    # q3_est.append(q3_real[-1])
    # w0_est.append(w0_real[-1])
    # w1_est.append(w1_real[-1])
    # w2_est.append(w2_real[-1])
    # w0s_est.append(w0s_real[-1])
    # w1s_est.append(w1s_real[-1])
    # w2s_est.append(w2s_real[-1])
    
    P_ki = P_k_pos
    
[MSE_cuat, MSE_omega]  = functions_03_rw_1.cuat_MSE_NL(q0_real, q1_real, q2_real, q3_real, w0_real, w1_real, w2_real, q0_est, q1_est, q2_est, q3_est, w0_est, w1_est, w2_est)   
[RPY_all_est,RPY_all_id,mse_roll,mse_pitch,mse_yaw] = functions_03_rw_1.RPY_MSE(t, q0_est, q1_est, q2_est, q3_est, q0_real, q1_real, q2_real, q3_real)   
print(aus[0])
# %% Gráficas de los resultados obtenidos en la suite de simulacion

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

# fig0, axes0 = plt.subplots(nrows=1, ncols=2, figsize=(16, 4))

# axes0[0].plot(t, w0s_real, label='w0 RW')
# axes0[0].plot(t, w1s_real, label='w1 RW')
# axes0[0].plot(t, w2s_real, label='w2 RW')
# axes0[0].set_xlabel('Tiempo [s]')
# axes0[0].set_ylabel('velocidad angular [rad/s]')
# axes0[0].legend()
# axes0[0].set_title('velocidades angulares de las tres ruedas de reaccion')
# axes0[0].grid()
# # axes0[0].set_ylim(-1, 1)  # Ajusta los límites en el eje Y

# axes0[1].plot(t, w0s_est, label='w0 RW estimada')
# axes0[1].plot(t, w1s_est, label='w1 RW estimada')
# axes0[1].plot(t, w2s_est, label='w2 RW estimada')
# axes0[1].set_xlabel('Tiempo [s]')
# axes0[1].set_ylabel('velocidad angular [rad/s]')
# axes0[1].legend()
# axes0[1].set_title('velocidades angulares estimados en las tres ruedas de reaccion')
# axes0[1].grid()
# plt.tight_layout()
# plt.show()

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

fig0, axes0 = plt.subplots(nrows=4, ncols=1, figsize=(18, 10))

axes0[0].plot(t[0:len(RPY_all_id[:,0])], RPY_all_id[:,0], label='Roll modelo')
axes0[0].plot(t[0:len(RPY_all_id[:,0])], RPY_all_id[:,1], label='Pitch modelo')
axes0[0].plot(t[0:len(RPY_all_id[:,0])], RPY_all_id[:,2], label='Yaw modelo')
axes0[0].set_xlabel('Tiempo [s]')
axes0[0].set_ylabel('Angulos de Euler [°]')
axes0[0].legend()
axes0[0].set_title('Angulos de Euler obtenidos por el modelo de control lineal discreto')
axes0[0].grid()

axes0[1].plot(t[0:len(RPY_all_est[:,0])], RPY_all_est[:,0], label='Roll kalman')
axes0[1].plot(t[0:len(RPY_all_est[:,0])], RPY_all_est[:,1], label='Pitch kalman')
axes0[1].plot(t[0:len(RPY_all_est[:,0])], RPY_all_est[:,2], label='Yaw kalman')
axes0[1].set_xlabel('Tiempo [s]')
axes0[1].set_ylabel('Angulos de Euler [°]')
axes0[1].legend()
axes0[1].set_title('Angulos de Euler estimados por el filtro de kalman lineal discreto')
axes0[1].grid()

axes0[2].plot(t[0:len(RPY_all_id[:,0])], RPY_all_id[:,0], label='Roll modelo')
axes0[2].plot(t[0:len(RPY_all_id[:,0])], RPY_all_id[:,1], label='Pitch modelo')
axes0[2].plot(t[0:len(RPY_all_id[:,0])], RPY_all_id[:,2], label='Yaw modelo')
axes0[2].set_xlabel('Tiempo [s]')
axes0[2].set_ylabel('Angulos de Euler [°]')
axes0[2].legend()
axes0[2].set_title('Angulos de Euler obtenidos por el modelo de control lineal discreto')
axes0[2].grid()
axes0[2].set_ylim(-5, 5)  # Ajusta los límites en el eje Y

axes0[3].plot(t[0:len(RPY_all_est[:,0])], RPY_all_est[:,0], label='Roll kalman')
axes0[3].plot(t[0:len(RPY_all_est[:,0])], RPY_all_est[:,1], label='Pitch kalman')
axes0[3].plot(t[0:len(RPY_all_est[:,0])], RPY_all_est[:,2], label='Yaw kalman')
axes0[3].set_xlabel('Tiempo [s]')
axes0[3].set_ylabel('Angulos de Euler [°]')
axes0[3].legend()
axes0[3].set_title('Angulos de Euler estimados por el filtro de kalman lineal discreto')
axes0[3].grid()
axes0[3].set_ylim(-5, 5)  # Ajusta los límites en el eje Y

plt.tight_layout()
plt.show()


# R_P_Y = np.array([0,1,2])
# plt.figure(figsize=(12, 6))
# plt.scatter(R_P_Y [0], mse_roll, label='mse roll', color='r',marker='*')
# plt.scatter(R_P_Y [1], mse_pitch, label='mse pitch', color='b',marker='*')
# plt.scatter(R_P_Y [2], mse_yaw, label='mse yaw', color='k',marker='*')
# plt.xlabel('Angulos de Euler')
# plt.ylabel('Mean Square Error [°]')
# plt.legend()
# plt.title('MSE de cada angulo de Euler entre lineal discreto y kalman lineal discreto')
# # plt.xlim(20000,100000)
# # plt.ylim(-0.005,0.005)
# plt.grid()
# plt.show()


#%% Guardar resultados en un .csv

# Nombre del archivo basado en las opciones seleccionadas
nombre_archivo = f"_sen{opcion}_act{opcion_tau}_RW.csv"

# Crear un diccionario con los datos
datos = {
    'Tiempo': t,
    'Roll_est': RPY_all_est[:,0],
    'Pitch_est': RPY_all_est[:,1],
    'Yaw_est': RPY_all_est[:,2],
    'q0_real': q0_real,
    'q1_real': q1_real,
    'q2_real': q2_real,
    'q3_real': q3_real,
    'q0_est': q0_est,
    'q1_est': q1_est,
    'q2_est': q2_est,
    'q3_est': q3_est,
    'w0_est': w0_est,
    'w1_est': w1_est,
    'w2_est': w2_est,
    'w0_real': w0_real,
    'w1_real': w1_real,
    'w2_real': w2_real,
    'Roll_real': RPY_all_id[:,0],
    'Pitch_real': RPY_all_id[:,1],
    'Yaw_real': RPY_all_id[:,2],
}

# Crear un DataFrame de pandas a partir del diccionario
df_resultados = pd.DataFrame(datos)

# Guardar el DataFrame en un archivo CSV
df_resultados.to_csv(nombre_archivo, index=False)

print(f"Los resultados se han guardado en {nombre_archivo}")


#%% Control LQR

# Definir las matrices Q y R del coste del LQR
# diag_Q = np.array([100, 1000000, 10000, 0.1, 0.1, 0.10, 0.01, 10, 10])*10000
# diag_R = np.array([0.1,0.1,0.1])*100000
# diag_Q = np.array([100000, 100000, 1000000, 10000, 10000, 10000, 1000000, 1000000, 1000000])*1000
# diag_R = np.array([0.1,10,0.1])*10
# diag_Q = np.array([10, 10, 10, 0.1, 0.1, 0.1, 10, 10, 10, 10, 10, 10, 0.1, 0.1, 0.1])
# diag_R = np.array([diag_Q[6],diag_Q[7],diag_Q[8]])
# Q = np.diag(diag_Q)
# R = np.diag(diag_R)

# Resolver la ecuación de Riccati
# P = solve_discrete_are(A_discrete, B_discrete, Q, R)

# Calcular la matriz de retroalimentación K
# K = np.linalg.in1v(B_discrete.T @ P @ B_discrete + R) @ (B_discrete.T @ P @ A_discrete)
# K, P, eigenvalues = control.dlqr(A_discrete, B_discrete, Q, R)

# us = []