# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 15:13:44 2024

@author: nachi
"""
import functions_03
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_are

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
# limite =  5762*5
limite = 5762*0.1
t = np.arange(0, limite, deltat)

#%% Parámetros geométricos y orbitales dados

# w0_O = 0.00163

I_x = 0.037
I_y = 0.036
I_z = 0.006

w0_O = 7.292115e-5 #rad/s
# I_x = 1030.17 #kgm^2
# I_y = 3015.65 #kgm^2
# I_z = 3030.43 #kgm^2

#%% Selección de nivel de sensor

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
    # 2: 0,
    2: 0.050 * np.pi / 180,
    3: 0.033 * np.pi / 180,
    4: 0
}

bias_w_values = {
    1: (0.05 * np.pi / 180) / 3600,
    # 2: 0,
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
    1: 0.5,
    2: 1.19,
    3: 15
}

# Solicitar al usuario que seleccione una opción
opcion_tau = int(input("Seleccione un nivel de actuador (1: bad, 2: med, 3: good): "))

# Asignar los valores seleccionados
lim = lim_tau_values[opcion_tau]


#%% Condiciones iniciales reales y estimadas

# q= np.array([0,0.7071,0,0.7071])
# q= np.array([0,0,0,1])
q = np.array([0.0789,0.0941,0.0789,0.9893])
w = np.array([0.0001, 0.0001, 0.0001])*100
# q_est = np.array([0.00985969, 0.70703804, 0.00985969, 0.70703804])
# q_est= np.array([0.0120039,0.0116517,0.0160542,0.999731])
q_est = np.array([0.0789,0.0941,0.0789,0.9893])

q0_est = [q_est[0]]
q1_est = [q_est[1]]
q2_est = [q_est[2]]
q3_est = [q_est[3]]
w0_est = [w[0]]
w1_est = [w[1]]
w2_est = [w[2]]

q0_real = [q[0]]
q1_real = [q[1]]
q2_real = [q[2]]
q3_real = [q[3]]
w0_real = [w[0]]
w1_real = [w[1]]
w2_real = [w[2]]
q_real = np.array([q0_real[-1],q1_real[-1],q2_real[-1],q3_real[-1]])
w_body = np.array([w0_real[-1], w1_real[-1], w2_real[-1]])
w_gyros = functions_03.simulate_gyros_reading(w_body, 0,0)
x_real = np.hstack((np.transpose(q_real[:3]), np.transpose(w_gyros)))

bi_orbit = [Bx_orbit[0],By_orbit[0],Bz_orbit[0]]
b_body_i = functions_03.rotacion_v(q_real, bi_orbit, 1e-6)

si_orbit = [vx_sun_orbit[0],vy_sun_orbit[0],vz_sun_orbit[0]]
s_body_i = functions_03.rotacion_v(q_real, si_orbit, 0.036)
hh = deltat
hh_mod = 0.2

#%% Obtencion de un B_prom representativo

[A,B,C,A_discrete,B_discrete,C_discrete] = functions_03.A_B(I_x, I_y, I_z, w0_O, 0, 0, 0, deltat, hh, bi_orbit,b_body_i, s_body_i)
[A_mod,B_mod,C_mod,A_discrete_mod,B_discrete_mod,C_discrete_mod] = functions_03.A_B(I_x, I_y, I_z, w0_O, 0, 0, 0, deltat, hh_mod, bi_orbit,b_body_i, s_body_i)

B_matrices = [B_discrete]
B_matrices_mod = [B_discrete_mod]

for i in range(len(t[0:2882])-1):
# for i in range(len(t)-1):

    # print(t[i+1])
    
    # u_est = np.array([15,15,15])

    b_orbit = [Bx_orbit[i+1],By_orbit[i+1],Bz_orbit[i+1]]

    [A,B,C,A_discrete,B_discrete,C_discrete] = functions_03.A_B(I_x, I_y, I_z, w0_O, 0, 0, 0, deltat, hh,b_orbit, np.zeros(3), np.zeros(3))
    [A_mod,B_mod,C_mod,A_discrete_mod,B_discrete_mod,C_discrete_mod] = functions_03.A_B(I_x, I_y, I_z, w0_O, 0, 0, 0, deltat, hh_mod,b_orbit, np.zeros(3), np.zeros(3))

    B_matrices.append(B_discrete)
    B_matrices_mod.append(B_discrete_mod)
    
Bs_a = np.array(B_matrices)
Bs_a_mod = np.array(B_matrices_mod)

B_concanate = [] 
B_concanate_mod = []    
   
for ii in range(len(Bs_a[0,:,0])):
    for jj in range(len(Bs_a[0,0,:])):
        B_concanate.append(np.sum(Bs_a[:,ii,jj]) / len(Bs_a[:,ii,jj]))
        B_concanate_mod.append(np.sum(Bs_a_mod[:,ii,jj]) / len(Bs_a_mod[:,ii,jj]))

B_prom = np.vstack((B_concanate[0:3],B_concanate[3:6],B_concanate[6:9],B_concanate[9:12],B_concanate[12:15],B_concanate[15:18]))
B_prom_mod = np.vstack((B_concanate_mod[0:3],B_concanate_mod[3:6],B_concanate_mod[6:9],B_concanate_mod[9:12],B_concanate_mod[12:15],B_concanate_mod[15:18]))

#%% Control LQR

## Definir las matrices Q y R del coste del LQR (antes del cagazo)
# diag_Q = np.array([100, 1000000, 10000, 0.1, 0.1, 0.10, 0.01, 10, 10])*10000
# diag_R = np.array([0.1,0.1,0.1])*100000 
# 
# diag_Q = np.array([100, 10, 100, 0.1, 0.1, 0.1])*1000000
# diag_R = np.array([0.1,0.1,0.1])*100000

# diag_Q = np.array([10, 10, 10, 1000, 1000, 1000])
# diag_R = np.array([0.1,0.1,0.1])*10

diag_Q = np.array([10000, 10000, 10000, 100000, 100000, 100000])
diag_R = np.array([0.1,0.1,0.1])*10

Q = np.diag(diag_Q)
R = np.diag(diag_R)

# Resolver la ecuación de Riccati
P = solve_discrete_are(A_discrete, B_prom, Q, R)

# Calcular la matriz de retroalimentación K
K = np.linalg.inv(B_prom.T @ P @ B_prom + R) @ (B_prom.T @ P @ A_discrete)

#%% Simulacion dinamica de actitud

diagonal_values = np.array([0.5**2, 0.5**2, 0.5**2, 0.1**2, 0.1**2, 0.1**2])
P_ki = np.diag(diagonal_values)
np.random.seed(42)
us = []
for i in range(len(t)-1):
    # print(t[i+1])
    # print(x_real)
    # print(np.dot(-K,x_real))

    q_est = np.array([q0_est[-1], q1_est[-1], q2_est[-1], q3_est[-1]])
    w_est = np.array([w0_est[-1], w1_est[-1], w2_est[-1]])
    x_est = np.hstack((np.transpose(q_est[:3]), np.transpose(w_est)))
    u_est = np.dot(-K,x_est)
    # print(u_est)
    u_est = functions_03.torquer(u_est,lim)
    us.append(u_est)
    [xx_new_d, qq3_new_d] = functions_03.mod_lineal_disc(
        x_real, u_est, deltat, hh_mod, A_discrete_mod,B_prom_mod)
    
    # print(u_est)
    # print(xx_new_d)
    # input()
    x_real = xx_new_d
    w_gyros = functions_03.simulate_gyros_reading(x_real[3:6],ruido_w,bias_w)
    q0_real.append(xx_new_d[0])
    q1_real.append(xx_new_d[1])
    q2_real.append(xx_new_d[2])
    q3_real.append(qq3_new_d)
    w0_real.append(w_gyros[0])
    w1_real.append(w_gyros[1])
    w2_real.append(w_gyros[2])

    q_real = np.array([q0_real[-1], q1_real[-1], q2_real[-1], q3_real[-1]])

    b_orbit = [Bx_orbit[i+1],By_orbit[i+1],Bz_orbit[i+1]]
    b_body_med = functions_03.rotacion_v(q_real, b_orbit,sigma_b)
    
    s_orbit = [vx_sun_orbit[i+1],vy_sun_orbit[i+1],vz_sun_orbit[i+1]]
    s_body_med = functions_03.rotacion_v(q_real, s_orbit,sigma_ss)

    [A,B,C,A_discrete,B_discrete,C_discrete] = functions_03.A_B(I_x, I_y, I_z, w0_O, 0, 0, 0, deltat, hh,b_orbit, b_body_med, s_body_med)
    
    if opcion == 4:
        [q_posteriori, w_posteriori, P_k_pos,K_k] = functions_03.kalman_lineal(A_discrete, B_prom,C_discrete, x_est, u_est, b_orbit,b_body_med, s_orbit,s_body_med, P_ki, sigma_b,sigma_ss, deltat,hh,1, 1)
    elif opcion == 1 or opcion == 2 or opcion == 3:
        [q_posteriori, w_posteriori, P_k_pos,K_k] = functions_03.kalman_lineal(A_discrete, B_prom,C_discrete, x_est, u_est, b_orbit,b_body_med, s_orbit,s_body_med, P_ki, sigma_b,sigma_ss, deltat,hh,sigma_b,sigma_ss)

    q0_est.append(q_posteriori[0])
    q1_est.append(q_posteriori[1])
    q2_est.append(q_posteriori[2])
    q3_est.append(q_posteriori[3])
    w0_est.append(w_posteriori[0])
    w1_est.append(w_posteriori[1])
    w2_est.append(w_posteriori[2])
    # q0_est.append(q0_real[-1])
    # q1_est.append(q1_real[-1])
    # q2_est.append(q2_real[-1])
    # q3_est.append(q3_real[-1])
    # w0_est.append(w0_real[-1])
    # w1_est.append(w1_real[-1])
    # w2_est.append(w2_real[-1])

    P_ki = P_k_pos
    
    
    
[MSE_cuat, MSE_omega]  = functions_03.cuat_MSE_NL(q0_real, q1_real, q2_real, q3_real, w0_real, w1_real, w2_real, q0_est, q1_est, q2_est, q3_est, w0_est, w1_est, w2_est)   
[RPY_all_est,RPY_all_id,mse_roll,mse_pitch,mse_yaw] = functions_03.RPY_MSE(t, q0_est, q1_est, q2_est, q3_est, q0_real, q1_real, q2_real, q3_real)   
    
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

fig0, axes0 = plt.subplots(nrows=1, ncols=2, figsize=(16, 4))

axes0[0].plot(t, RPY_all_id[:,0], label='Roll modelo')
axes0[0].plot(t, RPY_all_id[:,1], label='Pitch modelo')
axes0[0].plot(t, RPY_all_id[:,2], label='Yaw modelo')
axes0[0].set_xlabel('Tiempo [s]')
axes0[0].set_ylabel('Angulos de Euler [°]')
axes0[0].legend()
axes0[0].set_title('Angulos de Euler obtenidos por el modelo de control lineal discreto')
axes0[0].grid()
# axes0[0].set_xlim(0, 10000)  # Ajusta los límites en el eje Y

# axes0[0].set_ylim(-5, 5)  # Ajusta los límites en el eje Y

axes0[1].plot(t, RPY_all_est[:,0], label='Roll kalman')
axes0[1].plot(t, RPY_all_est[:,1], label='Pitch kalman')
axes0[1].plot(t, RPY_all_est[:,2], label='Yaw kalman')
axes0[1].set_xlabel('Tiempo [s]')
axes0[1].set_ylabel('Angulos de Euler [°]')
axes0[1].legend()
axes0[1].set_title('Angulos de Euler estimados por el filtro de kalman lineal discreto')
axes0[1].grid()
plt.tight_layout()
plt.show()

fig0, axes0 = plt.subplots(nrows=3, ncols=1, figsize=(13, 8))

axes0[0].plot(t, RPY_all_est[:,0], label= {'magnetorquer'})
axes0[0].set_xlabel('Tiempo [s]')
axes0[0].set_ylabel('Roll [°]')
axes0[0].legend()
axes0[0].grid()
axes0[0].set_xlim(0, 30000)  # Ajusta los límites en el eje Y
axes0[0].set_ylim(-10, 10)  # Ajusta los límites en el eje Y

axes0[1].plot(t, RPY_all_est[:,1], label={'magnetorquer'},color='orange')
axes0[1].set_xlabel('Tiempo [s]')
axes0[1].set_ylabel('Pitch [°]')
axes0[1].legend()
axes0[1].grid()
axes0[1].set_xlim(0, 30000) # Ajusta los límites en el eje Y
axes0[1].set_ylim(-10, 10)  # Ajusta los límites en el eje Y

axes0[2].plot(t, RPY_all_est[:,2], label={'magnetorquer'},color='green')
axes0[2].set_xlabel('Tiempo [s]')
axes0[2].set_ylabel('Yaw [°]')
axes0[2].legend()
axes0[2].grid()
axes0[2].set_xlim(0, 30000)  # Ajusta los límites en el eje Y
axes0[2].set_ylim(-10, 10)  # Ajusta los límites en el eje Y

plt.tight_layout()

# Save the plot as an SVG file
# plt.savefig('plot.svg', format='svg')

# Show the plot (optional)
plt.show()

fig0, axes0 = plt.subplots(nrows=3, ncols=1, figsize=(13, 8))

axes0[0].plot(t, RPY_all_id[:,0], label= {'magnetorquer'})
axes0[0].set_xlabel('Tiempo [s]')
axes0[0].set_ylabel('Roll [°]')
axes0[0].legend()
axes0[0].grid()
# axes0[0].set_xlim(0, 30000)  # Ajusta los límites en el eje Y

axes0[1].plot(t, RPY_all_id[:,1], label={'magnetorquer'},color='orange')
axes0[1].set_xlabel('Tiempo [s]')
axes0[1].set_ylabel('Pitch [°]')
axes0[1].legend()
axes0[1].grid()
# axes0[1].set_xlim(0, 30000) # Ajusta los límites en el eje Y
# axes0[1].set_ylim(-20, -5)  # Ajusta los límites en el eje Y

axes0[2].plot(t, RPY_all_id[:,2], label={'magnetorquer'},color='green')
axes0[2].set_xlabel('Tiempo [s]')
axes0[2].set_ylabel('Yaw [°]')
axes0[2].legend()
axes0[2].grid()
# axes0[2].set_xlim(0, 30000)  # Ajusta los límites en el eje Y

plt.tight_layout()

# Save the plot as an SVG file
# plt.savefig('plot.svg', format='svg')

# Show the plot (optional)
plt.show()
#%% Guardar resultados en un .csv

# Nombre del archivo basado en las opciones seleccionadas
nombre_archivo = f"_sen{opcion}_act{opcion_tau}_MT_LQR.csv"

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