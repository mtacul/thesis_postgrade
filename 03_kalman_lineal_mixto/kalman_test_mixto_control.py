# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 15:13:44 2024

@author: nachi
"""
import functions_03
import numpy as np
import control as ctrl
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from sklearn.decomposition import PCA

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
# limite =  5762*69
limite =  5762*69

t = np.arange(0, limite, deltat)

I_x = 0.037
I_y = 0.036
I_z = 0.006

w0_eq = 0
w1_eq = 0
w2_eq = 0
#%% seleccion de nivel de sensor

# Definir los valores
sigma_ss_values = {
    1: np.sin(0.833*np.pi/180),
    2: np.sin(0.167*np.pi/180),
    3: np.sin(0.05*np.pi/180),
    4: 0
}

sigma_b_values = {
    1: 1.18e-6,
    2: 0.1e-6,
    3: 0.012e-6,
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
opcion = int(input("Seleccione una opción (1: bad, 2: med, 3: good, 4:no ruido): "))

# Asignar los valores seleccionados
sigma_ss = sigma_ss_values[opcion]
sigma_b = sigma_b_values[opcion]
ruido_w = ruido_w_values[opcion]
bias_w = bias_w_values[opcion]

#%% Seleccion de nivel de actuador

# Definir los valores
lim_tau_values = {
    1: 0.24,
    2: 1.19,
    3: 15
}

# Solicitar al usuario que seleccione una opción
opcion_tau = int(input("Seleccione una opción (1: bad, 2: med, 3: good): "))

# Asignar los valores seleccionados
lim = lim_tau_values[opcion_tau]


#%%
# q= np.array([0,0.7071,0,0.7071])
# q= np.array([0,0,0,1])
q = np.array([0.7071/np.sqrt(3),0.7071/np.sqrt(3),0.7071/np.sqrt(3),0.7071])
w = np.array([0.0001, 0.0001, 0.0001])
# q_est = np.array([0.00985969, 0.70703804, 0.00985969, 0.70703804])
# q_est= np.array([0.0120039,0.0116517,0.0160542,0.999731])
q_est = np.array([0.366144,0.464586,0.300017,0.74839])

q0_est = [q_est[0]]
q1_est = [q_est[1]]
q2_est = [q_est[2]]
q3_est = [q_est[3]]
w0_est = [w[0]]
w1_est = [w[1]]
w2_est = [w[2]]
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
w_gyros = functions_03.simulate_gyros_reading(w_body, 0,0)
x_real = np.hstack((np.transpose(q_real[:3]), np.transpose(w_gyros)))

bi_orbit = [Bx_orbit[0],By_orbit[0],Bz_orbit[0]]
b_body_i = functions_03.rotacion_v(q_real, bi_orbit, sigma_b)

si_orbit = [vx_sun_orbit[0],vy_sun_orbit[0],vz_sun_orbit[0]]
s_body_i = functions_03.rotacion_v(q_real, si_orbit, sigma_ss)
hh =0.01

[A,B,C,A_discrete,B_discrete,C_discrete] = functions_03.A_B(I_x, I_y, I_z, w0_O, w0_eq, w1_eq, w2_eq, deltat, hh, bi_orbit,b_body_i, s_body_i)

B_matrices = [B_discrete]
#%%
# for i in range(len(t[0:2882])-1):
for i in range(len(t)-1):

    print(t[i+1])
    
    # u_est = np.array([15,15,15])

    b_orbit = [Bx_orbit[i+1],By_orbit[i+1],Bz_orbit[i+1]]

    [A,B,C,A_discrete,B_discrete,C_discrete] = functions_03.A_B(I_x, I_y, I_z, w0_O, w0_eq, w1_eq, w2_eq, deltat, hh,b_orbit, np.zeros(3), np.zeros(3))

    B_matrices.append(B_discrete)

Bs_a = np.array(B_matrices)

B_concanate = []    
for ii in range(len(Bs_a[0,:,0])):
    for jj in range(len(Bs_a[0,0,:])):
        B_concanate.append(np.sum(Bs_a[:,ii,jj]) / len(Bs_a[:,ii,jj]))

B_prom = np.vstack((B_concanate[0:3],B_concanate[3:6],B_concanate[6:9],B_concanate[9:12],B_concanate[12:15],B_concanate[15:18]))

# x0 = np.array([37.0034,
# 85.5048,
# 51.4523,
# 65.333,
# 25.471,
# 74.9946,])

# optimal_x = functions_03.opt_K(A_discrete, B_prom, deltat, hh, x0)
# K = np.hstack([np.diag(optimal_x[:3]), np.diag(optimal_x[3:])])

xx = np.array([335.118,
253.109,
448.579,
375.116,
335.062,
510.586
])

# xx = np.array([37.0034,
# 85.5048,
# 51.4523,
# 65.333,
# 25.471,
# 74.9946,])

K = np.hstack([np.diag(xx[:3]), np.diag(xx[3:])])

diagonal_values = np.array([0.5**2, 0.5**2, 0.5**2, 0.1**2, 0.1**2, 0.1**2])
P_ki = np.diag(diagonal_values)
#%%
np.random.seed(42)
for i in range(len(t)-1):
    print(t[i+1])
    q_est = np.array([q0_est[-1], q1_est[-1], q2_est[-1], q3_est[-1]])
    w_est = np.array([w0_est[-1], w1_est[-1], w2_est[-1]])
    x_est = np.hstack((np.transpose(q_est[:3]), np.transpose(w_est)))

    u_est = np.dot(K,x_est)
    u_est = functions_03.torquer(u_est,lim)

    x_est = np.hstack((q_est[0:3],w_est))  

    b_orbit = [Bx_orbit[i],By_orbit[i],Bz_orbit[i]]
    b_body_med = functions_03.rotacion_v(q_real, b_orbit,sigma_b)
    b_body_est = functions_03.rotacion_v(q_est, b_orbit,sigma_b)
    
    s_orbit = [vx_sun_orbit[i],vy_sun_orbit[i],vz_sun_orbit[i]]
    s_body_med = functions_03.rotacion_v(q_real, s_orbit,sigma_ss)
    s_body_est = functions_03.rotacion_v(q_est, s_orbit,sigma_ss)
    
    [A,B,C,A_discrete,B_discrete,C_discrete] = functions_03.A_B(I_x, I_y, I_z, w0_O, w0_eq, w1_eq, w2_eq, deltat, hh,b_orbit, b_body_med, s_body_med)
    
    if opcion == 4:
        [q_posteriori, w_posteriori, P_k_pos,K_k] = functions_03.kalman_lineal(A_discrete, B_discrete,C_discrete, x_est, u_est, b_body_med, b_body_est, s_body_med, s_body_est, P_ki, 0.012e-6, 0.05, deltat,hh)
    elif opcion == 1 or opcion == 2 or opcion == 3:
        [q_posteriori, w_posteriori, P_k_pos,K_k] = functions_03.kalman_lineal(A_discrete, B_discrete,C_discrete, x_est, u_est, b_body_med, b_body_est, s_body_med, s_body_est, P_ki, sigma_b, sigma_ss, deltat,hh)

    q0_est.append(q_posteriori[0])
    q1_est.append(q_posteriori[1])
    q2_est.append(q_posteriori[2])
    q3_est.append(q_posteriori[3])
    w0_est.append(w_posteriori[0])
    w1_est.append(w_posteriori[1])
    w2_est.append(w_posteriori[2])

    P_ki = P_k_pos
    
    [xx_new_d, qq3_new_d] = functions_03.mod_lineal_disc(
        x_real, u_est, deltat, hh, A_discrete,B_prom)
    
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
    print("q_real",q_real)


[MSE_cuat, MSE_omega]  = functions_03.cuat_MSE_NL(q0_real, q1_real, q2_real, q3_real, w0_real, w1_real, w2_real, q0_est, q1_est, q2_est, q3_est, w0_est, w1_est, w2_est)   
[RPY_all_est,RPY_all_id,mse_roll,mse_pitch,mse_yaw] = functions_03.RPY_MSE(t, q0_est, q1_est, q2_est, q3_est, q0_real, q1_real, q2_real, q3_real)   
    
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
axes0[2].set_ylim(-10, 10)  # Ajusta los límites en el eje Y

axes0[3].plot(t[0:len(RPY_all_est[:,0])], RPY_all_est[:,0], label='Roll kalman')
axes0[3].plot(t[0:len(RPY_all_est[:,0])], RPY_all_est[:,1], label='Pitch kalman')
axes0[3].plot(t[0:len(RPY_all_est[:,0])], RPY_all_est[:,2], label='Yaw kalman')
axes0[3].set_xlabel('Tiempo [s]')
axes0[3].set_ylabel('Angulos de Euler [°]')
axes0[3].legend()
axes0[3].set_title('Angulos de Euler estimados por el filtro de kalman lineal discreto')
axes0[3].grid()
axes0[3].set_ylim(-10, 10)  # Ajusta los límites en el eje Y

plt.tight_layout()
plt.show()

#%%
# Nombre del archivo basado en las opciones seleccionadas
nombre_archivo = f"_sen{opcion}_act{opcion_tau}_MT.csv"

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