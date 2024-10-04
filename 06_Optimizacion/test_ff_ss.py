# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 01:47:40 2024

@author: nachi
"""
import functions_06
import functions_06_rw
import numpy as np
import pandas as pd
import control as ctrl
import matplotlib.pyplot as plt

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
q_real = np.array([0.0789,0.0941,0.0789,0.9893])

deltat = 2
# limite =  5762*69
limite =  5762*4
hh =0.01
bi_orbit = [Bx_orbit[0],By_orbit[0],Bz_orbit[0]]
b_body_i = functions_06_rw.rotacion_v(q_real, bi_orbit, 0)

si_orbit = [vx_sun_orbit[0],vy_sun_orbit[0],vz_sun_orbit[0]]
s_body_i = functions_06_rw.rotacion_v(q_real, si_orbit, 0)
t = np.arange(0, limite, deltat)
w0_O = 0.00163

I_x = 0.037
I_y = 0.036
I_z = 0.006


[A,B,C,A_discrete,B_discrete,C_discrete] = functions_06.A_B(I_x, I_y, I_z, w0_O, 0, 0, 0, deltat, hh, bi_orbit,b_body_i, s_body_i)
B_matrices = [B_discrete]

for i in range(len(t[0:2882])-1):

    # print(t[i+1])
        
    b_orbit = [Bx_orbit[i+1],By_orbit[i+1],Bz_orbit[i+1]]

    [A,B,C,A_discrete,B_discrete,C_discrete] = functions_06.A_B(I_x, I_y, I_z, w0_O, 0, 0, 0, deltat, hh,b_orbit, np.zeros(3), np.zeros(3))

    B_matrices.append(B_discrete)

Bs_a = np.array(B_matrices)

B_concanate = []    
for ii in range(len(Bs_a[0,:,0])):
    for jj in range(len(Bs_a[0,0,:])):
        B_concanate.append(np.sum(Bs_a[:,ii,jj]) / len(Bs_a[:,ii,jj]))

B_prom = np.vstack((B_concanate[0:3],B_concanate[3:6],B_concanate[6:9],B_concanate[9:12],B_concanate[12:15],B_concanate[15:18]))

C = np.eye(6)
D = np.zeros((6, 3))
sys_ss = ctrl.ss(A_discrete, B_prom, C, D)
sys_tf = ctrl.ss2tf(sys_ss)

#%%

# Obtener el número de filas y columnas del sistema de funciones de transferencia
num_outputs = 6
num_inputs = 3

# Recorrer cada función de transferencia en el sistema
for i in range(num_outputs):  # Para cada salida
    for j in range(num_inputs):  # Para cada entrada
        sys_tf_ij = sys_tf[i, j]
        
        # Definir el tiempo de simulación
        t_impulse = np.linspace(0, 100, 1000)  # 100 segundos, 1000 puntos

        # Obtener la respuesta a un impulso
        t_out, y_out = ctrl.impulse_response(sys_tf_ij, T=t_impulse)

        # Graficar la respuesta a un impulso
        plt.figure()
        plt.plot(t_out, y_out)
        plt.title(f'Respuesta a un Impulso de sys_tf[{i}, {j}]')
        plt.xlabel('Tiempo [s]')
        plt.ylabel('Amplitud')
        plt.grid(True)
        plt.show()
# # Definir el sistema de la función de transferencia que quieres analizar
# sys_tf_00 = sys_tf[0, 2]

# # Definir el tiempo de simulación
# t_impulse = np.linspace(0, 100, 1000)  # 10 segundos, 1000 puntos

# # Obtener la respuesta a un impulso
# t_out, y_out = ctrl.impulse_response(sys_tf_00, T=t_impulse)

# # Graficar la respuesta a un impulso
# plt.figure()
# plt.plot(t_out, y_out)
# plt.title(f'Respuesta a un Impulso de sys_tf{sys_tf_00}')
# plt.xlabel('Tiempo [s]')
# plt.ylabel('Amplitud')
# plt.grid(True)
# plt.show()