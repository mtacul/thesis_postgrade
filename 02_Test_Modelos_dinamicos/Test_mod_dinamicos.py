# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 11:05:12 2024

@author: nachi
"""
#%%
import functions_02
import numpy as np   
import pandas as pd
import matplotlib.pyplot as plt

#%% Cargar datos del .csv obtenido

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
limite = 4002
# limite = 72
t = np.arange(0, limite, deltat)
																																																				
#%% Condiciones iniciales y parámetros geométricos dados

# q = np.array([-0.7071/np.sqrt(3),0.7071/np.sqrt(3),-0.7071/np.sqrt(3),0.7071])
q = np.array([0,0,0,1])
w= np.array([0.0001,0.0001,0.0001])

x = np.hstack((np.transpose(q[:3]), np.transpose(w)))

u = np.array([0.15,0.15,0.15])
u_disc = np.array([0.15,0.15,0.15])

w0_O = 0.00163

I_x = 0.037
I_y = 0.036
I_z = 0.006

w0 = 0
w1 = 0
w2 = 0

hh =0.01

#%% Simulación de los modelos dinámico lineales y no lineales

q0 = [q[0]]
q1 = [q[1]]
q2 = [q[2]]
q3 = [q[3]]
w0 = [w[0]]
w1 = [w[1]]
w2 = [w[2]]

q0_disc = [q[0]]
q1_disc = [q[1]]
q2_disc = [q[2]]
q3_disc = [q[3]]
w0_disc = [w[0]]
w1_disc = [w[1]]
w2_disc = [w[2]]

q0_nl = [q[0]]
q1_nl = [q[1]]
q2_nl = [q[2]]
q3_nl = [q[3]]
w0_nl = [w[0]]
w1_nl = [w[1]]
w2_nl = [w[2]]


for i in range(len(t)-1):
    print(t[i])
    qq = np.array([q0[-1],q1[-1],q2[-1],q3[-1]])
    ww = np.array([w0[-1],w1[-1],w2[-1]])
    xx = np.hstack((np.transpose(qq[:3]), np.transpose(ww)))
    uu = np.array([15,15,15])
    
    qq_disc = np.array([q0_disc[-1],q1_disc[-1],q2_disc[-1],q3_disc[-1]])
    ww_disc = np.array([w0_disc[-1],w1_disc[-1],w2_disc[-1]])
    xx_disc = np.hstack((np.transpose(qq_disc[:3]), np.transpose(ww_disc)))
    uu_disc = np.array([15,15,15])
    
    qq_nl = np.array([q0_nl[-1],q1_nl[-1],q2_nl[-1],q3_nl[-1]])
    ww_nl = np.array([w0_nl[-1],w1_nl[-1],w2_nl[-1]])
    xx_nl = np.hstack((np.transpose(qq_nl[:3]), np.transpose(ww_nl)))
    uu_nl = np.array([15,15,15])

    bb_orbit = [Bx_orbit[i],By_orbit[i],Bz_orbit[i]]
    ss_orbit = [vx_sun_orbit[i],vy_sun_orbit[i],vz_sun_orbit[i]]
    
    bb_body = functions_02.rotacion_v(qq , bb_orbit, 0)
    ss_body = functions_02.rotacion_v(qq , ss_orbit, 0)
    
    bb_body_disc = functions_02.rotacion_v(qq_disc , bb_orbit, 0)
    ss_body_disc = functions_02.rotacion_v(qq_disc , ss_orbit, 0)
    
    bb_body_nl = functions_02.rotacion_v(qq_nl, bb_orbit, 0)
    ss_body_nl = functions_02.rotacion_v(qq_nl, ss_orbit, 0)
    
    [A,B,C,A_discrete,B_discrete,C_discrete] = functions_02.A_B(I_x, I_y, I_z, w0_O, 0, 0, 0, deltat, hh,bb_orbit, bb_body, ss_body)
    [A_disc,B_disc,C_disc,A_discrete_disc,B_discrete_disc,C_discrete_disc] = functions_02.A_B(I_x, I_y, I_z, w0_O, 0, 0, 0, deltat, hh, bb_orbit,bb_body_disc, ss_body_disc)

    [xx_new, qq3_new] = functions_02.mod_lineal_cont(xx,uu,deltat,hh,A,B)
    [xx_new_disc, qq3_new_d] = functions_02.mod_lineal_disc(xx_disc,uu_disc,deltat,hh,A_discrete_disc,B_discrete_disc)
    [xx_new_nl, qq3_new_nl] = functions_02.mod_nolineal(xx_nl,uu_nl,deltat,bb_body_nl,hh, w0_O,I_x,I_y,I_z)
    
    q0.append(xx_new[0])
    q1.append(xx_new[1])
    q2.append(xx_new[2])
    q3.append(qq3_new)
    w0.append(xx_new[3])
    w1.append(xx_new[4])
    w2.append(xx_new[5])
    
    q0_disc.append(xx_new_disc[0])
    q1_disc.append(xx_new_disc[1])
    q2_disc.append(xx_new_disc[2])
    q3_disc.append(qq3_new_d)
    w0_disc.append(xx_new_disc[3])
    w1_disc.append(xx_new_disc[4])
    w2_disc.append(xx_new_disc[5])
        
    q0_nl.append(xx_new_nl[0])
    q1_nl.append(xx_new_nl[1])
    q2_nl.append(xx_new_nl[2])
    q3_nl.append(qq3_new_nl)
    w0_nl.append(xx_new_nl[3])
    w1_nl.append(xx_new_nl[4])
    w2_nl.append(xx_new_nl[5])
    


#%% Gráficas de los cambios en los cuaterniones con accion de control constante

fig0, axes0 = plt.subplots(nrows=1, ncols=2, figsize=(16, 4))

axes0[0].plot(t[0:len(q0)], q0[0:len(q0)], label='q0 lineal')
axes0[0].plot(t[0:len(q0)], q1[0:len(q0)], label='q1 lineal')
axes0[0].plot(t[0:len(q0)], q2[0:len(q0)], label='q2 lineal')
axes0[0].plot(t[0:len(q0)], q3[0:len(q0)], label='q3 lineal')
axes0[0].set_xlabel('Tiempo [s]')
axes0[0].set_ylabel('cuaternión [-]')
axes0[0].legend()
axes0[0].set_title('cuaterniones llevados a 0 lineales continuos MT')
axes0[0].grid()
# axes0[0].set_xlim(0, 2)  # Ajusta los límites en el eje x

axes0[1].plot(t[0:len(q0)], q0_disc[0:len(q0)], label='q0 discreto lineal')
axes0[1].plot(t[0:len(q0)], q1_disc[0:len(q0)], label='q1 discreto lineal')
axes0[1].plot(t[0:len(q0)], q2_disc[0:len(q0)], label='q2 discreto lineal')
axes0[1].plot(t[0:len(q0)], q3_disc[0:len(q0)], label='q3 discreto lineal')
axes0[1].set_xlabel('Tiempo [s]')
axes0[1].set_ylabel('cuaternion [-]')
axes0[1].legend()
axes0[1].set_title('cuaterniones llevados a 0 lineales discretos MT')
axes0[1].grid()
# axes0[1].set_xlim(0, 2)  # Ajusta los límites en el eje x

plt.tight_layout()
plt.show()

fig0, axes0 = plt.subplots(nrows=1, ncols=2, figsize=(16, 4))

axes0[0].plot(t[0:len(q0_nl)], q0[0:len(q0_nl)], label='q0 lineal')
axes0[0].plot(t[0:len(q0_nl)], q1[0:len(q0_nl)], label='q1 lineal')
axes0[0].plot(t[0:len(q0_nl)], q2[0:len(q0_nl)], label='q2 lineal')
axes0[0].plot(t[0:len(q0_nl)], q3[0:len(q0_nl)], label='q3 lineal')
axes0[0].set_xlabel('Tiempo [s]')
axes0[0].set_ylabel('cuaternión [-]')
axes0[0].legend()
axes0[0].set_title('cuaterniones llevados a 0 lineales continuos MT')
axes0[0].grid()
# axes0[0].set_xlim(0, 2)  # Ajusta los límites en el eje x

axes0[1].plot(t[0:len(q0_nl)], q0_nl[0:len(q0_nl)], label='q0 no lineal')
axes0[1].plot(t[0:len(q0_nl)], q1_nl[0:len(q0_nl)], label='q1 no lineal')
axes0[1].plot(t[0:len(q0_nl)], q2_nl[0:len(q0_nl)], label='q2 no lineal')
axes0[1].plot(t[0:len(q0_nl)], q3_nl[0:len(q0_nl)], label='q3 no lineal')
axes0[1].set_xlabel('Tiempo [s]')
axes0[1].set_ylabel('cuaternion [-]')
axes0[1].legend()
axes0[1].set_title('cuaterniones llevados a 0 no lineales continuos MT')
axes0[1].grid()
# axes0[1].set_xlim(0, 2)  # Ajusta los límites en el eje x

plt.tight_layout()
plt.show()
#%% Calculo del Mean Square Error

[MSE_cuat_nl,MSE_omega_nl] = functions_02.cuat_MSE_NL(q0, q1, q2, q3, w0, w1, w2, q0_nl, q1_nl, q2_nl, q3_nl, w0_nl, w1_nl, w2_nl)
[MSE_cuat_disc,MSE_omega_disc] = functions_02.cuat_MSE_NL(q0, q1, q2, q3, w0, w1, w2, q0_disc, q1_disc, q2_disc, q3_disc, w0_disc, w1_disc, w2_disc)

quats = np.array([0,1,2,3])
plt.figure(figsize=(12, 6))
plt.scatter(quats[0], MSE_cuat_disc[0], label='mse q0', color='r',marker='*')
plt.scatter(quats[1], MSE_cuat_disc[1], label='mse q1', color='b',marker='*')
plt.scatter(quats[2], MSE_cuat_disc[2], label='mse q2', color='k',marker='*')
plt.scatter(quats[3], MSE_cuat_disc[3], label='mse q3', color='g',marker='*')
plt.scatter(quats[0], MSE_cuat_nl[0], label='mse q0_nl', color='r',marker='o')
plt.scatter(quats[1], MSE_cuat_nl[1], label='mse q1_nl', color='b',marker='o')
plt.scatter(quats[2], MSE_cuat_nl[2], label='mse q2_nl', color='k',marker='o')
plt.scatter(quats[3], MSE_cuat_nl[3], label='mse q3_nl', color='g',marker='o')
plt.xlabel('Cuaterniones')
plt.ylabel('Mean Square Error [-]')
plt.legend()
plt.title('MSE de cada cuaternion entre lineal y no lineal y lineal discreto')
# plt.xlim(20000,100000)
# plt.ylim(-0.005,0.005)
plt.grid()
plt.show()

vels = np.array([0,1,2])
plt.figure(figsize=(12, 6))
plt.scatter(vels[0], MSE_omega_disc[0], label='mse w0', color='r',marker='*')
plt.scatter(vels[1], MSE_omega_disc[1], label='mse w1', color='b',marker='*')
plt.scatter(vels[2], MSE_omega_disc[2], label='mse w2', color='k',marker='*')
plt.scatter(vels[0], MSE_omega_nl[0], label='mse w0_nl', color='r',marker='o')
plt.scatter(vels[1], MSE_omega_nl[1], label='mse w1_nl', color='b',marker='o')
plt.scatter(vels[2], MSE_omega_nl[2], label='mse w2_nl', color='k',marker='o')
plt.xlabel('Velocidades angulares')
plt.ylabel('Mean Square Error [rad/s]')
plt.legend()
plt.title('MSE de cada velocidad angular entre lineal y no lineal y lineal discreto')
# plt.xlim(20000,100000)
# plt.ylim(-0.005,0.005)
plt.grid()
plt.show()

