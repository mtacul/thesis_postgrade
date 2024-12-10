# -*- coding: utf-8 -*-
"""
Created on Fri May  3 14:09:37 2024

@author: nachi
"""

import functions_01
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

#%% Cargar datos del .csv obtenido

# archivo_csv = "vectores_5.76_x5_k.csv"
# archivo_csv = "vectores_10.2k.csv"
# archivo_csv = "vectores_5.76k.csv"
# archivo_csv = "vectores_400k_1s.csv"
archivo_csv = "vectores_400k.csv"

# Leer el archivo CSV en un DataFrame de pandas
df = pd.read_csv(archivo_csv)

# Convertir el DataFrame a un array de NumPy
array_datos = df.to_numpy()

t_aux = array_datos[:, 0]
Bx_IGRF = np.hstack((array_datos[0, 16]*1e-9, array_datos[1:, 16]*1e-9))
By_IGRF = np.hstack((array_datos[0, 17]*1e-9, array_datos[1:, 17]*1e-9))
Bz_IGRF = np.hstack((array_datos[0, 18]*1e-9, array_datos[1:, 18]*1e-9))
vsun_x = array_datos[:, 19]
vsun_y = array_datos[:, 20]
vsun_z = array_datos[:, 21]

position = np.transpose(np.vstack((array_datos[:,1], array_datos[:,2], array_datos[:,3])))
velocity = np.transpose(np.vstack((array_datos[:,4], array_datos[:,5], array_datos[:,6])))

Z_orbits = position[:,:] / np.linalg.norm(position[:,:])
X_orbits = np.cross(velocity[:,:],Z_orbits) / np.linalg.norm(np.cross(velocity[:,:],Z_orbits))
Y_orbits = np.cross(Z_orbits,X_orbits)

#%% Obtencion de los vectores en marco de referencia orbit

q0_e2o = []
q1_e2o = []
q2_e2o = []
q3_e2o = []

Bx_orbit = []
By_orbit = []
Bz_orbit = []
vx_sun_orbit = []
vy_sun_orbit = []
vz_sun_orbit = []

for i in range(len(t_aux)):
    print(t_aux[i])
    Rs_ECI_orbit = np.vstack((X_orbits[i,:],Y_orbits[i,:],Z_orbits[i,:]))
    q_ECI_orbit= Rotation.from_matrix(Rs_ECI_orbit).as_quat()
    
    q0_e2o.append(q_ECI_orbit[0])
    q1_e2o.append(q_ECI_orbit[1])
    q2_e2o.append(q_ECI_orbit[2])
    q3_e2o.append(q_ECI_orbit[3])
    
    B_ECI_quat = [Bx_IGRF[i],By_IGRF[i],Bz_IGRF[i],0]
    inv_q_e2o = functions_01.inv_q(q_ECI_orbit)
    B_orbit = functions_01.quat_mult(functions_01.quat_mult(q_ECI_orbit,B_ECI_quat),inv_q_e2o)
    B_orbit_n = np.array([B_orbit[0],B_orbit[1],B_orbit[2]])
    B_orbit = B_orbit_n / np.linalg.norm(B_orbit_n)
    
    vsun_ECI_quat = [vsun_x[i],vsun_y[i],vsun_z[i],0]
    inv_qi_s = functions_01.inv_q(q_ECI_orbit)
    vsun_orbit = functions_01.quat_mult(functions_01.quat_mult(q_ECI_orbit,vsun_ECI_quat),inv_qi_s)
    vsun_orbit_n = np.array([vsun_orbit[0],vsun_orbit[1],vsun_orbit[2]]) 
    vsun_orbit =  vsun_orbit_n / np.linalg.norm(vsun_orbit_n)

    Bx_orbit.append(B_orbit_n[0])
    By_orbit.append(B_orbit_n[1])
    Bz_orbit.append(B_orbit_n[2])
    vx_sun_orbit.append(vsun_orbit_n[0])
    vy_sun_orbit.append(vsun_orbit_n[1])
    vz_sun_orbit.append(vsun_orbit_n[2])

q0_e2o = np.array(q0_e2o)
q1_e2o = np.array(q1_e2o)
q2_e2o = np.array(q2_e2o)
q3_e2o = np.array(q3_e2o)
Bx_orbit =np.array(Bx_orbit)
By_orbit =np.array(By_orbit)
Bz_orbit =np.array(Bz_orbit)
vx_sun_orbit = np.array(vx_sun_orbit)
vy_sun_orbit = np.array(vy_sun_orbit)
vz_sun_orbit = np.array(vz_sun_orbit)

B_IGRF =  np.transpose(np.vstack((Bx_IGRF,By_IGRF,Bz_IGRF)))
B_orbit = np.transpose(np.vstack((Bx_orbit,By_orbit,Bz_orbit)))
b_norm = []
b_norm_eci = []
position_norm = []

for i in range(len(Bx_orbit)):
    norm = np.linalg.norm(B_orbit[i,:])
    norm_eci = np.linalg.norm(B_IGRF[i,:])
    norm_p = np.linalg.norm(position[i,:])
    b_norm.append(norm)
    b_norm_eci.append(norm_eci)
    position_norm.append(norm_p)

#%% Gráficas de los vectores rotados a orbit

fig0, axes0 = plt.subplots(nrows=1, ncols=2, figsize=(16, 4))

axes0[0].plot(t_aux, Bx_IGRF, label='Bx ECI')
axes0[0].plot(t_aux, By_IGRF, label='By ECI')
axes0[0].plot(t_aux, Bz_IGRF, label='Bz ECI')
axes0[0].plot(t_aux,b_norm_eci, label='norma B ECI')
axes0[0].set_xlabel('Tiempo [s]')
axes0[0].set_ylabel('Fuerza magnetica [T]')
axes0[0].legend()
axes0[0].set_title('fuerza magnetica en ECI')
axes0[0].grid()
axes0[0].set_xlim(0, 5500)  # Ajusta los límites en el eje Y

axes0[1].plot(t_aux, position[:,0], label='posicion del satélite en x')
axes0[1].plot(t_aux, position[:,1], label='posicion del satélite en y')
axes0[1].plot(t_aux, position[:,2], label='posicion del satélite en z')
axes0[1].set_xlabel('Tiempo [s]')
axes0[1].set_ylabel('posición [km]')
axes0[1].legend()
axes0[1].set_title('posiciones del satélite en ECI')
axes0[1].grid()
plt.tight_layout()
plt.show()

fig0, axes0 = plt.subplots(nrows=1, ncols=2, figsize=(16, 4))

axes0[0].plot(t_aux, vsun_x, label='vx ECI')
axes0[0].plot(t_aux, vsun_y, label='vy ECI')
axes0[0].plot(t_aux, vsun_z, label='vz ECI')
axes0[0].set_xlabel('Tiempo [s]')
axes0[0].set_ylabel('Vector sol [-]')
axes0[0].legend()
axes0[0].set_title('vector sol en ECI')
axes0[0].grid()
# axes0[0].set_ylim(-1, 1)  # Ajusta los límites en el eje Y

axes0[1].plot(t_aux, position[:,0], label='posicion del satélite en x')
axes0[1].plot(t_aux, position[:,1], label='posicion del satélite en y')
axes0[1].plot(t_aux, position[:,2], label='posicion del satélite en z')
axes0[1].set_xlabel('Tiempo [s]')
axes0[1].set_ylabel('posición [km]')
axes0[1].legend()
axes0[1].set_title('posiciones del satélite en ECI')
axes0[1].grid()
plt.tight_layout()
plt.show()

fig0, axes0 = plt.subplots(nrows=1, ncols=2, figsize=(16, 4))

axes0[0].plot(t_aux, Bx_orbit, label='Bx orbit')
axes0[0].plot(t_aux, By_orbit, label='By orbit')
axes0[0].plot(t_aux, Bz_orbit, label='Bz orbit')
axes0[0].plot(t_aux,b_norm, label='norma B orbit')
axes0[0].set_xlabel('Tiempo [s]')
axes0[0].set_ylabel('Fuerza magnetica [T]')
axes0[0].legend()
axes0[0].set_title('fuerza magnetica en orbit')
axes0[0].grid()
# axes0[0].set_ylim(-1, 1)  # Ajusta los límites en el eje Y
axes0[0].set_xlim(0, 5500)  # Ajusta los límites en el eje Y

axes0[1].plot(t_aux, position[:,0], label='posicion del satélite en x')
axes0[1].plot(t_aux, position[:,1], label='posicion del satélite en y')
axes0[1].plot(t_aux, position[:,2], label='posicion del satélite en z')
axes0[1].plot(t_aux, position_norm, label='norma de la posicion del satélite')
axes0[1].set_xlabel('Tiempo [s]')
axes0[1].set_ylabel('posición [km]')
axes0[1].legend()
axes0[1].set_title('posiciones del satélite en ECI')
axes0[1].grid()
plt.tight_layout()
plt.show()

fig0, axes0 = plt.subplots(nrows=1, ncols=2, figsize=(16, 4))

axes0[0].plot(t_aux, vx_sun_orbit, label='vx orbit')
axes0[0].plot(t_aux, vy_sun_orbit, label='vy orbit')
axes0[0].plot(t_aux, vz_sun_orbit, label='vz orbit')
axes0[0].set_xlabel('Tiempo [s]')
axes0[0].set_ylabel('vector sol [-]')
axes0[0].legend()
axes0[0].set_title('vector sol en orbit')
axes0[0].grid()
# axes0[0].set_ylim(-1, 1)  # Ajusta los límites en el eje Y

axes0[1].plot(t_aux, position[:,0], label='posicion del satélite en x')
axes0[1].plot(t_aux, position[:,1], label='posicion del satélite en y')
axes0[1].plot(t_aux, position[:,2], label='posicion del satélite en z')
axes0[1].plot(t_aux, position_norm, label='norma de la posicion del satélite')
axes0[1].set_xlabel('Tiempo [s]')
axes0[1].set_ylabel('posición [km]')
axes0[1].legend()
axes0[1].set_title('posiciones del satélite en ECI')
axes0[1].grid()
plt.tight_layout()
plt.show()


fig0, axes0 = plt.subplots(nrows=1, ncols=2, figsize=(16, 4))

axes0[0].plot(t_aux,b_norm, label='norma B orbit')
axes0[0].plot(t_aux,b_norm_eci, label='norma B ECI')
axes0[0].set_xlabel('Tiempo [s]')
axes0[0].set_ylabel('Fuerza magnetica [T]')
axes0[0].legend()
axes0[0].set_title('fuerza magnetica en orbit')
axes0[0].grid()
# axes0[0].set_ylim(-1, 1)  # Ajusta los límites en el eje Y

axes0[1].plot(t_aux, position_norm, label='norma de la posicion del satélite')
axes0[1].set_xlabel('Tiempo [s]')
axes0[1].set_ylabel('posición [km]')
axes0[1].legend()
axes0[1].set_title('normas de posiciones del satélite en ECI')
axes0[1].grid()
plt.tight_layout()
plt.show()

#%% Guardar vectores orbit en un .csv

# Nombre del archivo
# archivo_c = "Vectores_orbit_ECI_1s.csv"
archivo_c = "Vectores_orbit_ECI.csv"

# Abrir el archivo en modo escritura
with open(archivo_c, 'w') as f:
    # Escribir los encabezados
    f.write("t, Bx_orbit, By_orbit, Bz_orbit, Bx_IGRF, By_IGRF, Bz_IGRF, vx_sun_orbit,vy_sun_orbit,vz_sun_orbit,vsun_x,vsun_y,vsun_z,q0_e2o,q1_e2o,q2_e2o,q3_e2o \n")

    # Escribir los datos en filas
    for i in range(len(Bx_orbit)):
        f.write("{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {},{}, {}, {},{} \n".format( 
            t_aux[i],Bx_orbit[i], By_orbit[i], Bz_orbit[i],  Bx_IGRF[i],  By_IGRF[i],  Bz_IGRF[i],
            vx_sun_orbit[i], vy_sun_orbit[i],vz_sun_orbit[i],vsun_x[i],vsun_y[i],vsun_z[i],
            q0_e2o[i],q1_e2o[i],q2_e2o[i],q3_e2o[i]))

print("Vectores guardados en el archivo:", archivo_c)


