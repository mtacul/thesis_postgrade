# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 02:24:57 2024

@author: nachi
"""

#%% Librerias a utilizar
import matplotlib.pyplot as plt
import pandas as pd
import functions
from scipy.signal import welch
import numpy as np

# archivo_c_csv = "control.csv"
# archivo_c_csv = "control5s.csv"
# archivo_c_csv = "control_TRIAD.csv"
# archivo_c_csv = "control_bad_gyros.csv"
# archivo_c_csv = "control_orbit.csv"
# archivo_c_csv = "control_orbits.csv"

    # "control_good_magnet_o_Worbit.csv"

#%%

# Definir la lista de archivos CSV disponibles
archivos_disponibles = [
    "control_bad_o.csv",
    "control_med_o.csv",
    "control_good_magnet_o.csv",
    "control_good_magnetmed_o.csv",
    "control_good_magnetbad_o.csv",
    "control_good_magnetTRIAD_o.csv",
    "control_rw.csv"

]

# Mostrar el menú al usuario
print("Seleccione un archivo CSV para abrir:")
for i, archivo in enumerate(archivos_disponibles, 1):
    print(f"{i}. {archivo}")

# Obtener la elección del usuario
opcion = int(input("Ingrese el número de archivo deseado: "))

# Validar la opción del usuario
if 1 <= opcion <= len(archivos_disponibles):
    archivo_c_csv = archivos_disponibles[opcion - 1]
    
    # Leer el archivo CSV en un DataFrame de pandas
    df_c = pd.read_csv(archivo_c_csv)

    # Convertir el DataFrame a un array de NumPy
    array_datos_c = df_c.to_numpy()
    
    # Ahora puedes usar 'array_datos_c' en tu código
    print(f"Archivo seleccionado: {archivo_c_csv}")
else:
    print("Opción inválida. Por favor, ingrese un número válido.")
    
print("\n")

t_aux = array_datos_c[:,0]
Roll = array_datos_c[:,1]
Pitch =  array_datos_c[:,2]
Yaw =  array_datos_c[:,3]
q0_rot = array_datos_c[:,4]
q1_rot = array_datos_c[:,5]
q2_rot = array_datos_c[:,6]
q3_rot = array_datos_c[:,7]
q0_TRIADS = array_datos_c[:,8]
q1_TRIADS = array_datos_c[:,9]
q2_TRIADS = array_datos_c[:,10]
q3_TRIADS = array_datos_c[:,11]
w0_values = array_datos_c[:,12]
w1_values = array_datos_c[:,13]
w2_values = array_datos_c[:,14]
# pot_act_x = abs(array_datos_c[:,15])
# pot_act_y = abs(array_datos_c[:,16])
# pot_act_z = abs(array_datos_c[:,17])
# pot_magn_x = array_datos_c[:,18]
# pot_magn_y = array_datos_c[:,19]
# pot_magn_z = array_datos_c[:,20]

#Filtro pasa alto para el Jitter y pasa bajo para exactitud de apuntamiento y agilidad
Roll_high_pass = functions.high_pass_filter(Roll, 10, len(t_aux))
Roll_low_pass = functions.low_pass_filter(Roll, 10, len(t_aux))
Pitch_high_pass = functions.high_pass_filter(Pitch, 10, len(t_aux))
Pitch_low_pass = functions.low_pass_filter(Pitch, 10, len(t_aux))
Yaw_high_pass = functions.high_pass_filter(Yaw, 10, len(t_aux))
Yaw_low_pass = functions.low_pass_filter(Yaw, 10, len(t_aux))

#Caso de ruido debido a valores de q3 forzados a 0
if archivo_c_csv == "control_good_magnetmed_o.csv": #or archivo_c_csv == "control_good_magnetbad_o.csv":
    # Definir la sección en el cual ocurre el fallo
    start_index = np.where(t_aux >= 45000)[0][0]
    end_index = np.where(t_aux <= 55000)[0][-1]
    
    Yaw_lowpass = functions.low_pass_filter(Yaw[start_index:end_index+1],5, len(t_aux[start_index:end_index+1]))
    Yaw_low_pass_section = functions.low_pass_filter(Yaw_high_pass[start_index:end_index+1], 5, len(t_aux[start_index:end_index+1]))
    Yaw_new = np.zeros_like(Yaw)
    Yaw_high_pass_new = np.zeros_like(Yaw)
    Yaw_new[int(t_aux[0]):start_index] = Yaw[int(t_aux[0]):start_index]
    Yaw_high_pass_new[int(t_aux[0]):start_index] = Yaw_high_pass[int(t_aux[0]):start_index]
    Yaw_new[start_index:end_index+1]= Yaw_lowpass
    Yaw_high_pass_new[start_index:end_index+1]= Yaw_low_pass_section
    Yaw_new[end_index:int(t_aux[-1])] = Yaw[end_index:int(t_aux[-1])]
    Yaw_high_pass_new[end_index:int(t_aux[-1])]= Yaw_high_pass[end_index:int(t_aux[-1])]
    Yaw = Yaw_new
    Yaw_high_pass = Yaw_high_pass_new

frequencies_R, psd_R = welch(Roll_high_pass, len(t_aux), nperseg=1024)
frequencies_P, psd_P = welch(Pitch_high_pass, len(t_aux), nperseg=1024)
frequencies_Y, psd_Y = welch(Yaw_high_pass, len(t_aux), nperseg=1024)

psd_R_R =[]
psd_P_R =[]
psd_Y_R =[]

for i in range(len(frequencies_R)):
    psd_R_r = np.real(psd_R[i])
    psd_P_r = np.real(psd_P[i])
    psd_Y_r = np.real(psd_Y[i])
    psd_R_R.append(psd_R_r)
    psd_P_R.append(psd_P_r)
    psd_Y_R.append(psd_Y_r)

psd_R_R = np.array(psd_R_R)
psd_P_R = np.array(psd_P_R)
psd_Y_R = np.array(psd_Y_R)

# Definir los anchos de banda deseados
bandwidth_1 = (0, 10000)  # Ancho de banda 1 en Hz

# Calcular la PSD dentro de los anchos de banda específicos
indices_bandwidth_1_R = np.where((frequencies_R >= bandwidth_1[0]) & (frequencies_R <= bandwidth_1[1]))
psd_bandwidth_1_R = np.trapz(psd_R[indices_bandwidth_1_R], frequencies_R[indices_bandwidth_1_R])

indices_bandwidth_1_P = np.where((frequencies_P >= bandwidth_1[0]) & (frequencies_P <= bandwidth_1[1]))
psd_bandwidth_1_P = np.trapz(psd_P[indices_bandwidth_1_P], frequencies_P[indices_bandwidth_1_P])

indices_bandwidth_1_Y = np.where((frequencies_Y >= bandwidth_1[0]) & (frequencies_Y <= bandwidth_1[1]))
psd_bandwidth_1_Y = np.trapz(psd_Y[indices_bandwidth_1_Y], frequencies_Y[indices_bandwidth_1_Y])

psd_RPY = np.array([psd_bandwidth_1_R,psd_bandwidth_1_P,psd_bandwidth_1_Y])
print("las densidades de potencia espectral en Roll, Pitch y Yaw son: \n",psd_RPY)
print("\n")


# Encontrar el tiempo de asentamiento en segundos de cada angulo de Euler
settling_band_R = 1
settling_band_P = 1
settling_band_Y = 1

settling_error_sup_R = np.full(len(t_aux),settling_band_R+ Roll_low_pass[-1])
settling_error_inf_R = np.full(len(t_aux),-settling_band_R+ Roll_low_pass[-1])

settling_error_sup_P = np.full(len(t_aux),settling_band_P+ Pitch_low_pass[-1])
settling_error_inf_P = np.full(len(t_aux),-settling_band_P+ Pitch_low_pass[-1])

settling_error_sup_Y = np.full(len(t_aux),settling_band_Y+ Yaw_low_pass[-1])
settling_error_inf_Y = np.full(len(t_aux),-settling_band_Y+ Yaw_low_pass[-1])


settling_time_indices_R = []
start_index_R = None

for i in range(len(Roll_low_pass)):
    if Roll_low_pass[i] <= settling_error_sup_R[i] and Roll_low_pass[i] >= settling_error_inf_R[i]:
        if start_index_R is None:
            start_index_R = i
    else:
        if start_index_R is not None:
            settling_time_indices_R.append((start_index_R, i - 1))
            start_index_R = None

if start_index_R is not None:
    settling_time_indices_R.append((start_index_R, len(Roll_low_pass) - 1))

if settling_time_indices_R:
    settling_times_R = []
    for start, end in settling_time_indices_R:
        settling_times_R.append((t_aux[start], t_aux[end]))
    print("Tiempos de asentamiento en Roll:")
    for start, end in settling_times_R:
        print("Inicio:", start,"[s]", "Fin:", end,"[s]")
else:
    print("La señal no entra en la banda de asentamiento.")
    
print("\n")
settling_time_indices_P = []
start_index_P = None
    
for i in range(len(Pitch_low_pass)):
    if Pitch_low_pass[i] <= settling_error_sup_P[i] and Pitch_low_pass[i] >= settling_error_inf_P[i]:
        if start_index_P is None:
            start_index_P = i
    else:
        if start_index_P is not None:
            settling_time_indices_P.append((start_index_P, i - 1))
            start_index_P = None

if start_index_P is not None:
    settling_time_indices_P.append((start_index_P, len(Pitch_low_pass) - 1))

if settling_time_indices_P:
    settling_times_P = []
    for start, end in settling_time_indices_P:
        settling_times_P.append((t_aux[start], t_aux[end]))
    print("Tiempos de asentamiento en Pitch:")
    for start, end in settling_times_P:
        print("Inicio:", start,"[s]", "Fin:", end,"[s]")
else:
    print("La señal no entra en la banda de asentamiento.")

print("\n")
settling_time_indices_Y = []
start_index_Y = None

for i in range(len(Yaw_low_pass)):
    if Yaw_low_pass[i] <= settling_error_sup_Y[i] and Yaw_low_pass[i] >= settling_error_inf_Y[i]:
        if start_index_Y is None:
            start_index_Y = i
    else:
        if start_index_Y is not None:
            settling_time_indices_Y.append((start_index_Y, i - 1))
            start_index_Y = None

if start_index_Y is not None:
    settling_time_indices_Y.append((start_index_Y, len(Yaw_low_pass) - 1))

if settling_time_indices_Y:
    settling_times_Y = []
    for start, end in settling_time_indices_Y:
        settling_times_Y.append((t_aux[start], t_aux[end]))
    print("Tiempos de asentamiento en Yaw:")
    for start, end in settling_times_Y:
        print("Inicio:", start,"[s]", "Fin:", end,"[s]")
else:
    print("La señal no entra en la banda de asentamiento.")

print("\n")


#Seccion para calcular la exactitud de apuntamiento en cada angulo de Euler
time_R = np.array(settling_times_R[-1])
data_R = Roll_low_pass[int(time_R[0]/2):int(time_R[1]/2)]
accuracy_R = abs(np.mean(data_R))

time_P = np.array(settling_times_P[-1])
data_P = Pitch_low_pass[int(time_P[0]/2):int(time_P[1]/2)]
accuracy_P = abs(np.mean(data_P))

time_Y = np.array(settling_times_Y[-1])
data_Y = Yaw_low_pass[int(time_Y[0]/2):int(time_Y[1]/2)]
accuracy_Y = abs(np.mean(data_Y))

accuracy_RPY = np.array([accuracy_R,accuracy_P,accuracy_Y])
print("La exactitud de apuntamiento para Roll, Pitch y Yaw son respecticamente: \n", accuracy_RPY, "[°]")


#%%

plt.figure(figsize=(12, 6))
plt.plot(t_aux, Roll, label='Roll')
plt.plot(t_aux, Pitch, label='Pitch')
plt.plot(t_aux, Yaw, label='Yaw')
plt.xlabel('Tiempo [s]')
plt.ylabel('Ángulos de Euler [°]')
plt.legend()
plt.title('Obtencion de los ángulos de Euler')
# plt.xlim(0.8e7,1.7e7)
# plt.ylim(-15,2)
plt.grid()
plt.show()
# plt.set_yli1m(-10, 10)  # Ajusta los límites en el eje Y


# fig0, axes0 = plt.subplots(nrows=1, ncols=2, figsize=(14, 4))

# axes0[0].plot(t_aux, yaw_b, label='Yaw nivel bajo')
# axes0[0].plot(t_aux, yaw_m, label='Yaw nivel medio')
# axes0[0].plot(t_aux, yaw_g, label='Yaw nivel alto')
# axes0[0].set_xlabel('Tiempo [s]')
# axes0[0].set_ylabel('Ángulos de Euler [°]')
# axes0[0].legend()
# axes0[0].grid()
# #axes0[0].set_ylim(-1, 1)  # Ajusta los límites en el eje Y

# axes0[1].plot(t_aux, yaw_b, label='Yaw nivel bajo')
# axes0[1].plot(t_aux, yaw_m, label='Yaw nivel medio')
# axes0[1].plot(t_aux, yaw_g, label='Yaw nivel alto')
# axes0[1].set_xlabel('Tiempo [s]')
# axes0[1].set_ylabel('Ángulos de Euler [°]')
# # axes0[1].legend()
# axes0[1].grid()
# axes0[1].set_ylim(-20, -5)  # Ajusta los límites en el eje Y
# axes0[1].set_xlim(150000, 400000)  # Ajusta los límites en el eje Y

# plt.tight_layout()
# plt.show()


fig0, axes0 = plt.subplots(nrows=1, ncols=3, figsize=(16, 4))

axes0[0].plot(t_aux, q0_rot, label='q0 PD_NL')
axes0[0].plot(t_aux, q1_rot, label='q1 PD_NL')
axes0[0].plot(t_aux, q2_rot, label='q2 PD_NL')
axes0[0].plot(t_aux, q3_rot, label='q3 PD_NL')
axes0[0].set_xlabel('Tiempo [s]')
axes0[0].set_ylabel('cuaternión [-]')
axes0[0].legend()
axes0[0].set_title('cuaterniones controlados llevados a 000')
axes0[0].grid()
#axes0[0].set_ylim(-1, 1)  # Ajusta los límites en el eje Y

axes0[1].plot(t_aux, w0_values, label='w0 PD_NL')
axes0[1].plot(t_aux, w1_values, label='w1 PD_NL')
axes0[1].plot(t_aux, w2_values, label='w2 PD_NL')
axes0[1].set_xlabel('Tiempo [s]')
axes0[1].set_ylabel('velocidad angular [rad/s]')
axes0[1].legend()
axes0[1].set_title('velocidades angulares controlados llevados a 000')
axes0[1].grid()


axes0[2].plot(t_aux, q0_TRIADS, label='q0 TRIAD')
axes0[2].plot(t_aux, q1_TRIADS, label='q1 TRIAD')
axes0[2].plot(t_aux, q2_TRIADS, label='q2 TRIAD')
axes0[2].plot(t_aux, q3_TRIADS, label='q3 TRIAD')
axes0[2].set_xlabel('Tiempo [s]')
axes0[2].set_ylabel('cuaternión TRIAD [-]')
axes0[2].legend()
axes0[2].set_title('cuaterniones TRIAD controlados llevados a 000')
axes0[2].grid()

plt.tight_layout()
plt.show()


# plt.figure(figsize=(12, 6))
# plt.plot(t_aux, Roll_high_pass, label='Roll')
# plt.plot(t_aux, Pitch_high_pass, label='Pitch')
# plt.plot(t_aux, Yaw_high_pass, label='Yaw')
# plt.xlabel('Tiempo[s]')
# plt.ylabel('Ángulos de Euler [°]')
# plt.legend()
# plt.title('Filtros pasa alto')
# # plt.xlim(0.8e7,1.7e7)
# # plt.ylim(-0.002,0.002)
# plt.grid()
# plt.show()

# plt.figure(figsize=(12, 6))
# plt.plot(t_aux, Roll_low_pass, label='Roll')
# plt.plot(t_aux, Pitch_low_pass, label='Pitch')
# plt.plot(t_aux, Yaw_low_pass, label='Yaw')
# plt.xlabel('Tiempo[s]')
# plt.ylabel('Ángulos de Euler [°]')
# plt.legend()
# plt.title('Filtros pasa bajo')
# # plt.xlim(0.8e7,1.7e7)
# # plt.ylim(-0.002,0.002)
# plt.grid()
# plt.show()


# fig0, axes0 = plt.subplots(nrows=2, ncols=1, figsize=(7, 7))

# axes0[0].plot(t_aux, Roll_low_pass, label='Roll')
# axes0[0].set_xlabel('Tiempo [s]')
# axes0[0].set_ylabel('Ángulos de Euler [°]')
# axes0[0].legend()
# axes0[0].grid()
# #axes0[0].set_ylim(-1, 1)  # Ajusta los límites en el eje Y

# axes0[1].plot(t_aux, Roll_low_pass, label='Roll')
# axes0[1].plot(t_aux, settling_error_sup_R , label='SET_sup_roll')
# axes0[1].plot(t_aux, settling_error_inf_R , label='SET_inf_roll')
# axes0[1].set_xlabel('Tiempo [s]')
# axes0[1].set_ylabel('Ángulos de Euler [°]')
# axes0[1].legend()
# # axes0[1].set_title('Filtros pasa bajo')
# axes0[1].grid()
# axes0[1].set_ylim(-25,0)  # Ajusta los límites en el eje Y

# plt.tight_layout()
# plt.show()

# fig0, axes0 = plt.subplots(nrows=2, ncols=1, figsize=(7, 7))

# axes0[0].plot(t_aux, Pitch_low_pass, label='Pitch')
# axes0[0].set_xlabel('Tiempo [s]')
# axes0[0].set_ylabel('Ángulos de Euler [°]')
# axes0[0].legend()
# axes0[0].grid()
# #axes0[0].set_ylim(-1, 1)  # Ajusta los límites en el eje Y

# axes0[1].plot(t_aux, Pitch_low_pass, label='Pitch')
# axes0[1].plot(t_aux, settling_error_sup_P , label='SET_sup_pitch')
# axes0[1].plot(t_aux, settling_error_inf_P , label='SET_inf_pitch')
# axes0[1].set_xlabel('Tiempo [s]')
# axes0[1].set_ylabel('Ángulos de Euler [°]')
# axes0[1].legend()
# # axes0[1].set_title('Filtros pasa bajo')
# axes0[1].grid()
# axes0[1].set_ylim(-10,10)  # Ajusta los límites en el eje Y

# plt.tight_layout()
# plt.show()

# fig0, axes0 = plt.subplots(nrows=2, ncols=1, figsize=(7, 7))

# axes0[0].plot(t_aux, Yaw_low_pass, label='Yaw')
# axes0[0].set_xlabel('Tiempo [s]')
# axes0[0].set_ylabel('Ángulos de Euler [°]')
# axes0[0].legend()
# axes0[0].grid()
# #axes0[0].set_ylim(-1, 1)  # Ajusta los límites en el eje Y

# axes0[1].plot(t_aux, Yaw_low_pass, label='Yaw')
# axes0[1].plot(t_aux, settling_error_sup_Y , label='SET_sup_yaw')
# axes0[1].plot(t_aux, settling_error_inf_Y , label='SET_inf_yaw')
# axes0[1].set_xlabel('Tiempo [s]')
# axes0[1].set_ylabel('Ángulos de Euler [°]')
# axes0[1].legend()
# # axes0[1].set_title('Filtros pasa bajo')
# axes0[1].grid()
# axes0[1].set_ylim(-25,0)  # Ajusta los límites en el eje Y

# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(12, 6))
# plt.plot(t_aux, Roll_low_pass, label='Roll')
# plt.plot(t_aux, settling_error_sup_R , label='SET_sup_roll')
# plt.plot(t_aux, settling_error_inf_R , label='SET_inf_roll')
# plt.xlabel('Tiempo[s]')
# plt.ylabel('Ángulos de Euler [°]')
# plt.legend()
# plt.title('Filtros pasa bajo')
# # plt.ylim(-25,0)
# # plt.xlim(0.8e7,1.7e7)
# # plt.ylim(-0.002,0.002)
# plt.grid()
# plt.show()

# plt.figure(figsize=(12, 6))
# plt.plot(t_aux, Pitch_low_pass, label='Pitch')
# plt.plot(t_aux, settling_error_sup_P , label='SET_sup_pitch')
# plt.plot(t_aux, settling_error_inf_P , label='SET_inf_pitch')
# plt.xlabel('Tiempo[s]')
# plt.ylabel('Ángulos de Euler [°]')
# plt.legend()
# plt.title('Filtros pasa bajo')
# # plt.ylim(-10,10)
# # plt.xlim(0.8e7,1.7e7)
# # plt.ylim(-0.002,0.002)
# plt.grid()
# plt.show()

# plt.figure(figsize=(12, 6))
# plt.plot(t_aux, Yaw_low_pass, label='Yaw')
# plt.plot(t_aux, settling_error_sup_Y , label='SET_sup_yaw')
# plt.plot(t_aux, settling_error_inf_Y , label='SET_inf_yaw')
# plt.xlabel('Tiempo[s]')
# plt.ylabel('Ángulos de Euler [°]')
# plt.legend()
# plt.title('Filtros pasa bajo')
# # plt.ylim(-25,0)
# # plt.xlim(0.8e7,1.7e7)
# # plt.ylim(-0.002,0.002)
# plt.grid()
# plt.show()


# plt.tight_layout()
# plt.show()
# plt.figure(figsize=(12, 6))
# plt.plot(t_aux,  pot_magn_x, label='potencia gastada en x')
# plt.plot(t_aux,  pot_magn_y, label='potencia gastada en y')
# plt.plot(t_aux,  pot_magn_z, label='potencia gastada en z')
# plt.xlabel('Tiempo')
# plt.ylabel('Potencia [W]')
# plt.legend()
# plt.title('Potencia gastada por el sensor en sus tres direcciones')
# # plt.ylim(-2,2)
# # plt.xlim(0.8e7,1.7e7)
# # plt.ylim(-0.002,0.002)
# plt.grid()
# plt.show()

# plt.figure(figsize=(12, 6))
# plt.plot(t_aux,  pot_act_x, label='potencia gastada en x')
# plt.plot(t_aux,  pot_act_y, label='potencia gastada en y')
# plt.plot(t_aux,  pot_act_z, label='potencia gastada en z')
# plt.xlabel('Tiempo')
# plt.ylabel('Potencia [W]')
# plt.legend()
# plt.title('Potencia gastada por el actuador en sus tres direcciones')
# # plt.ylim(-2,2)
# # plt.xlim(0.8e7,1.7e7)
# # plt.ylim(-0.002,0.002)
# plt.grid()
# plt.show()

plt.figure(figsize=(10, 6))
plt.semilogy(frequencies_R, psd_R_R, label='PSD Roll')
plt.fill_between(frequencies_R[indices_bandwidth_1_R], psd_R_R[indices_bandwidth_1_R], alpha=0.3, label='Ancho de banda 1')
plt.title('Densidad Espectral de Potencia con Anchos de Banda Específicos en Roll')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Densidad Espectral de Potencia')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.semilogy(frequencies_P, psd_P_R, label='PSD Pitch')
plt.fill_between(frequencies_P[indices_bandwidth_1_P], psd_P_R[indices_bandwidth_1_P], alpha=0.3, label='Ancho de banda 1')
plt.title('Densidad Espectral de Potencia con Anchos de Banda Específicos en Pitch')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Densidad Espectral de Potencia')
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(10, 6))
plt.semilogy(frequencies_Y, psd_Y_R, label='PSD Yaw')
plt.fill_between(frequencies_Y[indices_bandwidth_1_Y], psd_Y_R[indices_bandwidth_1_Y], alpha=0.3, label='Ancho de banda 1')
plt.title('Densidad Espectral de Potencia con Anchos de Banda Específicos en Yaw')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Densidad Espectral de Potencia')
plt.legend()
plt.grid(True)
plt.show()
