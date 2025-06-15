"""
Created on Sun Jun 23 22:39:30 2024

@author: nachi
"""

#%% Librerias a utilizar
import matplotlib.pyplot as plt
import pandas as pd
import functions_05
from scipy.signal import welch
import numpy as np
import scipy.stats as stats
import os

#%%

# Definir la lista de archivos CSV disponibles
archivos_disponibles = [
    "_sen1_act1_RW.csv",
    "_sen1_act2_RW.csv",
    "_sen1_act3_RW.csv",
    "_sen2_act1_RW.csv",
    "_sen2_act2_RW.csv",
    "_sen2_act3_RW.csv",
    "_sen3_act1_RW.csv",
    "_sen3_act2_RW.csv",
    "_sen3_act3_RW.csv",
    "_sen1_act1_MT.csv",
    "_sen1_act2_MT.csv",
    "_sen1_act3_MT.csv",
    "_sen2_act1_MT.csv",
    "_sen2_act2_MT.csv",
    "_sen2_act3_MT.csv",
    "_sen3_act1_MT.csv",
    "_sen3_act2_MT.csv",
    "_sen3_act3_MT.csv",
    "_sen1_act1_MT_LQR.csv",
    "_sen1_act2_MT_LQR.csv",
    "_sen1_act3_MT_LQR.csv",
    "_sen2_act1_MT_LQR.csv",
    "_sen2_act2_MT_LQR.csv",
    "_sen2_act3_MT_LQR.csv",
    "_sen3_act1_MT_LQR.csv",
    "_sen3_act2_MT_LQR.csv",
    "_sen3_act3_MT_LQR.csv"
    
    ]

# Mostrar el menú al usuario
print("Seleccione tres archivos CSV para abrir:")
for i, archivo in enumerate(archivos_disponibles, 1):
    print(f"{i}. {archivo}")

# Obtener las elecciones del usuario
opciones = input("Ingrese los números de los archivos deseados, separados por comas: ").split(',')

# Validar las opciones del usuario
opciones = [int(opcion.strip()) for opcion in opciones if opcion.strip().isdigit()]

if all(1 <= opcion <= len(archivos_disponibles) for opcion in opciones) and len(opciones) == 3:
    archivos_seleccionados = [archivos_disponibles[opcion - 1] for opcion in opciones]
    
    # Leer los archivos CSV en DataFrames de pandas
    dataframes = [pd.read_csv(archivo) for archivo in archivos_seleccionados]
    
    # Convertir los DataFrames a arrays de NumPy
    arrays_datos = [df.to_numpy() for df in dataframes]
    
    # Ahora puedes usar 'arrays_datos' en tu código
    print("Archivos seleccionados:")
    for archivo in archivos_seleccionados:
        print(archivo)
else:
    print("Opciones inválidas. Por favor, ingrese tres números válidos.")
    
print("\n")

array_datos_1 = arrays_datos[0]
array_datos_2 = arrays_datos[1]
array_datos_3 = arrays_datos[2]

t_aux = array_datos_1[:,0]
Roll_1 = array_datos_1[:,1]
Pitch_1 =  array_datos_1[:,2]
Yaw_1 =  array_datos_1[:,3]
q0_real_1 = array_datos_1[:,4]
q1_real_1 = array_datos_1[:,5]
q2_real_1 = array_datos_1[:,6]
q3_real_1 = array_datos_1[:,7]
q0_est_1 = array_datos_1[:,8]
q1_est_1 = array_datos_1[:,9]
q2_est_1 = array_datos_1[:,10]
q3_est_1 = array_datos_1[:,11]
w0_est_1 = array_datos_1[:,12]
w1_est_1 = array_datos_1[:,13]
w2_est_1 = array_datos_1[:,14]
w0_real_1 = array_datos_1[:,15]
w1_real_1 = array_datos_1[:,16]
w2_real_1 = array_datos_1[:,17]
Roll_low_pass_1 = array_datos_1[:,18]
Pitch_low_pass_1 =  array_datos_1[:,19]
Yaw_low_pass_1 =  array_datos_1[:,20]
# ux_1 = array_datos_1[:,21]
# uy_1 =  array_datos_1[:,22]
# uz_1 =  array_datos_1[:,23]
RPY_1 = np.transpose(np.vstack((Roll_1,Pitch_1,Yaw_1)))
norms_RPY_1 = []
for j in range(len(RPY_1)):
    norm_RPY_1 = np.linalg.norm(RPY_1[j,:])
    norms_RPY_1.append(norm_RPY_1)
norms_RPY_1 = np.array(norms_RPY_1)
RPY_low_pass_1 = np.transpose(np.vstack((Roll_low_pass_1,Pitch_low_pass_1,Yaw_low_pass_1)))
norms_RPY_low_pass_1 = []
for j in range(len(RPY_low_pass_1)):
    norm_RPY_low_pass_1 = np.linalg.norm(RPY_low_pass_1[j,:])
    norms_RPY_low_pass_1.append(norm_RPY_low_pass_1)
norms_RPY_low_pass_1 = np.array(norms_RPY_low_pass_1)


Roll_2 = array_datos_2[:,1]
Pitch_2 =  array_datos_2[:,2]
Yaw_2 =  array_datos_2[:,3]
q0_real_2 = array_datos_2[:,4]
q1_real_2 = array_datos_2[:,5]
q2_real_2 = array_datos_2[:,6]
q3_real_2 = array_datos_2[:,7]
q0_est_2 = array_datos_2[:,8]
q1_est_2 = array_datos_2[:,9]
q2_est_2 = array_datos_2[:,10]
q3_est_2 = array_datos_2[:,11]
w0_est_2 = array_datos_2[:,12]
w1_est_2 = array_datos_2[:,13]
w2_est_2 = array_datos_2[:,14]
w0_real_2 = array_datos_2[:,15]
w1_real_2 = array_datos_2[:,16]
w2_real_2 = array_datos_2[:,17]
Roll_low_pass_2 = array_datos_2[:,18]
Pitch_low_pass_2 =  array_datos_2[:,19]
Yaw_low_pass_2 =  array_datos_2[:,20]
# ux_2 = array_datos_2[:,21]
# uy_2 =  array_datos_2[:,22]
# uz_2 =  array_datos_2[:,23]
RPY_2 = np.transpose(np.vstack((Roll_2,Pitch_2,Yaw_2)))
norms_RPY_2 = []
for j in range(len(RPY_2)):
    norm_RPY_2 = np.linalg.norm(RPY_2[j,:])
    norms_RPY_2.append(norm_RPY_2)
norms_RPY_2 = np.array(norms_RPY_2)
RPY_low_pass_2 = np.transpose(np.vstack((Roll_low_pass_2,Pitch_low_pass_2,Yaw_low_pass_2)))
norms_RPY_low_pass_2 = []
for j in range(len(RPY_low_pass_2)):
    norm_RPY_low_pass_2 = np.linalg.norm(RPY_low_pass_2[j,:])
    norms_RPY_low_pass_2.append(norm_RPY_low_pass_2)
norms_RPY_low_pass_2 = np.array(norms_RPY_low_pass_2)

Roll_3 = array_datos_3[:,1]
Pitch_3 =  array_datos_3[:,2]
Yaw_3 =  array_datos_3[:,3]
q0_real_3 = array_datos_3[:,4]
q1_real_3 = array_datos_3[:,5]
q2_real_3 = array_datos_3[:,6]
q3_real_3 = array_datos_3[:,7]
q0_est_3 = array_datos_3[:,8]
q1_est_3 = array_datos_3[:,9]
q2_est_3 = array_datos_3[:,10]
q3_est_3 = array_datos_3[:,11]
w0_est_3 = array_datos_3[:,12]
w1_est_3 = array_datos_3[:,13]
w2_est_3 = array_datos_3[:,14]
w0_real_3 = array_datos_3[:,15]
w1_real_3 = array_datos_3[:,16]
w2_real_3 = array_datos_3[:,17]
Roll_low_pass_3 = array_datos_3[:,18]
Pitch_low_pass_3 =  array_datos_3[:,19]
Yaw_low_pass_3 =  array_datos_3[:,20]
# ux_3 = array_datos_3[:,21]
# uy_3 =  array_datos_3[:,22]
# uz_3 =  array_datos_3[:,23]
RPY_3 = np.transpose(np.vstack((Roll_3,Pitch_3,Yaw_3)))
norms_RPY_3 = []
for j in range(len(RPY_3)):
    norm_RPY_3 = np.linalg.norm(RPY_3[j,:])
    norms_RPY_3.append(norm_RPY_3)
norms_RPY_3 = np.array(norms_RPY_3)
RPY_low_pass_3 = np.transpose(np.vstack((Roll_low_pass_3,Pitch_low_pass_3,Yaw_low_pass_3)))
norms_RPY_low_pass_3 = []
for j in range(len(RPY_low_pass_3)):
    norm_RPY_low_pass_3 = np.linalg.norm(RPY_low_pass_3[j,:])
    norms_RPY_low_pass_3.append(norm_RPY_low_pass_3)
norms_RPY_low_pass_3 = np.array(norms_RPY_low_pass_3)

#%% Densidad espectro potencia para el jitter

#Filtro pasa alto para el Jitter y pasa bajo para exactitud de apuntamiento y agilidad
Roll_high_pass_1 = functions_05.high_pass_filter(Roll_1, 10, len(t_aux))
Pitch_high_pass_1 = functions_05.high_pass_filter(Pitch_1, 10, len(t_aux))
Yaw_high_pass_1 = functions_05.high_pass_filter(Yaw_1, 10, len(t_aux))

frequencies_R_1, psd_R_1 = welch(Roll_high_pass_1, len(t_aux), nperseg=1024)
frequencies_P_1, psd_P_1 = welch(Pitch_high_pass_1, len(t_aux), nperseg=1024)
frequencies_Y_1, psd_Y_1 = welch(Yaw_high_pass_1, len(t_aux), nperseg=1024)

psd_R_R_1 =[]
psd_P_R_1 =[]
psd_Y_R_1 =[]

for i in range(len(frequencies_R_1)):
    psd_R_r_1 = np.real(psd_R_1[i])
    psd_P_r_1 = np.real(psd_P_1[i])
    psd_Y_r_1 = np.real(psd_Y_1[i])
    psd_R_R_1.append(psd_R_r_1)
    psd_P_R_1.append(psd_P_r_1)
    psd_Y_R_1.append(psd_Y_r_1)

psd_R_R_1 = np.array(psd_R_R_1)
psd_P_R_1 = np.array(psd_P_R_1)
psd_Y_R_1 = np.array(psd_Y_R_1)

# Definir los anchos de banda deseados
bandwidth_1 = (0, 10000)  # Ancho de banda 1 en Hz

# Calcular la PSD dentro de los anchos de banda específicos
indices_bandwidth_1_R = np.where((frequencies_R_1 >= bandwidth_1[0]) & (frequencies_R_1 <= bandwidth_1[1]))
psd_bandwidth_1_R = np.trapz(psd_R_1[indices_bandwidth_1_R], frequencies_R_1[indices_bandwidth_1_R])

indices_bandwidth_1_P = np.where((frequencies_P_1 >= bandwidth_1[0]) & (frequencies_P_1 <= bandwidth_1[1]))
psd_bandwidth_1_P = np.trapz(psd_P_1[indices_bandwidth_1_P], frequencies_P_1[indices_bandwidth_1_P])

indices_bandwidth_1_Y = np.where((frequencies_Y_1 >= bandwidth_1[0]) & (frequencies_Y_1 <= bandwidth_1[1]))
psd_bandwidth_1_Y = np.trapz(psd_Y_1[indices_bandwidth_1_Y], frequencies_Y_1[indices_bandwidth_1_Y])

psd_RPY_1 = np.array([psd_bandwidth_1_R,psd_bandwidth_1_P,psd_bandwidth_1_Y])


#Filtro pasa alto para el Jitter y pasa bajo para exactitud de apuntamiento y agilidad
Roll_high_pass_2 = functions_05.high_pass_filter(Roll_2, 10, len(t_aux))
Pitch_high_pass_2 = functions_05.high_pass_filter(Pitch_2, 10, len(t_aux))
Yaw_high_pass_2 = functions_05.high_pass_filter(Yaw_2, 10, len(t_aux))

frequencies_R_2, psd_R_2 = welch(Roll_high_pass_2, len(t_aux), nperseg=1024)
frequencies_P_2, psd_P_2 = welch(Pitch_high_pass_2, len(t_aux), nperseg=1024)
frequencies_Y_2, psd_Y_2 = welch(Yaw_high_pass_2, len(t_aux), nperseg=1024)

psd_R_R_2 =[]
psd_P_R_2 =[]
psd_Y_R_2 =[]

for i in range(len(frequencies_R_2)):
    psd_R_r_2 = np.real(psd_R_2[i])
    psd_P_r_2 = np.real(psd_P_2[i])
    psd_Y_r_2 = np.real(psd_Y_2[i])
    psd_R_R_2.append(psd_R_r_2)
    psd_P_R_2.append(psd_P_r_2)
    psd_Y_R_2.append(psd_Y_r_2)

psd_R_R_2 = np.array(psd_R_R_2)
psd_P_R_2 = np.array(psd_P_R_2)
psd_Y_R_2 = np.array(psd_Y_R_2)

# Definir los anchos de banda deseados
bandwidth_2 = (0, 10000)  # Ancho de banda 1 en Hz

# Calcular la PSD dentro de los anchos de banda específicos
indices_bandwidth_2_R = np.where((frequencies_R_2 >= bandwidth_2[0]) & (frequencies_R_2 <= bandwidth_2[1]))
psd_bandwidth_2_R = np.trapz(psd_R_2[indices_bandwidth_2_R], frequencies_R_2[indices_bandwidth_2_R])

indices_bandwidth_2_P = np.where((frequencies_P_2 >= bandwidth_2[0]) & (frequencies_P_2 <= bandwidth_2[1]))
psd_bandwidth_2_P = np.trapz(psd_P_2[indices_bandwidth_2_P], frequencies_P_2[indices_bandwidth_2_P])

indices_bandwidth_2_Y = np.where((frequencies_Y_2 >= bandwidth_2[0]) & (frequencies_Y_2 <= bandwidth_2[1]))
psd_bandwidth_2_Y = np.trapz(psd_Y_2[indices_bandwidth_2_Y], frequencies_Y_2[indices_bandwidth_2_Y])

psd_RPY_2 = np.array([psd_bandwidth_2_R,psd_bandwidth_2_P,psd_bandwidth_2_Y])


#Filtro pasa alto para el Jitter y pasa bajo para exactitud de apuntamiento y agilidad
Roll_high_pass_3 = functions_05.high_pass_filter(Roll_3, 10, len(t_aux))
Pitch_high_pass_3 = functions_05.high_pass_filter(Pitch_3, 10, len(t_aux))
Yaw_high_pass_3 = functions_05.high_pass_filter(Yaw_3, 10, len(t_aux))

frequencies_R_3, psd_R_3 = welch(Roll_high_pass_3, len(t_aux), nperseg=1024)
frequencies_P_3, psd_P_3 = welch(Pitch_high_pass_3, len(t_aux), nperseg=1024)
frequencies_Y_3, psd_Y_3 = welch(Yaw_high_pass_3, len(t_aux), nperseg=1024)

psd_R_R_3 =[]
psd_P_R_3 =[]
psd_Y_R_3 =[]

for i in range(len(frequencies_R_3)):
    psd_R_r_3 = np.real(psd_R_3[i])
    psd_P_r_3 = np.real(psd_P_3[i])
    psd_Y_r_3 = np.real(psd_Y_3[i])
    psd_R_R_3.append(psd_R_r_3)
    psd_P_R_3.append(psd_P_r_3)
    psd_Y_R_3.append(psd_Y_r_3)

psd_R_R_3 = np.array(psd_R_R_3)
psd_P_R_3 = np.array(psd_P_R_3)
psd_Y_R_3 = np.array(psd_Y_R_3)

# Definir los anchos de banda deseados
bandwidth_3 = (0, 10000)  # Ancho de banda 1 en Hz

# Calcular la PSD dentro de los anchos de banda específicos
indices_bandwidth_3_R = np.where((frequencies_R_3 >= bandwidth_3[0]) & (frequencies_R_3 <= bandwidth_3[1]))
psd_bandwidth_3_R = np.trapz(psd_R_3[indices_bandwidth_3_R], frequencies_R_3[indices_bandwidth_3_R])

indices_bandwidth_3_P = np.where((frequencies_P_3 >= bandwidth_3[0]) & (frequencies_P_3 <= bandwidth_3[1]))
psd_bandwidth_3_P = np.trapz(psd_P_3[indices_bandwidth_3_P], frequencies_P_3[indices_bandwidth_3_P])

indices_bandwidth_3_Y = np.where((frequencies_Y_3 >= bandwidth_3[0]) & (frequencies_Y_3 <= bandwidth_3[1]))
psd_bandwidth_3_Y = np.trapz(psd_Y_3[indices_bandwidth_3_Y], frequencies_Y_3[indices_bandwidth_3_Y])

psd_RPY_3 = np.array([psd_bandwidth_3_R,psd_bandwidth_3_P,psd_bandwidth_3_Y])

#%% Encontrar el tiempo de asentamiento en segundos de cada angulo de Euler

settling_band_R = 2.5
settling_band_P = 2.5
settling_band_Y = 2.5

settling_error_sup_R = np.full(len(t_aux),settling_band_R)
settling_error_inf_R = np.full(len(t_aux),-settling_band_R)

settling_error_sup_P = np.full(len(t_aux),settling_band_P)
settling_error_inf_P = np.full(len(t_aux),-settling_band_P)

settling_error_sup_Y = np.full(len(t_aux),settling_band_Y)
settling_error_inf_Y = np.full(len(t_aux),-settling_band_Y)

#%% Asentamientos para la opcion 1

settling_time_indices_R_1 = []
start_index_R_1 = None
settling_time_indices_P_1 = []
start_index_P_1 = None
settling_time_indices_Y_1 = []
start_index_Y_1 = None

for i in range(len(Roll_low_pass_1)):
    if Roll_low_pass_1[i] <= settling_error_sup_R[i] and Roll_low_pass_1[i] >= settling_error_inf_R[i]:
        if start_index_R_1 is None:
            start_index_R_1 = i
    else:
        if start_index_R_1 is not None:
            settling_time_indices_R_1.append((start_index_R_1, i - 1))
            start_index_R_1 = None

if start_index_R_1 is not None:
    settling_time_indices_R_1.append((start_index_R_1, len(Roll_low_pass_1) - 1))

if settling_time_indices_R_1:
    settling_times_R_1 = []
    for start, end in settling_time_indices_R_1:
        settling_times_R_1.append((t_aux[start], t_aux[end]))
else:
    print("La señal no entra en la banda de asentamiento.")
    

for i in range(len(Pitch_low_pass_1)):
    if Pitch_low_pass_1[i] <= settling_error_sup_P[i] and Pitch_low_pass_1[i] >= settling_error_inf_P[i]:
        if start_index_P_1 is None:
            start_index_P_1 = i
    else:
        if start_index_P_1 is not None:
            settling_time_indices_P_1.append((start_index_P_1, i - 1))
            start_index_P_1 = None

if start_index_P_1 is not None:
    settling_time_indices_P_1.append((start_index_P_1, len(Pitch_low_pass_1) - 1))

if settling_time_indices_P_1:
    settling_times_P_1 = []
    for start, end in settling_time_indices_P_1:
        settling_times_P_1.append((t_aux[start], t_aux[end]))
else:
    print("La señal no entra en la banda de asentamiento.")


for i in range(len(Yaw_low_pass_1)):
    if Yaw_low_pass_1[i] <= settling_error_sup_Y[i] and Yaw_low_pass_1[i] >= settling_error_inf_Y[i]:
        if start_index_Y_1 is None:
            start_index_Y_1 = i
    else:
        if start_index_Y_1 is not None:
            settling_time_indices_Y_1.append((start_index_Y_1, i - 1))
            start_index_Y_1 = None

if start_index_Y_1 is not None:
    settling_time_indices_Y_1.append((start_index_Y_1, len(Yaw_low_pass_1) - 1))

if settling_time_indices_Y_1:
    settling_times_Y_1 = []
    for start, end in settling_time_indices_Y_1:
        settling_times_Y_1.append((t_aux[start], t_aux[end]))
else:
    print("La señal no entra en la banda de asentamiento.")


settling_time_indices_norm_1 = []
start_index_norm_1 = None
upper_limit = settling_band_R
lower_limit = -settling_band_R

for i in range(len(norms_RPY_low_pass_1)):
    if lower_limit <= norms_RPY_low_pass_1[i] <= upper_limit:
        if start_index_norm_1 is None:
            start_index_norm_1 = i
    else:
        if start_index_norm_1 is not None:
            settling_time_indices_norm_1.append((start_index_norm_1, i - 1))
            start_index_norm_1 = None

if start_index_norm_1 is not None:
    settling_time_indices_norm_1.append((start_index_norm_1, len(norms_RPY_low_pass_1) - 1))

if settling_time_indices_norm_1:
    settling_times_norm_1 = []
    for start, end in settling_time_indices_norm_1:
        settling_times_norm_1.append((t_aux[start], t_aux[end]))
else:
    print("La señal no entra en la banda de asentamiento.")

#%% Asentamientos para la opcion 2

settling_time_indices_R_2 = []
start_index_R_2 = None
settling_time_indices_P_2 = []
start_index_P_2 = None
settling_time_indices_Y_2 = []
start_index_Y_2 = None


for i in range(len(Roll_low_pass_2)):
    if Roll_low_pass_2[i] <= settling_error_sup_R[i] and Roll_low_pass_2[i] >= settling_error_inf_R[i]:
        if start_index_R_2 is None:
            start_index_R_2 = i
    else:
        if start_index_R_2 is not None:
            settling_time_indices_R_2.append((start_index_R_2, i - 1))
            start_index_R_2 = None

if start_index_R_2 is not None:
    settling_time_indices_R_2.append((start_index_R_2, len(Roll_low_pass_2) - 1))

if settling_time_indices_R_2:
    settling_times_R_2 = []
    for start, end in settling_time_indices_R_2:
        settling_times_R_2.append((t_aux[start], t_aux[end]))
else:
    print("La señal no entra en la banda de asentamiento.")
        
    
for i in range(len(Pitch_low_pass_2)):
    if Pitch_low_pass_2[i] <= settling_error_sup_P[i] and Pitch_low_pass_2[i] >= settling_error_inf_P[i]:
        if start_index_P_2 is None:
            start_index_P_2 = i
    else:
        if start_index_P_2 is not None:
            settling_time_indices_P_2.append((start_index_P_2, i - 1))
            start_index_P_2 = None

if start_index_P_2 is not None:
    settling_time_indices_P_2.append((start_index_P_2, len(Pitch_low_pass_2) - 1))

if settling_time_indices_P_2:
    settling_times_P_2 = []
    for start, end in settling_time_indices_P_2:
        settling_times_P_2.append((t_aux[start], t_aux[end]))
else:
    print("La señal no entra en la banda de asentamiento.")


for i in range(len(Yaw_low_pass_2)):
    if Yaw_low_pass_2[i] <= settling_error_sup_Y[i] and Yaw_low_pass_2[i] >= settling_error_inf_Y[i]:
        if start_index_Y_2 is None:
            start_index_Y_2 = i
    else:
        if start_index_Y_2 is not None:
            settling_time_indices_Y_2.append((start_index_Y_2, i - 1))
            start_index_Y_2 = None

if start_index_Y_2 is not None:
    settling_time_indices_Y_2.append((start_index_Y_2, len(Yaw_low_pass_2) - 1))

if settling_time_indices_Y_2:
    settling_times_Y_2 = []
    for start, end in settling_time_indices_Y_2:
        settling_times_Y_2.append((t_aux[start], t_aux[end]))
else:
    print("La señal no entra en la banda de asentamiento.")


settling_time_indices_norm_2 = []
start_index_norm_2 = None
upper_limit = settling_band_R
lower_limit = -settling_band_R

for i in range(len(norms_RPY_low_pass_2)):
    if lower_limit <= norms_RPY_low_pass_2[i] <= upper_limit:
        if start_index_norm_2 is None:
            start_index_norm_2 = i
    else:
        if start_index_norm_2 is not None:
            settling_time_indices_norm_2.append((start_index_norm_2, i - 1))
            start_index_norm_2 = None

if start_index_norm_2 is not None:
    settling_time_indices_norm_2.append((start_index_norm_2, len(norms_RPY_low_pass_2) - 1))

if settling_time_indices_norm_2:
    settling_times_norm_2 = []
    for start, end in settling_time_indices_norm_2:
        settling_times_norm_2.append((t_aux[start], t_aux[end]))
else:
    print("La señal no entra en la banda de asentamiento.")
#%% Asentamiento para la opcion 3

settling_time_indices_R_3 = []
start_index_R_3= None
settling_time_indices_P_3 = []
start_index_P_3 = None
settling_time_indices_Y_3 = []
start_index_Y_3 = None


for i in range(len(Roll_low_pass_3)):
    if Roll_low_pass_3[i] <= settling_error_sup_R[i] and Roll_low_pass_3[i] >= settling_error_inf_R[i]:
        if start_index_R_3 is None:
            start_index_R_3 = i
    else:
        if start_index_R_3 is not None:
            settling_time_indices_R_3.append((start_index_R_3, i - 1))
            start_index_R_3 = None

if start_index_R_3 is not None:
    settling_time_indices_R_3.append((start_index_R_3, len(Roll_low_pass_3) - 1))

if settling_time_indices_R_3:
    settling_times_R_3 = []
    for start, end in settling_time_indices_R_3:
        settling_times_R_3.append((t_aux[start], t_aux[end]))
else:
    print("La señal no entra en la banda de asentamiento.")
    


for i in range(len(Pitch_low_pass_3)):
    if Pitch_low_pass_3[i] <= settling_error_sup_P[i] and Pitch_low_pass_3[i] >= settling_error_inf_P[i]:
        if start_index_P_3 is None:
            start_index_P_3 = i
    else:
        if start_index_P_3 is not None:
            settling_time_indices_P_3.append((start_index_P_3, i - 1))
            start_index_P_3 = None

if start_index_P_3 is not None:
    settling_time_indices_P_3.append((start_index_P_3, len(Pitch_low_pass_3) - 1))

if settling_time_indices_P_3:
    settling_times_P_3 = []
    for start, end in settling_time_indices_P_3:
        settling_times_P_3.append((t_aux[start], t_aux[end]))
else:
    print("La señal no entra en la banda de asentamiento.")



for i in range(len(Yaw_low_pass_3)):
    if Yaw_low_pass_3[i] <= settling_error_sup_Y[i] and Yaw_low_pass_3[i] >= settling_error_inf_Y[i]:
        if start_index_Y_3 is None:
            start_index_Y_3 = i
    else:
        if start_index_Y_3 is not None:
            settling_time_indices_Y_3.append((start_index_Y_3, i - 1))
            start_index_Y_3 = None

if start_index_Y_3 is not None:
    settling_time_indices_Y_3.append((start_index_Y_3, len(Yaw_low_pass_3) - 1))

if settling_time_indices_Y_3:
    settling_times_Y_3 = []
    for start, end in settling_time_indices_Y_3:
        settling_times_Y_3.append((t_aux[start], t_aux[end]))
else:
    print("La señal no entra en la banda de asentamiento.")


settling_time_indices_norm_3 = []
start_index_norm_3 = None
upper_limit = settling_band_R
lower_limit = -settling_band_R

for i in range(len(norms_RPY_low_pass_3)):
    if lower_limit <= norms_RPY_low_pass_3[i] <= upper_limit:
        if start_index_norm_3 is None:
            start_index_norm_3 = i
    else:
        if start_index_norm_3 is not None:
            settling_time_indices_norm_3.append((start_index_norm_3, i - 1))
            start_index_norm_3 = None

if start_index_norm_3 is not None:
    settling_time_indices_norm_3.append((start_index_norm_3, len(norms_RPY_low_pass_3) - 1))

if settling_time_indices_norm_3:
    settling_times_norm_3 = []
    for start, end in settling_time_indices_norm_3:
        settling_times_norm_3.append((t_aux[start], t_aux[end]))
else:
    print("La señal no entra en la banda de asentamiento.")
#%% Exactitud de apuntamiento

time_R_1 = np.array(settling_times_R_1[-1])
data_R_1 = Roll_1[int(time_R_1[0]/2):int(time_R_1[1]/2)]
# Calcular media y desviación estándar
media_R_1 = np.mean(data_R_1)
sigma_R_1 = np.std(data_R_1)
# Calcular los límites de 3 sigma
lim_inf_R_1 = media_R_1 - 3 * sigma_R_1
lim_sup_R_1 = media_R_1 + 3 * sigma_R_1
accuracy_R_1 = 3*sigma_R_1

time_P_1 = np.array(settling_times_P_1[-1])
data_P_1 = Pitch_1[int(time_P_1[0]/2):int(time_P_1[1]/2)]
# Calcular media y desviación estándar
media_P_1 = np.mean(data_P_1)
sigma_P_1 = np.std(data_P_1)
# Calcular los límites de 3 sigma
lim_inf_P_1 = media_P_1 - 3 * sigma_P_1
lim_sup_P_1 = media_P_1 + 3 * sigma_P_1
accuracy_P_1 = 3*sigma_P_1

time_Y_1 = np.array(settling_times_Y_1[-1])
data_Y_1= Yaw_1[int(time_Y_1[0]/2):int(time_Y_1[1]/2)]
# Calcular media y desviación estándar
media_Y_1 = np.mean(data_Y_1)
sigma_Y_1 = np.std(data_Y_1)
# Calcular los límites de 3 sigma
lim_inf_Y_1 = media_Y_1 - 3 * sigma_Y_1
lim_sup_Y_1 = media_Y_1 + 3 * sigma_Y_1
accuracy_Y_1 = 3*sigma_Y_1

accuracy_RPY_1 = np.array([accuracy_R_1,accuracy_P_1,accuracy_Y_1])
# print("La exactitud de apuntamiento para Roll, Pitch y Yaw son respecticamente: \n", accuracy_RPY_1, "[°]")


time_R_2 = np.array(settling_times_R_2[-1])
data_R_2 = Roll_2[int(time_R_2[0]/2):int(time_R_2[1]/2)]
# Calcular media y desviación estándar
media_R_2 = np.mean(data_R_2)
sigma_R_2 = np.std(data_R_2)
# Calcular los límites de 3 sigma
lim_inf_R_2 = media_R_2 - 3 * sigma_R_2
lim_sup_R_2 = media_R_2 + 3 * sigma_R_2
accuracy_R_2 = 3*sigma_R_2

time_P_2 = np.array(settling_times_P_2[-1])
data_P_2 = Pitch_2[int(time_P_2[0]/2):int(time_P_2[1]/2)]
# Calcular media y desviación estándar
media_P_2 = np.mean(data_P_2)
sigma_P_2 = np.std(data_P_2)
# Calcular los límites de 3 sigma
lim_inf_P_2 = media_P_2 - 3 * sigma_P_2
lim_sup_P_2 = media_P_2 + 3 * sigma_P_2
accuracy_P_2 = 3*sigma_P_2

time_Y_2 = np.array(settling_times_Y_2[-1])
data_Y_2= Yaw_2[int(time_Y_2[0]/2):int(time_Y_2[1]/2)]
# Calcular media y desviación estándar
media_Y_2 = np.mean(data_Y_2)
sigma_Y_2 = np.std(data_Y_2)
# Calcular los límites de 3 sigma
lim_inf_Y_2 = media_Y_2 - 3 * sigma_Y_2
lim_sup_Y_2 = media_Y_2 + 3 * sigma_Y_2
accuracy_Y_2 = 3*sigma_Y_2

accuracy_RPY_2 = np.array([accuracy_R_2,accuracy_P_2,accuracy_Y_2])
# print("La exactitud de apuntamiento para Roll, Pitch y Yaw son respecticamente: \n", accuracy_RPY_2, "[°]")


time_R_3 = np.array(settling_times_R_3[-1])
data_R_3 = Roll_3[int(time_R_3[0]/2):int(time_R_3[1]/2)]
# Calcular media y desviación estándar
media_R_3 = np.mean(data_R_3)
sigma_R_3 = np.std(data_R_3)
# Calcular los límites de 3 sigma
lim_inf_R_3 = media_R_3 - 3 * sigma_R_3
lim_sup_R_3 = media_R_3 + 3 * sigma_R_3
accuracy_R_3 = 3*sigma_R_3

time_P_3 = np.array(settling_times_P_3[-1])
data_P_3 = Pitch_3[int(time_P_3[0]/2):int(time_P_3[1]/2)]
# Calcular media y desviación estándar
media_P_3 = np.mean(data_P_3)
sigma_P_3 = np.std(data_P_3)
# Calcular los límites de 3 sigma
lim_inf_P_3 = media_P_3 - 3 * sigma_P_3
lim_sup_P_3 = media_P_3 + 3 * sigma_P_3
accuracy_P_3 = 3*sigma_P_3

time_Y_3 = np.array(settling_times_Y_3[-1])
data_Y_3= Yaw_3[int(time_Y_3[0]/2):int(time_Y_3[1]/2)]
# Calcular media y desviación estándar
media_Y_3 = np.mean(data_Y_3)
sigma_Y_3 = np.std(data_Y_3)
# Calcular los límites de 3 sigma
lim_inf_Y_3 = media_Y_3 - 3 * sigma_Y_3
lim_sup_Y_3 = media_Y_3 + 3 * sigma_Y_3
accuracy_Y_3 = 3*sigma_Y_3

accuracy_RPY_3 = np.array([accuracy_R_3,accuracy_P_3,accuracy_Y_3])
# print("La exactitud de apuntamiento para Roll, Pitch y Yaw son respecticamente: \n", accuracy_RPY_3, "[°]")

# normas exactitud de apuntamiento
ti_1 = np.array([time_R_1[0],time_P_1[0],time_Y_1[0]])
ti_2 = np.array([time_R_2[0],time_P_2[0],time_Y_2[0]])
ti_3 = np.array([time_R_3[0],time_P_3[0],time_Y_3[0]])

time_norm_1 = np.array([np.linalg.norm(ti_1),t_aux[-1]])
time_norm_2 = np.array([np.linalg.norm(ti_2),t_aux[-1]])
time_norm_3 = np.array([np.linalg.norm(ti_3),t_aux[-1]])

data_norm_1= norms_RPY_1[int(time_norm_1[0]/2):int(time_norm_1[1]/2)]
data_norm_2= norms_RPY_2[int(time_norm_2[0]/2):int(time_norm_2[1]/2)]
data_norm_3= norms_RPY_3[int(time_norm_3[0]/2):int(time_norm_3[1]/2)]

# Calcular media y desviación estándar
media_norm_1 = np.mean(data_norm_1)
sigma_norm_1 = np.std(data_norm_1)
media_norm_2 = np.mean(data_norm_2)
sigma_norm_2 = np.std(data_norm_2)
media_norm_3 = np.mean(data_norm_3)
sigma_norm_3 = np.std(data_norm_3)

accuracy_norm_1 = 3*sigma_norm_1
accuracy_norm_2 = 3*sigma_norm_2
accuracy_norm_3 = 3*sigma_norm_3

accuracy_norms = np.array([accuracy_norm_1,accuracy_norm_2,accuracy_norm_3])
#%%

# normas de densidad espectro potencia
norm_psd_RPY_1 = np.linalg.norm(psd_RPY_1)
norm_psd_RPY_2 = np.linalg.norm(psd_RPY_2)
norm_psd_RPY_3 = np.linalg.norm(psd_RPY_3)

# normas de tiempo de asentamiento
norm_settling_time_1 = np.linalg.norm(np.array([time_R_1[0],time_P_1[0],time_Y_1[0]]))
norm_settling_time_2 = np.linalg.norm(np.array([time_R_2[0],time_P_2[0],time_Y_2[0]]))
norm_settling_time_3 = np.linalg.norm(np.array([time_R_3[0],time_P_3[0],time_Y_3[0]]))

# normas de exactitud de apuntamiento
norm_accuracy_1 = np.linalg.norm(accuracy_RPY_1)
norm_accuracy_2 = np.linalg.norm(accuracy_RPY_2)
norm_accuracy_3 = np.linalg.norm(accuracy_RPY_3)

# Crear el DataFrame
resumen_1 = {
    "Opción": archivos_seleccionados,
    "PSD Roll [W/Hz]": [psd_RPY_1[0], psd_RPY_2[0], psd_RPY_3[0]],
    "PSD Pitch [W/Hz]": [psd_RPY_1[1], psd_RPY_2[1], psd_RPY_3][1],
    "PSD Yaw [W/Hz]": [psd_RPY_1[2], psd_RPY_2[2], psd_RPY_3[2]],

    "Agilidad Roll [s]": [time_R_1[0], time_R_2[0], time_R_3[0]],
    "Agilidad Pitch [s]": [time_P_1[0], time_P_2[0], time_P_3[0]],
    "Agilidad Yaw[s]": [time_Y_1[0], time_Y_2[0], time_Y_3[0]],

    "Exactitud Roll [°]": [accuracy_RPY_1[0], accuracy_RPY_2[0], accuracy_RPY_3[0]],
    "Exactitud Pitch [°]": [accuracy_RPY_1[1], accuracy_RPY_3[1], accuracy_RPY_3[1]],
    "Exactitud Yaw [°]": [accuracy_RPY_1[2], accuracy_RPY_3[2], accuracy_RPY_3[2]],

}

resumen_2 = {
    "Opción": archivos_seleccionados,
    "Norma PSD [W/Hz]": [norm_psd_RPY_1, norm_psd_RPY_2, norm_psd_RPY_3],
    "Norma Agilidad [s]": [norm_settling_time_1, norm_settling_time_2, norm_settling_time_3],
    "Norma Exactitud [°]": [accuracy_norms[0], accuracy_norms[1], accuracy_norms[2]]
}

tabla_1 = pd.DataFrame(resumen_1)
tabla_1_transposed = tabla_1.set_index("Opción").transpose()

tabla_2 = pd.DataFrame(resumen_2)
tabla_2_transposed = tabla_2.set_index("Opción").transpose()
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Imprimir la tabla
print("\n")
print(tabla_1_transposed)
print("\n")
print(tabla_2_transposed)

nombre_archivo_1 = os.path.splitext(os.path.basename(archivos_seleccionados[0]))[0]
nombre_archivo_2 = os.path.splitext(os.path.basename(archivos_seleccionados[1]))[0]
nombre_archivo_3 = os.path.splitext(os.path.basename(archivos_seleccionados[2]))[0]

#%%


fig0, axes0 = plt.subplots(nrows=3, ncols=3, figsize=(17, 8))

axes0[0,0].plot(t_aux, Roll_1, label= {nombre_archivo_1})
axes0[0,0].set_xlabel('Tiempo [s]')
axes0[0,0].set_ylabel('Roll [°]')
axes0[0,0].legend()
axes0[0,0].grid()
#axes0[0].set_ylim(-1, 1)  # Ajusta los límites en el eje Y

axes0[0,1].plot(t_aux, Roll_2, label= {nombre_archivo_2})
axes0[0,1].set_xlabel('Tiempo [s]')
axes0[0,1].set_ylabel('Roll [°]')
axes0[0,1].legend()
axes0[0,1].grid()
#axes0[0].set_ylim(-1, 1)  # Ajusta los límites en el eje Y

axes0[0,2].plot(t_aux, Roll_3, label= {nombre_archivo_3})
axes0[0,2].set_xlabel('Tiempo [s]')
axes0[0,2].set_ylabel('Roll [°]')
axes0[0,2].legend()
axes0[0,2].grid()
#axes0[0].set_ylim(-1, 1)  # Ajusta los límites en el eje Y

axes0[1,0].plot(t_aux, Pitch_1, label={nombre_archivo_1})
axes0[1,0].set_xlabel('Tiempo [s]')
axes0[1,0].set_ylabel('Pitch [°]')
axes0[1,0].legend()
axes0[1,0].grid()
# axes0[1].set_ylim(-20, -5)  # Ajusta los límites en el eje Y
# axes0[1].set_xlim(150000, 400000)  # Ajusta los límites en el eje Y

axes0[1,1].plot(t_aux, Pitch_2, label={nombre_archivo_2})
axes0[1,1].set_xlabel('Tiempo [s]')
axes0[1,1].set_ylabel('Pitch [°]')
axes0[1,1].legend()
axes0[1,1].grid()
# axes0[1].set_ylim(-20, -5)  # Ajusta los límites en el eje Y
# axes0[1].set_xlim(150000, 400000)  # Ajusta los límites en el eje Y

axes0[1,2].plot(t_aux, Pitch_3, label={nombre_archivo_3})
axes0[1,2].set_xlabel('Tiempo [s]')
axes0[1,2].set_ylabel('Pitch [°]')
axes0[1,2].legend()
axes0[1,2].grid()
# axes0[1].set_ylim(-20, -5)  # Ajusta los límites en el eje Y
# axes0[1].set_xlim(150000, 400000)  # Ajusta los límites en el eje Y

axes0[2,0].plot(t_aux, Yaw_1, label={nombre_archivo_1})
axes0[2,0].set_xlabel('Tiempo [s]')
axes0[2,0].set_ylabel('Yaw [°]')
axes0[2,0].legend()
axes0[2,0].grid()

axes0[2,1].plot(t_aux, Yaw_2, label={nombre_archivo_2})
axes0[2,1].set_xlabel('Tiempo [s]')
axes0[2,1].set_ylabel('Yaw [°]')
axes0[2,1].legend()
axes0[2,1].grid()

axes0[2,2].plot(t_aux, Yaw_3, label={nombre_archivo_3})
axes0[2,2].set_xlabel('Tiempo [s]')
axes0[2,2].set_ylabel('Yaw [°]')
axes0[2,2].legend()
axes0[2,2].grid()
# axes0[2].set_ylim(-20, -5)  # Ajusta los límites en el eje Y
# axes0[2].set_xlim(150000, 400000)  # Ajusta los límites en el eje Y

plt.tight_layout()
plt.show()

#%%
fig0, axes0 = plt.subplots(nrows=3, ncols=3, figsize=(17, 8))

axes0[0,0].plot(t_aux, Roll_1, label= {nombre_archivo_1})
axes0[0,0].set_xlabel('Tiempo [s]')
axes0[0,0].set_ylabel('Roll [°]')
axes0[0,0].legend()
axes0[0,0].grid()
# axes0[0,0].set_xlim(0, 60000)   # Ajusta los límites en el eje Y

axes0[0,1].plot(t_aux, Roll_2, label= {nombre_archivo_2})
axes0[0,1].set_xlabel('Tiempo [s]')
axes0[0,1].set_ylabel('Roll [°]')
axes0[0,1].legend()
axes0[0,1].grid()
# axes0[0,1].set_xlim(0, 60000)   # Ajusta los límites en el eje Y

axes0[0,2].plot(t_aux, Roll_3, label= {nombre_archivo_3})
axes0[0,2].set_xlabel('Tiempo [s]')
axes0[0,2].set_ylabel('Roll [°]')
axes0[0,2].legend()
axes0[0,2].grid()
# axes0[0,2].set_xlim(0, 60000)   # Ajusta los límites en el eje Y

axes0[1,0].plot(t_aux, Pitch_1, label={nombre_archivo_1},color='orange')
axes0[1,0].set_xlabel('Tiempo [s]')
axes0[1,0].set_ylabel('Pitch [°]')
axes0[1,0].legend()
axes0[1,0].grid()
# axes0[1,0].set_xlim(0, 60000)   # Ajusta los límites en el eje Y
# axes0[1].set_ylim(-20, -5)  # Ajusta los límites en el eje Y
# axes0[1].set_xlim(150000, 400000)  # Ajusta los límites en el eje Y

axes0[1,1].plot(t_aux, Pitch_2, label={nombre_archivo_2},color='orange')
axes0[1,1].set_xlabel('Tiempo [s]')
axes0[1,1].set_ylabel('Pitch [°]')
axes0[1,1].legend()
axes0[1,1].grid()
# axes0[1,1].set_xlim(0, 60000)   # Ajusta los límites en el eje Y
# axes0[1].set_ylim(-20, -5)  # Ajusta los límites en el eje Y
# axes0[1].set_xlim(150000, 400000)  # Ajusta los límites en el eje Y

axes0[1,2].plot(t_aux, Pitch_3, label={nombre_archivo_3},color='orange')
axes0[1,2].set_xlabel('Tiempo [s]')
axes0[1,2].set_ylabel('Pitch [°]')
axes0[1,2].legend()
axes0[1,2].grid()
# axes0[1,2].set_xlim(0, 60000)   # Ajusta los límites en el eje Y
# axes0[1].set_ylim(-20, -5)  # Ajusta los límites en el eje Y
# axes0[1].set_xlim(150000, 400000)  # Ajusta los límites en el eje Y

axes0[2,0].plot(t_aux, Yaw_1, label={nombre_archivo_1},color='green')
axes0[2,0].set_xlabel('Tiempo [s]')
axes0[2,0].set_ylabel('Yaw [°]')
axes0[2,0].legend()
axes0[2,0].grid()
# axes0[2,0].set_xlim(0, 60000)  # Ajusta los límites en el eje Y

axes0[2,1].plot(t_aux, Yaw_2, label={nombre_archivo_2},color='green')
axes0[2,1].set_xlabel('Tiempo [s]')
axes0[2,1].set_ylabel('Yaw [°]')
axes0[2,1].legend()
axes0[2,1].grid()
# axes0[2,1].set_xlim(0, 60000)   # Ajusta los límites en el eje Y

axes0[2,2].plot(t_aux, Yaw_3, label={nombre_archivo_3},color='green')
axes0[2,2].set_xlabel('Tiempo [s]')
axes0[2,2].set_ylabel('Yaw [°]')
axes0[2,2].legend()
axes0[2,2].grid()
# axes0[2,2].set_xlim(0, 60000)   # Ajusta los límites en el eje Y
# axes0[2].set_ylim(-20, -5)  # Ajusta los límites en el eje Y
# axes0[2].set_xlim(150000, 400000)  # Ajusta los límites en el eje Y

plt.tight_layout()
# Save the plot as an SVG file
plt.savefig('plot.pdf', format='pdf')

# Mostrar el gráfico
plt.show()

# pdf_path
#%%

xticks = range(0, 6000, 1000)
fig0, axes0 = plt.subplots(nrows=1, ncols=3, figsize=(15,4.9))

axes0[0].plot(t_aux, norms_RPY_1, label= "Low level sensors")
# axes0[0].plot(t_aux, norms_RPY_1, label= "Low level actuators")
axes0[0].set_xlabel('Time [s]', fontsize=15)
axes0[0].set_ylabel('$\eta$ [°]', fontsize=15)
axes0[0].tick_params(axis='both', which='major', labelsize=15)  # Ajusta el tamaño de las etiquetas de los ejes
axes0[0].set_xticks(xticks)
axes0[0].legend(fontsize=14)
axes0[0].grid()
# axes0[0].set_xlim(0, 60000)   # Ajusta los límites en el eje Y

axes0[1].plot(t_aux, norms_RPY_2, label="Medium level sensors",color='orange')
# axes0[1].plot(t_aux, norms_RPY_2, label="Medium level actuators",color='orange')
axes0[1].set_xlabel('Time [s]', fontsize=15)
axes0[1].set_ylabel('$\eta$ [°]', fontsize=15)
axes0[1].tick_params(axis='both', which='major', labelsize=15)
axes0[1].set_xticks(xticks)
axes0[1].legend(fontsize=14)
axes0[1].grid()
# axes0[1].set_xlim(0, 60000)   # Ajusta los límites en el eje Y
# axes0[1].set_ylim(-20, -5)  # Ajusta los límites en el eje Y
# axes0[1].set_xlim(150000, 400000)  # Ajusta los límites en el eje Y

axes0[2].plot(t_aux, norms_RPY_3, label="High level sensors",color='green')
# axes0[2].plot(t_aux, norms_RPY_3, label="High level actuators",color='green')
axes0[2].set_xlabel('Time [s]', fontsize=15)
axes0[2].set_ylabel('$\eta$ [°]', fontsize=15)
axes0[2].tick_params(axis='both', which='major', labelsize=15)
axes0[2].set_xticks(xticks)
axes0[2].legend(fontsize=14)
axes0[2].grid()
# axes0[2].set_xlim(0, 60000)  # Ajusta los límites en el eje Y
period = 5760  # segundos

for i in range(3):
    ax = axes0[i]
    ax2 = ax.twiny()  # eje superior
    ax2.set_xlim(ax.get_xlim())
    xticks_sec = ax.get_xticks()
    xticks_orbit = xticks_sec / period
    ax2.set_xticks(xticks_sec)
    ax2.set_xticklabels([f"{o:.1f}" for o in xticks_orbit], fontsize=14)
    ax2.set_xlabel("Orbits", fontsize=14)

plt.tight_layout()

# Save the plot as an SVG file
plt.savefig('RW_LQR_sensores.pdf', format='pdf')
# plt.savefig('RW_LQR_actuadores.pdf', format='pdf')
# plt.savefig('MT_LQR_sensores.pdf', format='pdf')
# plt.savefig('MT_LQR_actuadores.pdf', format='pdf')

# Mostrar el gráfico
plt.show()

# %% Gráfica artículo

# if opciones == [19,23,27]:
    
#     area_norm1x = np.trapz(ux_1, t_aux)
#     area_norm1y = np.trapz(uy_1, t_aux)
#     area_norm1z = np.trapz(uz_1, t_aux)
#     area_norm1 = np.array([area_norm1x,area_norm1y,area_norm1z])
#     area_norm1 = np.round(area_norm1,3)

#     area_norm2x = np.trapz(ux_2, t_aux)
#     area_norm2y = np.trapz(uy_2, t_aux)
#     area_norm2z = np.trapz(uz_2, t_aux)
#     area_norm2 = np.array([area_norm2x,area_norm2y,area_norm2z])
#     area_norm2 = np.round(area_norm2,3)

#     area_norm3x = np.trapz(ux_3, t_aux)
#     area_norm3y = np.trapz(uy_3, t_aux)
#     area_norm3z = np.trapz(uz_3, t_aux)
#     area_norm3 = np.array([area_norm3x,area_norm3y,area_norm3z])
#     area_norm3 = np.round(area_norm3,3)

#     # Datos Q y R
#     # diag_Q = np.array([1e4, 1e4, 1e4, 1e5, 1e5, 1e5])
#     # diag_R = np.array([0.1, 0.1, 0.1]) * 10
    
#     # Q = np.diag(diag_Q)
#     # R = np.diag(diag_R)
    
#     # def plot_norm_table_LQR(t, norm, diag_Q, diag_R, area, filename,letter):
#     def plot_norm_table_LQR(t, norm, area, filename,letter):
#         # Q1_str = f"[{', '.join([str(int(q)) for q in diag_Q[0:3]])}]"
#         # Q2_str = f"[{', '.join([str(int(q)) for q in diag_Q[3:6]])}]"
#         # R_str = f"[{', '.join([str(r) for r in diag_R])}]"
#         area_str = f"[{', '.join([str(r) for r in area])}]"
#         cell_text = [
#             # ["diag(Q)[0:3]", Q1_str],
#             # ["diag(Q)[3:6]", Q2_str],
#             # ["diag(R)", R_str],
#             # ["Area [$^\\circ \\cdot s$]", f"{area:.2f}"]
#             ["Area [$Am^2 \\cdot s$]", area_str]
#         ]
        
#         xticks = range(0, 6000, 1000)
#         period = 5760
    
#         fig, ax = plt.subplots(figsize=(11, 7), dpi=300)  # dpi agregado
#         ax.plot(t, norm, color='blue')
#         ax.set_xlabel('Time [s]', fontsize=19)
#         ax.set_ylabel('$\eta$ [°]', fontsize=19)
#         ax.set_xticks(xticks)
#         ax.tick_params(labelsize=18)
#         ax.grid()
#         ax.text(-0.075, 1.05, str(letter), transform=ax.transAxes, fontsize=35, fontweight='bold')

#         # Eje superior en órbitas
#         ax2 = ax.twiny()
#         ax2.set_xlim(ax.get_xlim())
#         xticks_sec = ax.get_xticks()
#         xticks_orbit = xticks_sec / period
#         ax2.set_xticks(xticks_sec)
#         ax2.set_xticklabels([f"{o:.1f}" for o in xticks_orbit], fontsize=18)
#         ax2.set_xlabel("Orbits", fontsize=18)
    
#         # Tabla dentro del gráfico
#         table = ax.table(
#             cellText=cell_text,
#             colWidths=[0.78,1.2],
#             cellLoc='left',
#             loc='upper right',
#             bbox=[0.35, 0.75, 0.6, 0.2]  # x, y, width, height (en coord. del gráfico)
#         )
#         table.auto_set_font_size(False)
#         table.set_fontsize(19)
    
#         plt.tight_layout()
#         plt.savefig(filename, format='pdf', bbox_inches='tight')
#         plt.show()
    
#     # Llamadas
#     # plot_norm_table_LQR(t_aux, norms_RPY_1, diag_Q, diag_R, area_norm1, "norm_LQR_low_table.pdf","(b)")
#     # plot_norm_table_LQR(t_aux, norms_RPY_2, diag_Q, diag_R, area_norm2, "norm_LQR_medium_table.pdf","(d)")
#     # plot_norm_table_LQR(t_aux, norms_RPY_3, diag_Q, diag_R, area_norm3, "norm_LQR_high_table.pdf","(f)")
#     plot_norm_table_LQR(t_aux, norms_RPY_1, area_norm1, "norm_LQR_low_table.pdf","(b)")
#     plot_norm_table_LQR(t_aux, norms_RPY_2, area_norm2, "norm_LQR_medium_table.pdf","(d)")
#     plot_norm_table_LQR(t_aux, norms_RPY_3, area_norm3, "norm_LQR_high_table.pdf","(f)")

# elif opciones == [10,14,18]:
#     # area_norm1 = np.trapz(norms_RPY_1, t_aux)
#     # area_norm2 = np.trapz(norms_RPY_2, t_aux)
#     # area_norm3 = np.trapz(norms_RPY_3, t_aux)
#     area_norm1x = np.trapz(ux_1, t_aux)
#     area_norm1y = np.trapz(uy_1, t_aux)
#     area_norm1z = np.trapz(uz_1, t_aux)
#     area_norm1 = np.array([area_norm1x,area_norm1y,area_norm1z])
#     area_norm1 = np.round(area_norm1,2)

#     area_norm2x = np.trapz(ux_2, t_aux)
#     area_norm2y = np.trapz(uy_2, t_aux)
#     area_norm2z = np.trapz(uz_2, t_aux)
#     area_norm2 = np.array([area_norm2x,area_norm2y,area_norm2z])
#     area_norm2 = np.round(area_norm2,2)

#     area_norm3x = np.trapz(ux_3, t_aux)
#     area_norm3y = np.trapz(uy_3, t_aux)
#     area_norm3z = np.trapz(uz_3, t_aux)
#     area_norm3 = np.array([area_norm3x,area_norm3y,area_norm3z])
#     area_norm3 = np.round(area_norm3,2)
    
#     # # Datos Q y R
#     # diag_Kp = np.array([77.6523, 36.7972, 66.6921])
#     # diag_Kd = np.array([94.1048, 74.1149, 98.26])
    
#     # K = np.hstack([np.diag(diag_Kp), np.diag(diag_Kd)])
    
    
#     # def plot_norm_table_PD(t, norm, diag_Kp, diag_Kd, area, filename,letter):
#     def plot_norm_table_PD(t, norm, area, filename,letter):
#         # Kp_str = f"[{', '.join([str(int(q)) for q in diag_Kp])}]"
#         # Kd_str = f"[{', '.join([str(int(q)) for q in diag_Kd])}]"
#         area_str = f"[{', '.join([str(r) for r in area])}]"

#         cell_text = [
#             # ["diag(Kp)", Kp_str],
#             # ["diag(Kd)", Kd_str],
#             # ["Area [$^\\circ \\cdot s$]", f"{area:.2f}"]
#             ["Area [$Am^2 \\cdot s$]", area_str]

#         ]
        
#         xticks = range(0, 6000, 1000)
#         period = 5760
    
#         fig, ax = plt.subplots(figsize=(11, 7), dpi=300)  # dpi agregado
#         ax.plot(t, norm, color='blue')
#         ax.set_xlabel('Time [s]', fontsize=19)
#         ax.set_ylabel('$\eta$ [°]', fontsize=19)
#         ax.set_xticks(xticks)
#         ax.tick_params(labelsize=18)
#         ax.grid()
#         ax.text(-0.075, 1.05, str(letter), transform=ax.transAxes, fontsize=35, fontweight='bold')

#         # Eje superior en órbitas
#         ax2 = ax.twiny()
#         ax2.set_xlim(ax.get_xlim())
#         xticks_sec = ax.get_xticks()
#         xticks_orbit = xticks_sec / period
#         ax2.set_xticks(xticks_sec)
#         ax2.set_xticklabels([f"{o:.1f}" for o in xticks_orbit], fontsize=18)
#         ax2.set_xlabel("Orbits", fontsize=18)
    
#         # Tabla dentro del gráfico
#         table = ax.table(
#             cellText=cell_text,
#             # colWidths=[0.8, 1.2],
#             # cellLoc='left',
#             # loc='upper right',
#             # bbox=[0.35, 0.6, 0.5, 0.3]  # x, y, width, height (en coord. del gráfico)
#             colWidths=[0.78,1.2],
#             cellLoc='left',
#             loc='upper right',
#             bbox=[0.35, 0.75, 0.6, 0.2]  # x, y, width, height (en coord. del gráfico)
#         )
#         table.auto_set_font_size(False)
#         table.set_fontsize(19)
    
#         plt.tight_layout()
#         plt.savefig(filename, format='pdf', bbox_inches='tight')
#         plt.show()
    
#     # Llamadas
#     # plot_norm_table_PD(t_aux, norms_RPY_1, diag_Kp, diag_Kd, area_norm1, "norm_PD_low_table.pdf","(a)")
#     # plot_norm_table_PD(t_aux, norms_RPY_2, diag_Kp, diag_Kd, area_norm2, "norm_PD_medium_table.pdf","(c)")
#     # plot_norm_table_PD(t_aux, norms_RPY_3, diag_Kp, diag_Kd, area_norm3, "norm_PD_high_table.pdf","(e)")
#     plot_norm_table_PD(t_aux, norms_RPY_1, area_norm1, "norm_PD_low_table.pdf","(a)")
#     plot_norm_table_PD(t_aux, norms_RPY_2, area_norm2, "norm_PD_medium_table.pdf","(c)")
#     plot_norm_table_PD(t_aux, norms_RPY_3, area_norm3, "norm_PD_high_table.pdf","(e)")


#%%


# #%%
# fig0, ax = plt.subplots(figsize=(15,5))  # Crea un solo set de ejes

# # Graficar los tres conjuntos de datos en la misma gráfica
# ax.plot(t_aux, norms_RPY_1, label='Costo bajo')
# ax.plot(t_aux, norms_RPY_2, label='Costo medio', color='orange')
# ax.plot(t_aux, norms_RPY_3, label='Costo alto', color='green')

# # Configurar etiquetas, leyenda y grid
# ax.set_xlabel('Tiempo [s]', fontsize=18)
# ax.set_ylabel('Error en ángulo de orientación [°]', fontsize=18)
# ax.legend(fontsize=18)
# ax.grid()

# # Ajustar límites del eje X
# ax.set_xlim(0, 60000)

# # Ajustar el tamaño de las etiquetas de los ticks
# ax.tick_params(axis='both', which='major', labelsize=18)

# plt.tight_layout()

# # Guardar la gráfica como archivo SVG
# plt.savefig('norm.svg', format='svg')

# # Mostrar la gráfica
# plt.show()
# #%%
# fig0, axes0 = plt.subplots(nrows=3, ncols=3, figsize=(17, 8))

# axes0[0,0].plot(t_aux, Roll_low_pass_1, label= {nombre_archivo_1})
# axes0[0,0].set_xlabel('Tiempo [s]')
# axes0[0,0].set_ylabel('Roll [°]')
# axes0[0,0].legend()
# axes0[0,0].grid()
# axes0[0,0].set_ylim(-5, 5)  # Ajusta los límites en el eje Y
# axes0[0,0].set_xlim(0, 5000)   # Ajusta los límites en el eje Y

# axes0[0,1].plot(t_aux, Roll_low_pass_2, label= {nombre_archivo_2})
# axes0[0,1].set_xlabel('Tiempo [s]')
# axes0[0,1].set_ylabel('Roll [°]')
# axes0[0,1].legend()
# axes0[0,1].grid()
# axes0[0,1].set_ylim(-5, 5)  # Ajusta los límites en el eje Y
# axes0[0,1].set_xlim(0, 5000)   # Ajusta los límites en el eje Y

# axes0[0,2].plot(t_aux, Roll_low_pass_3, label= {nombre_archivo_3})
# axes0[0,2].set_xlabel('Tiempo [s]')
# axes0[0,2].set_ylabel('Roll [°]')
# axes0[0,2].legend()
# axes0[0,2].grid()
# axes0[0,2].set_ylim(-5, 5)  # Ajusta los límites en el eje Y
# axes0[0,2].set_xlim(0, 5000)   # Ajusta los límites en el eje Y

# axes0[1,0].plot(t_aux, Pitch_low_pass_1, label={nombre_archivo_1})
# axes0[1,0].set_xlabel('Tiempo [s]')
# axes0[1,0].set_ylabel('Pitch [°]')
# axes0[1,0].legend()
# axes0[1,0].grid()
# axes0[1,0].set_ylim(-7, 7)  # Ajusta los límites en el eje Y
# axes0[1,0].set_xlim(0, 5000)   # Ajusta los límites en el eje Y

# axes0[1].set_ylim(-20, -5)  # Ajusta los límites en el eje Y
# axes0[1].set_xlim(150000, 400000)  # Ajusta los límites en el eje Y

# axes0[1,1].plot(t_aux, Pitch_low_pass_2, label={nombre_archivo_2})
# axes0[1,1].set_xlabel('Tiempo [s]')
# axes0[1,1].set_ylabel('Pitch [°]')
# axes0[1,1].legend()
# axes0[1,1].grid()
# axes0[1,1].set_ylim(-7, 7)  # Ajusta los límites en el eje Y
# axes0[1,1].set_xlim(0, 5000)   # Ajusta los límites en el eje Y

# # axes0[1].set_ylim(-20, -5)  # Ajusta los límites en el eje Y
# # axes0[1].set_xlim(150000, 400000)  # Ajusta los límites en el eje Y

# axes0[1,2].plot(t_aux, Pitch_low_pass_3, label={nombre_archivo_3})
# axes0[1,2].set_xlabel('Tiempo [s]')
# axes0[1,2].set_ylabel('Pitch [°]')
# axes0[1,2].legend()
# axes0[1,2].grid()
# axes0[1,2].set_ylim(-7, 7)  # Ajusta los límites en el eje Y
# axes0[1,2].set_xlim(0, 5000)   # Ajusta los límites en el eje Y

# # axes0[1].set_ylim(-20, -5)  # Ajusta los límites en el eje Y
# # axes0[1].set_xlim(150000, 400000)  # Ajusta los límites en el eje Y

# axes0[2,0].plot(t_aux, Yaw_low_pass_1, label={nombre_archivo_1})
# axes0[2,0].set_xlabel('Tiempo [s]')
# axes0[2,0].set_ylabel('Yaw [°]')
# axes0[2,0].legend()
# axes0[2,0].grid()
# axes0[2,0].set_ylim(-5, 5)  # Ajusta los límites en el eje Y
# axes0[2,0].set_xlim(0, 5000)   # Ajusta los límites en el eje Y

# axes0[2,1].plot(t_aux, Yaw_low_pass_2, label={nombre_archivo_2})
# axes0[2,1].set_xlabel('Tiempo [s]')
# axes0[2,1].set_ylabel('Yaw [°]')
# axes0[2,1].legend()
# axes0[2,1].grid()
# axes0[2,1].set_ylim(-5, 5)  # Ajusta los límites en el eje Y
# axes0[2,1].set_xlim(0, 5000)   # Ajusta los límites en el eje Y

# axes0[2,2].plot(t_aux, Yaw_low_pass_3, label={nombre_archivo_3})
# axes0[2,2].set_xlabel('Tiempo [s]')
# axes0[2,2].set_ylabel('Yaw [°]')
# axes0[2,2].legend()
# axes0[2,2].grid()
# axes0[2,2].set_ylim(-5, 5)  # Ajusta los límites en el eje Y
# axes0[2,2].set_xlim(0, 5000)   # Ajusta los límites en el eje Y
# # axes0[2].set_ylim(-20, -5)  # Ajusta los límites en el eje Y
# # axes0[2].set_xlim(150000, 400000)  # Ajusta los límites en el eje Y

# plt.tight_layout()
# plt.show()

# #%%
# fig0, axes0 = plt.subplots(nrows=3, ncols=1, figsize=(10, 10))

# axes0[0].plot(t_aux, norms_RPY_low_pass_1, label= {nombre_archivo_1})
# axes0[0].set_xlabel('Tiempo [s]')
# axes0[0].set_ylabel('Norma lp_RPY [°]')
# axes0[0].legend()
# axes0[0].grid()
# # axes0[0].set_xlim(0, 60000)   # Ajusta los límites en el eje Y
# axes0[0].set_ylim(-7, 7)  # Ajusta los límites en el eje Y

# axes0[1].plot(t_aux, norms_RPY_low_pass_2, label={nombre_archivo_2},color='orange')
# axes0[1].set_xlabel('Tiempo [s]')
# axes0[1].set_ylabel('Norma lp_RPY [°]')
# axes0[1].legend()
# axes0[1].grid()
# # axes0[1].set_xlim(0, 60000)   # Ajusta los límites en el eje Y
# axes0[1].set_ylim(-7, 7)  # Ajusta los límites en el eje Y

# axes0[2].plot(t_aux, norms_RPY_low_pass_3, label={nombre_archivo_3},color='green')
# axes0[2].set_xlabel('Tiempo [s]')
# axes0[2].set_ylabel('Norma lp_RPY [°]')
# axes0[2].legend()
# axes0[2].grid()
# # axes0[2].set_xlim(0, 60000)  # Ajusta los límites en el eje Y
# axes0[2].set_ylim(-7, 7)  # Ajusta los límites en el eje Y


# plt.tight_layout()
# # # Save the plot as an SVG file
# # plt.savefig('plot.svg', format='svg')

# # Mostrar el gráfico
# plt.show()
#%%

[MSE_cuat_1, MSE_omega_1]  = functions_05.cuat_MSE_NL(q0_real_1, q1_real_1, q2_real_1, q3_real_1, w0_real_1, w1_real_1, w2_real_1, q0_est_1, q1_est_1, q2_est_1, q3_est_1, w0_est_1, w1_est_1, w2_est_1)   
[asd,asd,mse_roll_1,mse_pitch_1,mse_yaw_1] = functions_05.RPY_MSE(t_aux, q0_est_1, q1_est_1, q2_est_1, q3_est_1, q0_real_1, q1_real_1, q2_real_1, q3_real_1)   

[MSE_cuat_2, MSE_omega_2]  = functions_05.cuat_MSE_NL(q0_real_2, q1_real_2, q2_real_2, q3_real_2, w0_real_2, w1_real_2, w2_real_2, q0_est_2, q1_est_2, q2_est_2, q3_est_2, w0_est_2, w1_est_2, w2_est_2)   
[asd,asd,mse_roll_2,mse_pitch_2,mse_yaw_2] = functions_05.RPY_MSE(t_aux, q0_est_2, q1_est_2, q2_est_2, q3_est_2, q0_real_2, q1_real_2, q2_real_2, q3_real_2)   

[MSE_cuat_3, MSE_omega_3]  = functions_05.cuat_MSE_NL(q0_real_3, q1_real_3, q2_real_3, q3_real_3, w0_real_3, w1_real_3, w2_real_3, q0_est_3, q1_est_3, q2_est_3, q3_est_3, w0_est_3, w1_est_3, w2_est_3)   
[asd,asd,mse_roll_3,mse_pitch_3,mse_yaw_3] = functions_05.RPY_MSE(t_aux, q0_est_3, q1_est_3, q2_est_3, q3_est_3, q0_real_3, q1_real_3, q2_real_3, q3_real_3)   



# quats = np.array([0,1,2,3])
# plt.figure(figsize=(12, 6))
# plt.scatter(quats[0], MSE_cuat_1[0], label='mse q0_1', color='r',marker='*')
# plt.scatter(quats[1], MSE_cuat_1[1], label='mse q1_1', color='b',marker='*')
# plt.scatter(quats[2], MSE_cuat_1[2], label='mse q2_1', color='k',marker='*')
# plt.scatter(quats[3], MSE_cuat_1[3], label='mse q3_1', color='g',marker='*')
# plt.xlabel('Cuaterniones')
# plt.ylabel('Mean Square Error [-]')
# plt.legend()
# plt.title('MSE de cada cuaternion entre lineal discreto y kalman lineal discreto sen1_act1_RW')
# # plt.xlim(20000,100000)
# # plt.ylim(-0.005,0.005)
# plt.grid()
# plt.show()

# plt.figure(figsize=(12, 6))
# plt.scatter(quats[0], MSE_cuat_2[0], label='mse q0_2', color='r',marker='*')
# plt.scatter(quats[1], MSE_cuat_2[1], label='mse q1_2', color='b',marker='*')
# plt.scatter(quats[2], MSE_cuat_2[2], label='mse q2_2', color='k',marker='*')
# plt.scatter(quats[3], MSE_cuat_2[3], label='mse q3_2', color='g',marker='*')
# plt.xlabel('Cuaterniones')
# plt.ylabel('Mean Square Error [-]')
# plt.legend()
# plt.title('MSE de cada cuaternion entre lineal discreto y kalman lineal discreto opcion 2')
# # plt.xlim(20000,100000)
# # plt.ylim(-0.005,0.005)
# plt.grid()
# plt.show()

# plt.figure(figsize=(12, 6))
# plt.scatter(quats[0], MSE_cuat_3[0], label='mse q0_3', color='r',marker='*')
# plt.scatter(quats[1], MSE_cuat_3[1], label='mse q1_3', color='b',marker='*')
# plt.scatter(quats[2], MSE_cuat_3[2], label='mse q2_3', color='k',marker='*')
# plt.scatter(quats[3], MSE_cuat_3[3], label='mse q3_3', color='g',marker='*')
# plt.xlabel('Cuaterniones')
# plt.ylabel('Mean Square Error [-]')
# plt.legend()
# plt.title('MSE de cada cuaternion entre lineal discreto y kalman lineal discreto opcion 3')
# # plt.xlim(20000,100000)
# # plt.ylim(-0.005,0.005)
# plt.grid()
# plt.show()