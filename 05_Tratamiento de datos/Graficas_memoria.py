# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 02:24:57 2024

@author: nachi
"""

#%% Librerias a utilizar
import matplotlib.pyplot as plt
import pandas as pd
import functions_05
from scipy.signal import welch
import numpy as np
import scipy.stats as stats

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
print("Seleccione un archivo CSV para abrir (1:bad, 2:med, 3:good):")
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
q0_real = array_datos_c[:,4]
q1_real = array_datos_c[:,5]
q2_real = array_datos_c[:,6]
q3_real = array_datos_c[:,7]
q0_est = array_datos_c[:,8]
q1_est = array_datos_c[:,9]
q2_est = array_datos_c[:,10]
q3_est = array_datos_c[:,11]
w0_est = array_datos_c[:,12]
w1_est = array_datos_c[:,13]
w2_est = array_datos_c[:,14]
w0_real = array_datos_c[:,15]
w1_real = array_datos_c[:,16]
w2_real = array_datos_c[:,17]
Roll_low_pass = array_datos_c[:,18]
Pitch_low_pass =  array_datos_c[:,19]
Yaw_low_pass =  array_datos_c[:,20]

#%% Densidad espectro potencia para el jitter

#Filtro pasa alto para el Jitter y pasa bajo para exactitud de apuntamiento y agilidad
Roll_high_pass = functions_05.high_pass_filter(Roll, 10, len(t_aux))
Pitch_high_pass = functions_05.high_pass_filter(Pitch, 10, len(t_aux))
Yaw_high_pass = functions_05.high_pass_filter(Yaw, 10, len(t_aux))

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

#%% Encontrar el tiempo de asentamiento en segundos de cada angulo de Euler

settling_band_R = 5
settling_band_P = 5
settling_band_Y = 5

settling_error_sup_R = np.full(len(t_aux),settling_band_R)
settling_error_inf_R = np.full(len(t_aux),-settling_band_R)

settling_error_sup_P = np.full(len(t_aux),settling_band_P)
settling_error_inf_P = np.full(len(t_aux),-settling_band_P)

settling_error_sup_Y = np.full(len(t_aux),settling_band_Y)
settling_error_inf_Y = np.full(len(t_aux),-settling_band_Y)


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


#%%Seccion para calcular la exactitud de apuntamiento en cada angulo de Euler

time_R = np.array(settling_times_R[-1])
data_R = Roll[int(time_R[0]/2):int(time_R[1]/2)]
# Calcular media y desviación estándar
media_R = np.mean(data_R)
sigma_R = np.std(data_R)
# Calcular los límites de 3 sigma
lim_inf_R = media_R - 3 * sigma_R
lim_sup_R = media_R + 3 * sigma_R
vals_confianza_R = np.sum((data_R >= lim_inf_R) & (data_R <= lim_sup_R))
porc_confianza_R = (vals_confianza_R / len(data_R)) * 100

# # Generar valores para la campana de Gauss
# x_R = np.linspace(media_R - 4*sigma_R, media_R + 4*sigma_R, len(data_R))
# y_R = stats.norm.pdf(x_R, media_R, sigma_R)
# # Crear el histograma de los datos
# plt.hist(data_R, bins=30, density=True, alpha=0.6, color='g')
# # Graficar la campana de Gauss
# plt.plot(x_R, y_R, 'k', linewidth=2)
# plt.title("Distribución Normal del Roll obtenido")
# plt.xlabel("Valor del Roll [°]")
# plt.ylabel("Densidad de probabilidad")
# plt.show()

accuracy_R = 3*sigma_R


time_P = np.array(settling_times_P[-1])
data_P = Pitch[int(time_P[0]/2):int(time_P[1]/2)]
# Calcular media y desviación estándar
media_P = np.mean(data_P)
sigma_P = np.std(data_P)
# Calcular los límites de 3 sigma
lim_inf_P = media_P - 3 * sigma_P
lim_sup_P = media_P + 3 * sigma_P
vals_confianza_P = np.sum((data_P >= lim_inf_P) & (data_P <= lim_sup_P))
porc_confianza_P = (vals_confianza_P / len(data_P)) * 100

# Generar valores para la campana de Gauss
x_P = np.linspace(media_P - 4*sigma_P, media_P + 4*sigma_P, len(data_P))
y_P = stats.norm.pdf(x_P, media_P, sigma_P)
# Crear el histograma de los datos
plt.hist(data_P, bins=30, density=True, alpha=0.6, color='g')
# Graficar la campana de Gauss
plt.plot(x_P, y_P, 'k', linewidth=2)
plt.title("Distribución Normal del Roll obtenido")
plt.xlabel("Valor del Roll [°]")
plt.ylabel("Densidad de probabilidad")
plt.show()

accuracy_P = 3*sigma_P


time_Y = np.array(settling_times_Y[-1])
data_Y = Yaw[int(time_Y[0]/2):int(time_Y[1]/2)]
# Calcular media y desviación estándar
media_Y = np.mean(data_Y)
sigma_Y = np.std(data_Y)
# Calcular los límites de 3 sigma
lim_inf_Y = media_Y - 3 * sigma_Y
lim_sup_Y = media_Y + 3 * sigma_Y
vals_confianza_Y = np.sum((data_Y >= lim_inf_Y) & (data_Y <= lim_sup_Y))
porc_confianza_Y = (vals_confianza_Y / len(data_Y)) * 100

# Generar valores para la campana de Gauss
x_Y = np.linspace(media_Y - 4*sigma_Y, media_Y + 4*sigma_Y, len(data_Y))
y_Y = stats.norm.pdf(x_Y, media_Y, sigma_Y)
# Crear el histograma de los datos
plt.hist(data_Y, bins=30, density=True, alpha=0.6, color='g')
# Graficar la campana de Gauss
plt.plot(x_Y, y_Y, 'k', linewidth=2)
plt.title("Distribución Normal del Roll obtenido")
plt.xlabel("Valor del Roll [°]")
plt.ylabel("Densidad de probabilidad")
plt.show()

accuracy_Y = 3*sigma_Y


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

axes0[0].plot(t_aux, q0_real, label='q0_real')
axes0[0].plot(t_aux, q1_real, label='q1_real')
axes0[0].plot(t_aux, q2_real, label='q2_real')
axes0[0].plot(t_aux, q3_real, label='q3_real')
axes0[0].set_xlabel('Tiempo [s]')
axes0[0].set_ylabel('cuaternión [-]')
axes0[0].legend()
axes0[0].set_title('cuaterniones del modelo llevados al equilibrio')
axes0[0].grid()
#axes0[0].set_ylim(-1, 1)  # Ajusta los límites en el eje Y

axes0[1].plot(t_aux, w0_est, label='w0_est')
axes0[1].plot(t_aux, w1_est, label='w1_est')
axes0[1].plot(t_aux, w2_est, label='w2_est')
axes0[1].set_xlabel('Tiempo [s]')
axes0[1].set_ylabel('velocidad angular [rad/s]')
axes0[1].legend()
axes0[1].set_title('velocidades angulares estimadas por EKF llevadas al equilibrio')
axes0[1].grid()


axes0[2].plot(t_aux, q0_est, label='q0_est')
axes0[2].plot(t_aux, q1_est, label='q1_est')
axes0[2].plot(t_aux, q2_est, label='q2_est')
axes0[2].plot(t_aux, q3_est, label='q3_est')
axes0[2].set_xlabel('Tiempo [s]')
axes0[2].set_ylabel('cuaternión EKF [-]')
axes0[2].legend()
axes0[2].set_title('cuaterniones estimados por EKF llevados al equilibrio')
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

fig0, axes0 = plt.subplots(nrows=2, ncols=1, figsize=(7, 7))

axes0[0].plot(t_aux, Roll_low_pass, label='Roll')
axes0[0].set_xlabel('Tiempo [s]')
axes0[0].set_ylabel('Ángulos de Euler [°]')
axes0[0].legend()
axes0[0].grid()
#axes0[0].set_ylim(-1, 1)  # Ajusta los límites en el eje Y

axes0[1].plot(t_aux, Roll_low_pass, label='Roll')
axes0[1].plot(t_aux, settling_error_sup_R , label='SET_sup_roll')
axes0[1].plot(t_aux, settling_error_inf_R , label='SET_inf_roll')
axes0[1].set_xlabel('Tiempo [s]')
axes0[1].set_ylabel('Ángulos de Euler [°]')
axes0[1].legend()
# axes0[1].set_title('Filtros pasa bajo')
axes0[1].grid()
axes0[1].set_ylim(-10,10)  # Ajusta los límites en el eje Y

plt.tight_layout()
plt.show()

fig0, axes0 = plt.subplots(nrows=2, ncols=1, figsize=(7, 7))

axes0[0].plot(t_aux, Pitch_low_pass, label='Pitch')
axes0[0].set_xlabel('Tiempo [s]')
axes0[0].set_ylabel('Ángulos de Euler [°]')
axes0[0].legend()
axes0[0].grid()
#axes0[0].set_ylim(-1, 1)  # Ajusta los límites en el eje Y

axes0[1].plot(t_aux, Pitch_low_pass, label='Pitch')
axes0[1].plot(t_aux, settling_error_sup_P , label='SET_sup_pitch')
axes0[1].plot(t_aux, settling_error_inf_P , label='SET_inf_pitch')
axes0[1].set_xlabel('Tiempo [s]')
axes0[1].set_ylabel('Ángulos de Euler [°]')
axes0[1].legend()
# axes0[1].set_title('Filtros pasa bajo')
axes0[1].grid()
axes0[1].set_ylim(-10,10)  # Ajusta los límites en el eje Y

plt.tight_layout()
plt.show()

fig0, axes0 = plt.subplots(nrows=2, ncols=1, figsize=(7, 7))

axes0[0].plot(t_aux, Yaw_low_pass, label='Yaw')
axes0[0].set_xlabel('Tiempo [s]')
axes0[0].set_ylabel('Ángulos de Euler [°]')
axes0[0].legend()
axes0[0].grid()
#axes0[0].set_ylim(-1, 1)  # Ajusta los límites en el eje Y

axes0[1].plot(t_aux, Yaw_low_pass, label='Yaw')
axes0[1].plot(t_aux, settling_error_sup_Y , label='SET_sup_yaw')
axes0[1].plot(t_aux, settling_error_inf_Y , label='SET_inf_yaw')
axes0[1].set_xlabel('Tiempo [s]')
axes0[1].set_ylabel('Ángulos de Euler [°]')
axes0[1].legend()
# axes0[1].set_title('Filtros pasa bajo')
axes0[1].grid()
axes0[1].set_ylim(-10,10)  # Ajusta los límites en el eje Y

plt.tight_layout()
plt.show()

# plt.figure(figsize=(10, 6))
# plt.semilogy(frequencies_R, psd_R_R, label='PSD Roll')
# plt.fill_between(frequencies_R[indices_bandwidth_1_R], psd_R_R[indices_bandwidth_1_R], alpha=0.3, label='Ancho de banda 1')
# plt.title('Densidad Espectral de Potencia con Anchos de Banda Específicos en Roll')
# plt.xlabel('Frecuencia (Hz)')
# plt.ylabel('Densidad Espectral de Potencia')
# plt.legend()
# plt.grid(True)
# plt.show()

# plt.figure(figsize=(10, 6))
# plt.semilogy(frequencies_P, psd_P_R, label='PSD Pitch')
# plt.fill_between(frequencies_P[indices_bandwidth_1_P], psd_P_R[indices_bandwidth_1_P], alpha=0.3, label='Ancho de banda 1')
# plt.title('Densidad Espectral de Potencia con Anchos de Banda Específicos en Pitch')
# plt.xlabel('Frecuencia (Hz)')
# plt.ylabel('Densidad Espectral de Potencia')
# plt.legend()
# plt.grid(True)
# plt.show()


# plt.figure(figsize=(10, 6))
# plt.semilogy(frequencies_Y, psd_Y_R, label='PSD Yaw')
# plt.fill_between(frequencies_Y[indices_bandwidth_1_Y], psd_Y_R[indices_bandwidth_1_Y], alpha=0.3, label='Ancho de banda 1')
# plt.title('Densidad Espectral de Potencia con Anchos de Banda Específicos en Yaw')
# plt.xlabel('Frecuencia (Hz)')
# plt.ylabel('Densidad Espectral de Potencia')
# plt.legend()
# plt.grid(True)
# plt.show()

[MSE_cuat, MSE_omega]  = functions_05.cuat_MSE_NL(q0_real, q1_real, q2_real, q3_real, w0_real, w1_real, w2_real, q0_est, q1_est, q2_est, q3_est, w0_est, w1_est, w2_est)   
[RPY_all_est,RPY_all_id,mse_roll,mse_pitch,mse_yaw] = functions_05.RPY_MSE(t_aux, q0_est, q1_est, q2_est, q3_est, q0_real, q1_real, q2_real, q3_real)   

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