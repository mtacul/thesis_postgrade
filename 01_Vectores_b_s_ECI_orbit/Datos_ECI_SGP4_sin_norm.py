# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 21:26:19 2024

@author: nachi
"""

#%% Librerias a utilizar
from skyfield.api import utc
from datetime import datetime, timedelta
from sgp4.earth_gravity import wgs84
from sgp4.io import twoline2rv
import pyIGRF
import numpy as np
import functions_tesis

#%% Define la ubicación de tu archivo TLE y encuentra la condicion inicial del estado
tle_file = "suchai_3.txt"

# Lee el contenido del archivo
with open(tle_file, "r") as file:
    tle_lines = file.read().splitlines()
    
# Asegúrate de que el archivo contiene al menos dos líneas de TLE
if len(tle_lines) < 2:
    print("El archivo TLE debe contener al menos dos líneas.")
else:
    # Convierte las líneas del archivo en un objeto Satrec
    satellite = twoline2rv(tle_lines[0], tle_lines[1], wgs84)
    
    # Define la fecha inicial
    start_time = datetime(2023, 11, 1, 12, 0, 0)  # Ejemplo: 1 de noviembre de 2023, 12:00:00

    # Define el tiempo de propagación en segundos 
    # propagation_time = 60*60*24*187
    propagation_time = 60*60*24


    #posicion y velocidad del satelite en la fecha inicial
    position_i, velocity_i = satellite.propagate(start_time.year, start_time.month, start_time.day,
        start_time.hour, start_time.minute, start_time.second)
    
    #Para transformar el vector a LLA inicial
    start_time_gps = datetime(2023, 11, 1, 12, 0, 0,tzinfo=utc)  # Ejemplo: 5 de julio de 2023, 12:00:00
    lla_i = functions_tesis.eci2lla(position_i,start_time_gps)
    
    #Obtener fuerzas magneticas de la Tierra inicial
    Bi = pyIGRF.igrf_value(lla_i[0],lla_i[1],lla_i[2]/1000, start_time.year)
    Bi_fn = np.array([Bi[3], Bi[4], Bi[5]])
    Bi_f = Bi_fn/np.linalg.norm(Bi_fn)
    
    #obtener vector sol ECI inicial
    jd2000i = functions_tesis.datetime_to_jd2000(start_time)
    sunvectorin = functions_tesis.sun_vector(jd2000i)
    sunvectori = sunvectorin / np.linalg.norm(sunvectorin)
    

    
    #%% Listas donde guardar las variables ECI y body
    
    positions = [position_i]
    velocities = [velocity_i]
    latitudes = [lla_i[0]]
    longitudes = [lla_i[1]]
    altitudes = [lla_i[2]]
    Bx_IGRF = [Bi_f[0]]
    By_IGRF = [Bi_f[1]]
    Bz_IGRF = [Bi_f[2]]
    vsun_x = [sunvectori[0]]
    vsun_y = [sunvectori[1]]
    vsun_z = [sunvectori[2]]


    
    #%% Propagacion SGP4 y obtencion de vectores magneticos y sol en ECI a traves del tiempo
    
    # Inicializa el tiempo actual un segundo despues del inicio
    deltat = 2
    current_time = start_time+ timedelta(seconds=deltat)
    current_time_gps = start_time_gps + timedelta(seconds=deltat)
    
    t0 = 0
    t_aux = [t0]
    
    while current_time < start_time + timedelta(seconds=propagation_time):
        t0 = t0 + deltat
        position, velocity = satellite.propagate(
            current_time.year, current_time.month, current_time.day,
            current_time.hour, current_time.minute, current_time.second
        )
            
        lla = functions_tesis.eci2lla(position,current_time_gps)
            
        #Obtener fuerzas magneticas de la Tierra
        Bm_PD = pyIGRF.igrf_value(lla[0],lla[1],lla[2]/1000, current_time.year)
        B_f = np.array([Bm_PD[3], Bm_PD[4], Bm_PD[5]])

        jd2000 = functions_tesis.datetime_to_jd2000(current_time)
        print(current_time)
        sunvector = functions_tesis.sun_vector(jd2000)
        
        t_aux.append(t0)
        positions.append(position)
        velocities.append(velocity)
        latitudes.append(lla[0])
        longitudes.append(lla[1])
        altitudes.append(lla[2])
        Bx_IGRF.append(B_f[0])
        By_IGRF.append(B_f[1])
        Bz_IGRF.append(B_f[2])
        vsun_x.append(sunvector[0])
        vsun_y.append(sunvector[1])
        vsun_z.append(sunvector[2])
             
                 
        current_time += timedelta(seconds= deltat)
        current_time_gps += timedelta(seconds=deltat)
            
    t_aux = np.array(t_aux)
    positions = np.array(positions)
    velocities = np.array(velocities)
    latitudes = np.array(latitudes)
    longitudes = np.array(longitudes)
    altitudes = np.array(altitudes)
    Bx_IGRF = np.array(Bx_IGRF)
    By_IGRF = np.array(By_IGRF)
    Bz_IGRF = np.array(Bz_IGRF)
    vsun_x = np.array(vsun_x)
    vsun_y = np.array(vsun_z)
    vsun_z = np.array(vsun_z)
    
    #%% Guardar datos en un archivo csv
    
    # Nombre del archivo
    archivo = "vectores_14h_s_norm.csv"

    # Abrir el archivo en modo escritura
    with open(archivo, 'w') as f:
        # Escribir los encabezados
        f.write("t_aux, position_x,position_y,position_z, velocity_x, velocity_y, velocity_z, latitudes, longitudes, altitudes, Bx_IGRF, By_IGRF, Bz_IGRF, vsun_x, vsun_y, vsun_z\n")
    
        # Escribir los datos en filas
        for i in range(len(t_aux)):
            f.write("{},{},{},{},{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(
                t_aux[i], positions[i,0], positions[i,1],positions[i,2],velocities[i,0],velocities[i,1],velocities[i,2],
                latitudes[i],longitudes[i], altitudes[i], Bx_IGRF[i], By_IGRF[i],
                Bz_IGRF[i], vsun_x[i], vsun_y[i], vsun_z[i]
            ))
    
    print("Vectores guardados en el archivo:", archivo)
    
    #%%
    import pandas as pd
    import matplotlib.pyplot as plt

    # Nombre del archivo CSV
    archivo_csv = "vectores_14h_s_norm.csv"

    # Leer el archivo CSV en un DataFrame de pandas
    df = pd.read_csv(archivo_csv)

    # Convertir el DataFrame a un array de NumPy
    array_datos = df.to_numpy()
    
    t_aux = array_datos[:,0]
    position = np.transpose(np.vstack((array_datos[:,1], array_datos[:,2], array_datos[:,3])))
    velocity = np.transpose(np.vstack((array_datos[:,4], array_datos[:,5], array_datos[:,6])))
    Bx_IGRF = array_datos[:,10]
    By_IGRF = array_datos[:,11]
    Bz_IGRF = array_datos[:,12]
    vsun_x = array_datos[:,13]
    vsun_y = array_datos[:,14]
    vsun_z = array_datos[:,15]
    
    plt.figure(figsize=(12, 6))
    plt.plot(t_aux, position[:,0], label='Posición en X')
    plt.plot(t_aux, position[:,1], label='Posición en Y')
    plt.plot(t_aux, position[:,2], label='Posición en Z')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Posición (ECI) [km]')
    plt.legend()
    plt.title('Posición del satélite durante 24 horas')
    plt.xlim(0,5600)
    # plt.ylim(-15,2)
    plt.grid()
    plt.show()
    # plt.set_yli1m(-10, 10)  # Ajusta los límites en el eje Y
    
    plt.figure(figsize=(12, 6))
    plt.plot(t_aux, velocity[:,0], label='Velocidad en X')
    plt.plot(t_aux, velocity[:,1], label='Velocidad en Y')
    plt.plot(t_aux, velocity[:,2], label='Velocidad en Z')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Velocidad (ECI) [km/s]')
    plt.legend()
    plt.title('Velocidad del satélite durante 24 horas')
    # plt.xlim(0.8e7,1.7e7)
    # plt.ylim(-15,2)
    plt.grid()
    plt.show()
    # plt.set_yli1m(-10, 10)  # Ajusta los límites en el eje Y
    
    plt.figure(figsize=(12, 6))
    plt.plot(t_aux, Bx_IGRF, label='Fuerza magnética en X')
    plt.plot(t_aux, By_IGRF, label='Fuerza magnética en Y')
    plt.plot(t_aux, Bz_IGRF, label='Fuerza magnética en Z')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Fuerza magnética [nT]')
    plt.legend()
    plt.title('Fuerza magnetica durante 24 horas')
    # plt.xlim(0.8e7,1.7e7)
    # plt.ylim(-15,2)
    plt.grid()
    plt.show()
    # plt.set_yli1m(-10, 10)  # Ajusta los límites en el eje Y