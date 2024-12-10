# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 00:52:05 2024

@author: nachi
"""

#%% Librerias a utilizar
from skyfield.api import utc
from datetime import datetime, timedelta
from sgp4.earth_gravity import wgs84
from sgp4.io import twoline2rv
import pyIGRF
import numpy as np
import functions_01
import matplotlib.pyplot as plt
from astropy.time import Time

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

    #posicion y velocidad del satelite en la fecha inicial
    position_i, velocity_i = satellite.propagate(start_time.year, start_time.month, start_time.day,
        start_time.hour, start_time.minute, start_time.second)
    
    pos_itrs_coords_i, vel_itrs_coords_i  = functions_01.teme_to_itrs(position_i, velocity_i, start_time)
    pos_gcrs_coords_i = functions_01.itrs_to_gcrs(pos_itrs_coords_i, start_time)
    vel_gcrs_coords_i = functions_01.itrs_to_gcrs(vel_itrs_coords_i, start_time)

    #Para transformar el vector a LLA inicial
    start_time_gps = datetime(2023, 11, 1, 12, 0, 0,tzinfo=utc)  # Ejemplo: 5 de julio de 2023, 12:00:00
    lla_i = functions_01.eci2lla(position_i,start_time_gps)
    
    # Definir el tiempo en UTC
    utc_time_i = Time(start_time, scale="utc")
    # Convertir a UT1
    ut1_time_i = utc_time_i.ut1.value
        
    #obtener vector sol ECI inicial
    jd2000i = functions_01.datetime_to_jd2000(ut1_time_i)
    sunvectorin = functions_01.sun_vector(jd2000i)
    sunvectori = sunvectorin / np.linalg.norm(sunvectorin)
    
    #Obtener fuerzas magneticas de la Tierra inicial
    jd2000i_UTC = functions_01.datetime_to_jd2000(utc_time_i.value)
    Bi = pyIGRF.igrf_value(lla_i[0],lla_i[1],lla_i[2]/1000, start_time.year)
    Bi_fn = np.array([Bi[3], Bi[4], Bi[5]])
    Bi_ECEF = functions_01.NED_2_ECEF(lla_i[0], lla_i[1], Bi_fn)
    Bi_ECI = functions_01.ECEF_2_ECI(jd2000i_UTC, Bi_ECEF)
    Bi_f = Bi_fn/np.linalg.norm(Bi_fn)
    Bi_ECI_s = Bi_ECI / np.linalg.norm(Bi_ECI)
    
    # Define el tiempo de propagación en segundos 
    # propagation_time = 5762 * 5
    propagation_time = 5762 * 1
    
    #%% Listas donde guardar las variables ECI y body
    
    positions = [position_i]
    velocities = [velocity_i]
    pos_itrs = [pos_itrs_coords_i.cartesian.xyz]
    vel_itrs = [vel_itrs_coords_i.cartesian.xyz]
    pos_gcrs = [pos_gcrs_coords_i.cartesian.xyz]
    vel_gcrs = [vel_gcrs_coords_i.cartesian.xyz]
    latitudes = [lla_i[0]]
    longitudes = [lla_i[1]]
    altitudes = [lla_i[2]]
    Bx_IGRF = [Bi_f[0]]
    By_IGRF = [Bi_f[1]]
    Bz_IGRF = [Bi_f[2]]
    Bx_IGRF_n = [Bi_fn[0]]
    By_IGRF_n = [Bi_fn[1]]
    Bz_IGRF_n = [Bi_fn[2]]
    vsun_x = [sunvectori[0]]
    vsun_y = [sunvectori[1]]
    vsun_z = [sunvectori[2]]
    vsun_x_n = [sunvectorin[0]]
    vsun_y_n = [sunvectorin[1]]
    vsun_z_n = [sunvectorin[2]]
    Bx_ECI_n = [Bi_ECI[0]]
    By_ECI_n = [Bi_ECI[1]]
    Bz_ECI_n = [Bi_ECI[2]]
    Bx_ECI = [Bi_ECI_s[0]]
    By_ECI = [Bi_ECI_s[1]]
    Bz_ECI = [Bi_ECI_s[2]]
    
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
        
        pos_itrs_coords, vel_itrs_coords  = functions_01.teme_to_itrs(position, velocity, start_time)
        pos_gcrs_coords = functions_01.itrs_to_gcrs(pos_itrs_coords, start_time)
        vel_gcrs_coords = functions_01.itrs_to_gcrs(vel_itrs_coords, start_time)

        lla = functions_01.eci2lla(position,current_time_gps)
        
        # Definir el tiempo en UTC
        utc_time = Time(current_time, scale="utc")
        # Convertir a UT1
        ut1_time = utc_time.ut1.value
        
        jd2000 = functions_01.datetime_to_jd2000(ut1_time)
        print(current_time)
        sunvector_n = functions_01.sun_vector(jd2000)
        sunvector = sunvector_n / np.linalg.norm(sunvector_n)
        
        #Obtener fuerzas magneticas de la Tierra
        jd2000_UTC = functions_01.datetime_to_jd2000(utc_time.value)
        Bm_PD = pyIGRF.igrf_value(lla[0],lla[1],lla[2]/1000, current_time.year)
        B_fn = np.array([Bm_PD[3], Bm_PD[4], Bm_PD[5]])
        B_ECEF = functions_01.NED_2_ECEF(lla[0], lla[1], B_fn)
        B_ECI = functions_01.ECEF_2_ECI(jd2000_UTC, B_ECEF)
        B_f = B_fn / np.linalg.norm(B_fn) 
        B_ECI_s = B_ECI / np.linalg.norm(B_ECI)
        
        t_aux.append(t0)
        positions.append(position)
        velocities.append(velocity)
        latitudes.append(lla[0])
        longitudes.append(lla[1])
        altitudes.append(lla[2])
        Bx_IGRF_n.append(B_fn[0])
        By_IGRF_n.append(B_fn[1])
        Bz_IGRF_n.append(B_fn[2])
        Bx_ECI_n.append(B_ECI[0])
        By_ECI_n.append(B_ECI[1])
        Bz_ECI_n.append(B_ECI[2])
        Bx_IGRF.append(B_f[0])
        By_IGRF.append(B_f[1])
        Bz_IGRF.append(B_f[2])
        Bx_ECI.append(B_ECI_s[0])
        By_ECI.append(B_ECI_s[1])
        Bz_ECI.append(B_ECI_s[2])
        vsun_x.append(sunvector[0])
        vsun_y.append(sunvector[1])
        vsun_z.append(sunvector[2])
        vsun_x_n.append(sunvector_n[0])
        vsun_y_n.append(sunvector_n[1])
        vsun_z_n.append(sunvector_n[2])     
        pos_itrs.append(pos_itrs_coords.cartesian.xyz)
        vel_itrs.append(vel_itrs_coords.cartesian.xyz)
        pos_gcrs.append(pos_gcrs_coords.cartesian.xyz)        
        vel_gcrs.append(vel_gcrs_coords.cartesian.xyz)        

         
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
    vsun_y = np.array(vsun_y)
    vsun_z = np.array(vsun_z)
    Bx_IGRF_n = np.array(Bx_IGRF_n)
    By_IGRF_n = np.array(By_IGRF_n)
    Bz_IGRF_n = np.array(Bz_IGRF_n)
    Bx_ECI_n = np.array(Bx_ECI_n)
    By_ECI_n = np.array(By_ECI_n)
    Bz_ECI_n = np.array(Bz_ECI_n)
    Bx_ECI = np.array(Bx_ECI)
    By_ECI = np.array(By_ECI)
    Bz_ECI = np.array(Bz_ECI)
    vsun_x_n = np.array(vsun_x_n)
    vsun_y_n = np.array(vsun_y_n)
    vsun_z_n = np.array(vsun_z_n)
    pos_itrs = np.array(pos_itrs)
    vel_itrs = np.array(vel_itrs)
    pos_gcrs = np.array(pos_gcrs)
    vel_gcrs = np.array(vel_gcrs)
    
    B_IGRF_n = np.transpose(np.vstack((Bx_IGRF_n,By_IGRF_n,Bz_IGRF_n)))
    b_norm = []
    positions_norm = []
    
    for i in range(len(Bx_IGRF_n)):
        norm = np.linalg.norm(B_IGRF_n[i,:])
        norm_p = np.linalg.norm(positions[i,:])
        b_norm.append(norm)
        positions_norm.append(norm_p)
    
    fig0, axes0 = plt.subplots(nrows=1, ncols=2, figsize=(16, 4))

    axes0[0].plot(t_aux, Bx_IGRF, label='Bx norm NED')
    axes0[0].plot(t_aux, By_IGRF, label='By norm NED')
    axes0[0].plot(t_aux, Bz_IGRF, label='Bz norm NED')
    axes0[0].set_xlabel('Tiempo [s]')
    axes0[0].set_ylabel('Fuerza magnetica [-]')
    axes0[0].legend()
    axes0[0].set_title('fuerza magnetica en NED')
    axes0[0].grid()
    # axes0[0].set_ylim(-1, 1)  # Ajusta los límites en el eje Y

    axes0[1].plot(t_aux, positions[:,0], label='posicion del satélite en x')
    axes0[1].plot(t_aux, positions[:,1], label='posicion del satélite en y')
    axes0[1].plot(t_aux, positions[:,2], label='posicion del satélite en z')
    axes0[1].set_xlabel('Tiempo [s]')
    axes0[1].set_ylabel('posición [km]')
    axes0[1].legend()
    axes0[1].set_title('posiciones del satélite en TEME')
    axes0[1].grid()
    plt.tight_layout()
    plt.show()
    
    fig0, axes0 = plt.subplots(nrows=1, ncols=2, figsize=(16, 4))

    axes0[0].plot(t_aux, Bx_IGRF_n, label='Bx NED')
    axes0[0].plot(t_aux, By_IGRF_n, label='By NED')
    axes0[0].plot(t_aux, Bz_IGRF_n, label='Bz NED')
    axes0[0].plot(t_aux,b_norm, label='norma B NED')
    axes0[0].set_xlabel('Tiempo [s]')
    axes0[0].set_ylabel('Fuerza magnetica [nT]')
    axes0[0].legend()
    axes0[0].set_title('fuerza magnetica en NED')
    axes0[0].grid()
    # axes0[0].set_ylim(-1, 1)  # Ajusta los límites en el eje Y

    axes0[1].plot(t_aux, positions[:,0], label='posicion del satélite en x')
    axes0[1].plot(t_aux, positions[:,1], label='posicion del satélite en y')
    axes0[1].plot(t_aux, positions[:,2], label='posicion del satélite en z')
    axes0[1].plot(t_aux, positions_norm, label='norma de la posicion del satélite')
    axes0[1].set_xlabel('Tiempo [s]')
    axes0[1].set_ylabel('posición [km]')
    axes0[1].legend()
    axes0[1].set_title('posiciones del satélite en TEME')
    axes0[1].grid()
    plt.tight_layout()
    plt.show()
    
    fig0, axes0 = plt.subplots(nrows=1, ncols=2, figsize=(16, 4))

    axes0[0].plot(t_aux,b_norm, label='norma B NED')
    axes0[0].set_xlabel('Tiempo [s]')
    axes0[0].set_ylabel('Fuerza magnetica [nT]')
    axes0[0].legend()
    axes0[0].set_title('fuerza magnetica en NED')
    axes0[0].grid()
    # axes0[0].set_ylim(-1, 1)  # Ajusta los límites en el eje Y

    axes0[1].plot(t_aux, positions_norm, label='norma de la posicion del satélite')
    axes0[1].set_xlabel('Tiempo [s]')
    axes0[1].set_ylabel('posición [km]')
    axes0[1].legend()
    axes0[1].set_title('posiciones del satélite en TEME')
    axes0[1].grid()
    plt.tight_layout()
    plt.show()
    
    fig0, axes0 = plt.subplots(nrows=1, ncols=2, figsize=(16, 4))

    axes0[0].plot(t_aux, positions[:,0], label='posicion del satélite en x')
    axes0[0].plot(t_aux, positions[:,1], label='posicion del satélite en y')
    axes0[0].plot(t_aux, positions[:,2], label='posicion del satélite en z')
    axes0[0].set_xlabel('Tiempo [s]')
    axes0[0].set_ylabel('posición [km]')
    axes0[0].legend()
    axes0[0].set_title('posiciones del satélite en TEME')
    axes0[0].grid()
    # axes0[0].set_ylim(-1, 1)  # Ajusta los límites en el eje Y

    axes0[1].plot(t_aux, pos_gcrs[:,0], label='posicion del satélite en x')
    axes0[1].plot(t_aux, pos_gcrs[:,1], label='posicion del satélite en y')
    axes0[1].plot(t_aux, pos_gcrs[:,2], label='posicion del satélite en z')
    axes0[1].set_xlabel('Tiempo [s]')
    axes0[1].set_ylabel('posición [km]')
    axes0[1].legend()
    axes0[1].set_title('posiciones del satélite en ECI')
    axes0[1].grid()
    plt.tight_layout()
    plt.show()
    
    fig0, axes0 = plt.subplots(nrows=1, ncols=2, figsize=(16, 4))

    axes0[0].plot(t_aux, velocities[:,0], label='velocidad del satélite en x')
    axes0[0].plot(t_aux, velocities[:,1], label='velocidad del satélite en y')
    axes0[0].plot(t_aux, velocities[:,2], label='velocidad del satélite en z')
    axes0[0].set_xlabel('Tiempo [s]')
    axes0[0].set_ylabel('velocidad [km/s]')
    axes0[0].legend()
    axes0[0].set_title('velocidades del satélite en TEME')
    axes0[0].grid()
    # axes0[0].set_ylim(-1, 1)  # Ajusta los límites en el eje Y

    axes0[1].plot(t_aux, vel_gcrs[:,0], label='velocidad del satélite en x')
    axes0[1].plot(t_aux, vel_gcrs[:,1], label='velocidad del satélite en y')
    axes0[1].plot(t_aux, vel_gcrs[:,2], label='velocidad del satélite en z')
    axes0[1].set_xlabel('Tiempo [s]')
    axes0[1].set_ylabel('velocidad [km/s]')
    axes0[1].legend()
    axes0[1].set_title('velocidades del satélite en ECI')
    axes0[1].grid()
    plt.tight_layout()
    plt.show()   
    
    
    fig0, axes0 = plt.subplots(nrows=1, ncols=2, figsize=(16, 4))

    axes0[0].plot(t_aux, Bx_IGRF_n, label='Bx NED')
    axes0[0].plot(t_aux, By_IGRF_n, label='By NED')
    axes0[0].plot(t_aux, Bz_IGRF_n, label='Bz NED')
    axes0[0].set_xlabel('Tiempo [s]')
    axes0[0].set_ylabel('Fuerza magnetica [nT]')
    axes0[0].legend()
    axes0[0].set_title('fuerza magnetica en NED')
    axes0[0].grid()
    # axes0[0].set_ylim(-1, 1)  # Ajusta los límites en el eje Y

    axes0[1].plot(t_aux, Bx_ECI_n, label='Bx ECI')
    axes0[1].plot(t_aux, By_ECI_n, label='By ECI')
    axes0[1].plot(t_aux, Bz_ECI_n, label='Bz ECI')
    axes0[1].set_xlabel('Tiempo [s]')
    axes0[1].set_ylabel('Fuerza magnetica [nT]')
    axes0[1].legend()
    axes0[1].set_title('fuerza magnetica en ECI')
    axes0[1].grid()
    plt.tight_layout()
    plt.show()
    
    
    # #%%
    # fig0, ax = plt.subplots(figsize=(10,6))  # Crea un solo set de ejes
    
    # # Graficar los tres conjuntos de datos en la misma gráfica
    # ax.plot(t_aux, positions[:,0], label='x')
    # ax.plot(t_aux, positions[:,1], label='y')
    # ax.plot(t_aux, positions[:,2], label='z')

    # # Configurar etiquetas, leyenda y grid
    # ax.set_xlabel('Tiempo [s]', fontsize=18)
    # ax.set_ylabel('Posición [km]', fontsize=18)
    # ax.legend(fontsize=18)
    # ax.grid()
    
    # # Ajustar límites del eje X
    # # ax.set_xlim(0, 30000)
    
    # # Ajustar el tamaño de las etiquetas de los ticks
    # ax.tick_params(axis='both', which='major', labelsize=18)
    
    # plt.tight_layout()
    
    # # Guardar la gráfica como archivo SVG
    # plt.savefig('pos.pdf', format='pdf')
    
    # # Mostrar la gráfica
    # plt.show()
    
    
    # fig0, ax = plt.subplots(figsize=(10,6))  # Crea un solo set de ejes
    
    # # Graficar los tres conjuntos de datos en la misma gráfica
    # ax.plot(t_aux, velocities[:,0], label='x')
    # ax.plot(t_aux, velocities[:,1], label='y')
    # ax.plot(t_aux, velocities[:,2], label='z')

    # # Configurar etiquetas, leyenda y grid
    # ax.set_xlabel('Tiempo [s]', fontsize=18)
    # ax.set_ylabel('Velocidad [km/s]', fontsize=18)
    # ax.legend(fontsize=18)
    # ax.grid()
    
    # # Ajustar límites del eje X
    # # ax.set_xlim(0, 30000)
    
    # # Ajustar el tamaño de las etiquetas de los ticks
    # ax.tick_params(axis='both', which='major', labelsize=18)
    
    # plt.tight_layout()
    
    # # Guardar la gráfica como archivo SVG
    # plt.savefig('vel.pdf', format='pdf')
    
    # # Mostrar la gráfica
    # plt.show()
    
    #%%
    
    fig0, ax = plt.subplots(figsize=(10,6))  # Crea un solo set de ejes
    
    # Graficar los tres conjuntos de datos en la misma gráfica
    ax.plot(t_aux, vsun_x_n, label='x')
    ax.plot(t_aux, vsun_y_n, label='y')
    ax.plot(t_aux, vsun_z_n, label='z')

    # Configurar etiquetas, leyenda y grid
    ax.set_xlabel('Tiempo [s]', fontsize=18)
    ax.set_ylabel('Vector sol [-]', fontsize=18)
    ax.legend(fontsize=18)
    ax.grid()
    
    # Ajustar límites del eje X
    # ax.set_xlim(0, 30000)
    
    # Ajustar el tamaño de las etiquetas de los ticks
    ax.tick_params(axis='both', which='major', labelsize=18)
    
    plt.tight_layout()
    
    # Guardar la gráfica como archivo SVG
    # plt.savefig('ss.pdf', format='pdf')
    
    # Mostrar la gráfica
    plt.show()
    
    
    fig0, ax = plt.subplots(figsize=(10,6))  # Crea un solo set de ejes
    
    # Graficar los tres conjuntos de datos en la misma gráfica
    ax.plot(t_aux,Bx_IGRF_n, label='x')
    ax.plot(t_aux, By_IGRF_n, label='y')
    ax.plot(t_aux, Bz_IGRF_n, label='z')

    # Configurar etiquetas, leyenda y grid
    ax.set_xlabel('Tiempo [s]', fontsize=18)
    ax.set_ylabel('Fuerza magnetica [nT]', fontsize=18)
    ax.legend(fontsize=18)
    ax.grid()
    
    # Ajustar límites del eje X
    # ax.set_xlim(0, 30000)
    
    # Ajustar el tamaño de las etiquetas de los ticks
    ax.tick_params(axis='both', which='major', labelsize=18)
    
    plt.tight_layout()
    
    # Guardar la gráfica como archivo SVG
    # plt.savefig('bb.pdf', format='pdf')
    
    # Mostrar la gráfica
    plt.show()
    
    
    
    #%% Guardar datos en un archivo csv
    
    # # Nombre del archivo
    # # archivo = "vectores_400k_1s.csv"
    # archivo = "vectores_400k.csv"

    # # Abrir el archivo en modo escritura
    # with open(archivo, 'w') as f:
    #     # Escribir los encabezados
    #     f.write("t_aux, position_x,position_y,position_z, velocity_x, velocity_y, velocity_z, latitudes, longitudes, altitudes, Bx_IGRF, By_IGRF, Bz_IGRF, vsun_x, vsun_y, vsun_z, Bx_IGRF_n, By_IGRF_n, Bz_IGRF_n, vsun_x_n, vsun_y_n, vsun_z_n\n")
    
    #     # Escribir los datos en filas
    #     for i in range(len(t_aux)):
    #         f.write("{},{},{},{},{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(
    #             t_aux[i], pos_gcrs[i,0], pos_gcrs[i,1],pos_gcrs[i,2],vel_gcrs[i,0],vel_gcrs[i,1],vel_gcrs[i,2],
    #             latitudes[i],longitudes[i], altitudes[i], Bx_ECI[i], By_ECI[i],
    #             Bz_ECI[i], vsun_x[i], vsun_y[i], vsun_z[i], Bx_ECI_n[i], By_ECI_n[i],
    #             Bz_ECI_n[i], vsun_x_n[i], vsun_y_n[i], vsun_z_n[i]
    #         ))
    
    # print("Vectores guardados en el archivo:", archivo)
    
    # import pandas as pd

    # # Nombre del archivo CSV
    # archivo_csv = "vectores.csv"
    
    # # Leer el archivo CSV en un DataFrame de pandas
    # df = pd.read_csv(archivo_csv)
    
    # # Convertir el DataFrame a un array de NumPy
    # array_datos = df.to_numpy()