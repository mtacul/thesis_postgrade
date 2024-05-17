# -*- coding: utf-8 -*-
"""
Created on Fri May 10 16:07:10 2024

@author: nachi
"""

from skyfield.positionlib import Geocentric
import numpy as np
from datetime import datetime
import math
from scipy.signal import butter, lfilter
import control as ctrl

def eci2lla(posicion, fecha):
    from skyfield.api import Distance, load, utc, wgs84
    ts = load.timescale()
    fecha = fecha.replace(tzinfo=utc)
    t = ts.utc(fecha)
    d = [Distance(m=i).au for i in (posicion[0]*1000, posicion[1]*1000, posicion[2]*1000)]
    p = Geocentric(d,t=t)
    g = wgs84.subpoint(p)
    latitud = g.latitude.degrees
    longitud = g.longitude.degrees
    altitud = g.elevation.m
    return latitud, longitud, altitud

#%%inversa de un cuaternion

def inv_q(q):
    inv_q = np.array([-q[0],-q[1],-q[2],q[3]])
    return inv_q

#%% Para vector sol

def datetime_to_jd2000(fecha):
    t1 = (fecha - datetime(2000, 1, 1, 12, 0, 0)).total_seconds()
    t2 = 86400  # Número de segundos en un día
    jd2000 = t1 / t2
    return jd2000

def sun_vector(jd2000):
    M_sun = 357.528 + 0.9856003*jd2000
    M_sun_rad = M_sun * np.pi/180
    lambda_sun = 280.461 + 0.9856474*jd2000 + 1.915*np.sin(M_sun_rad)+0.020*np.sin(2*M_sun_rad)
    lambda_sun_rad = lambda_sun * np.pi/180
    epsilon_sun = 23.4393 - 0.0000004*jd2000
    epsilon_sun_rad = epsilon_sun * np.pi/180
    X_sun = np.cos(lambda_sun_rad)
    Y_sun = np.cos(epsilon_sun_rad)*np.sin(lambda_sun_rad)
    Z_sun = np.sin(epsilon_sun_rad)*np.sin(lambda_sun_rad)
    return X_sun, Y_sun, Z_sun

#%% TRIAD solution

def TRIAD(V1,V2,W1,W2):
    r1 = V1
    r2 = np.cross(V1,V2) / np.linalg.norm(np.cross(V1,V2))
    r3 = np.cross(r1,r2)
    M_obs = np.array([r1,r2,r3])
    s1 = W1
    s2 = np.cross(W1,W2) / np.linalg.norm(np.cross(W1,W2))
    s3 = np.cross(s1,s2)
    M_ref = np.array([s1,s2,s3])
    
    A = np.dot(M_ref,np.transpose(M_obs))
    return A

#%% de cuaternion a angulos de euler
def quaternion_to_euler(q):
    # Extracción de los componentes del cuaternión
    x, y, z, w = q

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x**2 + y**2)
    roll_x = np.arctan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)  # Clamp t2 to avoid numerical errors
    pitch_y = np.arcsin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y**2 + z**2)
    yaw_z = np.arctan2(t3, t4)
         
    # Convierte los ángulos a grados 
    roll_deg = np.degrees(roll_x)
    pitch_deg = np.degrees(pitch_y)
    yaw_deg = np.degrees(yaw_z)

    return roll_deg, pitch_deg, yaw_deg

def quat_mult(dqk,qk_priori):
    
    dqk_n = dqk 
    

# Realizar la multiplicación de cuaterniones
    result = np.array([
    dqk_n[3]*qk_priori[0] + dqk_n[0]*qk_priori[3] + dqk_n[1]*qk_priori[2] - dqk_n[2]*qk_priori[1],
    dqk_n[3]*qk_priori[1] + dqk_n[1]*qk_priori[3] + dqk_n[2]*qk_priori[0] - dqk_n[0]*qk_priori[2],  # Componente j
    dqk_n[3]*qk_priori[2] + dqk_n[2]*qk_priori[3] + dqk_n[0]*qk_priori[1] - dqk_n[1]*qk_priori[0],  # Componente k
    dqk_n[3]*qk_priori[3] - dqk_n[0]*qk_priori[0] - dqk_n[1]*qk_priori[1] - dqk_n[2]*qk_priori[2]   # Componente escalar
    ])
    return result