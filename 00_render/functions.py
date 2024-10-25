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


#%%Funciones para imposicion de ruido en sensores

def simulate_magnetometer_reading(B_eci, ruido):
    # np.random.seed(42)  # Puedes cambiar el número 42 a cualquier otro número entero

    # Simular el ruido gaussiano
    noise = np.random.normal(0, ruido, 1)
        
    # Simular la medición del magnetómetro con ruido
    measurement = B_eci + noise


    return measurement

# Obtener la desviacion estandar del sun sensor
def sigma_sensor(acc):
    sigma = acc/(2*3)
    return sigma

# Funcion para generar realismo del sun sensor
def simulate_sunsensor_reading(vsun,sigma):
    # np.random.seed(42)  # Puedes cambiar el número 42 a cualquier otro número entero

    sigma_rad = sigma*np.pi/180
        
    # Simulación de la medición con error
    error = np.random.normal(0, sigma_rad, 1)  # Genera un error aleatorio dentro de la precisión del sensor
        
    measured_vsun = vsun + error

    return measured_vsun

# Funcion para generar realismo del giroscopio
def simulate_gyros_reading(w,ruido,bias):
    # np.random.seed(42)  # Puedes cambiar el número 42 a cualquier otro número entero

    #aplicar el ruido del sensor
    noise = np.random.normal(0, ruido, 1)
        
    #Simular la medicion del giroscopio
    measurement = w + noise + bias
        
    return measurement

#%% matrices A y B control PD lineal

def A_PD(I_x,I_y,I_z,w0_O, w0,w1,w2):
    A1 = np.array([0, 0.5*w2, -0.5*w1, 0.5, 0,0])
    A2 = np.array([-0.5*w2,0,0.5*w0,0,0.5,0])
    A3 = np.array([0.5*w1,-0.5*w0,0,0,0,0.5])
    A4 = np.array([6*w0_O**2*(I_x-I_y), 0, 0, 0, w2*(I_y-I_z)/I_x, w1*(I_y-I_z)/I_x])
    A5 = np.array([0, 6*w0_O**2*(I_z-I_y), 0, w2*(I_x-I_z)/I_y,0, (w0+w0_O)*(I_x-I_z)/I_y + I_y*w0_O])
    A6 = np.array([0, 0, 0, w1*(I_y-I_x)/I_z, (w0+w0_O)*(I_y-I_x)/I_z - I_z*w0_O, 0])
    
    A_k = np.array([A1,A2,A3,A4,A5,A6])
    
    return A_k    

def B_PD(I_x,I_y,I_z,B_magnet):
    b_norm = np.linalg.norm(B_magnet)
    B123 = np.zeros((3,3))
    B4 = np.array([(-(B_magnet[2]**2)-B_magnet[1]**2)/(b_norm*I_x), B_magnet[1]*B_magnet[0]/(b_norm*I_x), B_magnet[2]*B_magnet[0]/(b_norm*I_x)])
    B5 = np.array([B_magnet[0]*B_magnet[1]/(b_norm*I_y), (-B_magnet[2]**2-B_magnet[0]**2)/(b_norm*I_y), B_magnet[2]*B_magnet[1]/(b_norm*I_y)])
    B6 = np.array([B_magnet[0]*B_magnet[2]/(b_norm*I_z), B_magnet[1]*B_magnet[2]/(b_norm*I_z), (-B_magnet[1]**2-B_magnet[0]**2)/(b_norm*I_z)])
    
    B_k = np.vstack((B123,B4,B5,B6))
    #B_k = np.array([B123,B4,B5,B6])

    return B_k

def rk4_step_PD(dynamics, x, A, B, u, h):
    k1 = h * dynamics(A, x, B, u)
    k2 = h * dynamics(A, x + 0.5 * k1, B, u)
    k3 = h * dynamics(A, x + 0.5 * k2, B, u)
    k4 = h * dynamics(A, x + k3, B, u)
        
    # Update components of q
    q0_new = x[0] + (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6
    q1_new = x[1] + (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6
    q2_new = x[2] + (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) / 6
    
    q_new_real = np.array([q0_new, q1_new, q2_new])

    # Update components of w
    w0_new = x[3] + (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3]) / 6
    w1_new = x[4] + (k1[4] + 2 * k2[4] + 2 * k3[4] + k4[4]) / 6
    w2_new = x[5] + (k1[5] + 2 * k2[5] + 2 * k3[5] + k4[5]) / 6
    w_new = np.array([w0_new, w1_new, w2_new])

    return q_new_real, w_new

# funcion de la ecuacicon xDot = Ax - Bu 
def dynamics(A, x, B, u):
    return np.dot(A, x) - np.dot(B, u)

def K(Kp_x, Kp_y, Kp_z, Kd_x, Kd_y, Kd_z):
    K_gain = np.hstack((np.diag([Kp_x,Kp_y,Kp_z]), np.diag([Kd_x,Kd_y,Kd_z])))
    return K_gain

def quaternion_to_dcm(q):
    x, y, z, w = q
    dcm = np.array([
        [w**2 + x*2 + 2*y**2 + 2*z**2, 2*x*y + 2*w*z, 2*x*z - 2*w*y],
        [2*x*y - 2*w*z, w**2-x**2+y**2+z**2, 2*y*z + 2*w*x],
        [2*x*z + 2*w*y, 2*y*z - 2*w*x, w**2-x**2-y**2+z**2]
    ])
    return dcm

def high_pass_filter(signal, cutoff_freq, sample_rate, order=4):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    filtered_signal = lfilter(b, a, signal)
    return filtered_signal

def low_pass_filter(signal, cutoff_freq, sample_rate, order=4):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_signal = lfilter(b, a, signal)
    return filtered_signal


#%% MODELOS

# Modelo lineal continuo
def mod_lineal_cont(x,u,deltat,h,A,B):
    
    for j in range(int(deltat/h)):
        q_rot,w_new = rk4_step_PD(dynamics, x, A, B, u, h)
        if  1-q_rot[0]**2-q_rot[1]**2-q_rot[2]**2 < 0:
            q_rot = q_rot / np.linalg.norm(q_rot)
            x_new = np.hstack((np.transpose(q_rot), np.transpose(w_new)))
            q3s_rot = 0

        else:
            x_new = np.hstack((np.transpose(q_rot), np.transpose(w_new)))
            q3s_rot = np.sqrt(1-q_rot[0]**2-q_rot[1]**2-q_rot[2]**2)
                
    return x_new, q3s_rot

#%% matrices posibles

def H_EKF(Bref,Bmagn):
    I = np.identity(3)
    ccc = np.zeros((3,3))
    H11 = 2*I - 2*(np.dot(Bmagn,np.transpose(Bref))) / (np.linalg.norm(Bmagn)*np.linalg.norm(Bref))
    H111 = np.hstack((H11,ccc))
    
    H1 = np.array([0, 0, 0, 1, 0, 0])
    H2 = np.array([0, 0, 0, 0, 1, 0])
    H3 = np.array([0, 0, 0, 0, 0, 1])
    
    H222 = np.array([H1,H2,H3])
    
    H_mat = np.vstack((H111 , H222))
    
    return H_mat #JACOBIANO DEL MODELO DEL SENSOR (POR AHORA SOLO MAGNETOMETRO)

# Matriz H, que representa las mediciones derivadas respecto al vector estado x (q0,q1,q2,biasx,biasy,biasz)
def H_k_bar(b0,b1,b2,s0,s1,s2):
    
    b = np.array([b0,b1,b2])
    s = np.array([s0,s1,s2])

    skew_b_1 = np.array([0, -b2, b1])
    skew_b_2 = np.array([b2,0,-b0])
    skew_b_3 = np.array([-b1,b0,0])
    skew_b = np.array([skew_b_1,skew_b_2,skew_b_3])
    
    skew_s_1 = np.array([0, -s2, s1])
    skew_s_2 = np.array([s2,0,-s0])
    skew_s_3 = np.array([-s1,s0,0])
    skew_s = np.array([skew_s_1,skew_s_2,skew_s_3])
    
    H_11 = 2*skew_b
    H_12 = np.zeros((3,3))
    H_21 = 2*skew_s
    H_22 = H_12
    
    H_1 = np.hstack((H_11, H_12))
    H_2 = np.hstack((H_21,H_22))
    
    H = np.vstack((H_1,H_2))
    
    return H

def A_B(I_x,I_y,I_z,w0_O,w0,w1,w2,deltat,h,B_ref,B_body_ctrl):
    
    A =A_PD(I_x,I_y,I_z,w0_O, w0,w1,w2)
    B = B_PD(I_x,I_y,I_z,B_body_ctrl)
    
    # Define an identity matrix for C and a zero matrix for D to complete state-space model
    # C = np.eye(6)  # Assuming a 6x6 identity matrix for C
    C = H_EKF(B_ref, B_body_ctrl)
    D = np.zeros((6, 3))  # Assuming D has the same number of rows as A and the same number of columns as B

    # Create the continuous state-space model
    sys_continuous = ctrl.StateSpace(A, B, C, D)

    # Discretize the system
    sys_discrete = ctrl.c2d(sys_continuous, deltat*h, method='zoh')

    # Extract the discretized A and B matrices
    A_discrete = sys_discrete.A
    B_discrete = sys_discrete.B
    C_discrete = sys_discrete.C
    
    return A,B,C,A_discrete,B_discrete,C_discrete

def A_B_bar(I_x,I_y,I_z,w0_O,w0,w1,w2,deltat,h,B_body_ctrl,s0,s1,s2):
    
    A =A_PD(I_x,I_y,I_z,w0_O, w0,w1,w2)
    B = B_PD(I_x,I_y,I_z,B_body_ctrl)
    
    # Define an identity matrix for C and a zero matrix for D to complete state-space model
    # C = np.eye(6)  # Assuming a 6x6 identity matrix for C
    C = H_k_bar(B_body_ctrl[0], B_body_ctrl[1],B_body_ctrl[2],s0,s1,s2)
    D = np.zeros((6, 3))  # Assuming D has the same number of rows as A and the same number of columns as B

    # Create the continuous state-space model
    sys_continuous = ctrl.StateSpace(A, B, C, D)

    # Discretize the system
    sys_discrete = ctrl.c2d(sys_continuous, deltat*h, method='zoh')

    # Extract the discretized A and B matrices
    A_discrete = sys_discrete.A
    B_discrete = sys_discrete.B
    C_discrete = sys_discrete.C
    
    return A,B,C,A_discrete,B_discrete,C_discrete

#%% Modelo lineal discreto

def mod_lineal_disc(x,u,deltat, h,A_discrete,B_discrete):
        
    for i in range(int(1/h)):
        x_k_1 = np.dot(A_discrete,x) - np.dot(B_discrete,u)
        q_rot = x_k_1[0:3]
        w_new = x_k_1[3:6]
    
        if  1-q_rot[0]**2-q_rot[1]**2-q_rot[2]**2 < 0:
            q_rot = q_rot / np.linalg.norm(q_rot)
            x_new = np.hstack((np.transpose(q_rot), np.transpose(w_new)))
            q3s_rot = 0
    
        else:
            x_new = np.hstack((np.transpose(q_rot), np.transpose(w_new)))
            q3s_rot = np.sqrt(1-q_rot[0]**2-q_rot[1]**2-q_rot[2]**2)
        
    return x_new, q3s_rot



#%% Modelo no lineal continuo

#Funciones ecuaciones no lineales, en conjunto con su resolucion en RK4

def f1_K(t, q0, q1, q2,q3, w0, w1, w2): #q0_dot
    return 0.5*(q1*w2 - q2*w1 + w0*q3)

def f2_K(t, q0, q1, q2,q3, w0, w1, w2): #q1_dot
    return 0.5*(-q0*w2 + q2*w0 + w1*q3)

def f3_K(t, q0, q1, q2,q3, w0, w1, w2): #q2_dot
    return 0.5*(q0*w1 - q1*w0 + w2*q3)

def f4_K(t, q0, q1, q2,q3, w0, w1, w2): #q3_dot
    return 0.5*(-q0*w0 - q1*w1 - w2*q2)

def f5_K(t, q0, q1, q2,q3, w0, w1, w2,w0_o,tau_x_ctrl,tau_x_per,I_x,I_y,I_z):#w1_dot
    part_1_w0 = w1 + w0_o*(2*(q0*q1 - q2*q3))
    part_2_w0 = w2 + w0_o*(2*(q0*q2 + q1*q3))
    part_3_w0 = w0_o*(2*q3*0.5*(-q0*w0 - q1*w1 - w2*q2)+2*q0*0.5*(q1*w2 - q2*w1 + w0*q3)-2*q1*0.5*(-q0*w2 + q2*w0 + w1*q3)-2*q2*0.5*(q0*w1 - q1*w0 + w2*q3))
    part_4_w0 = tau_x_ctrl/I_x
    part_5_w0 = tau_x_per/I_x
    return part_1_w0*part_2_w0*(I_y-I_z)/I_x - part_3_w0 + part_4_w0 + part_5_w0

def f6_K(t, q0, q1, q2,q3, w0, w1, w2,w0_o,tau_y_ctrl,tau_y_per,I_x,I_y,I_z): #w2_dot
    part_1_w1 = w0 + w0_o*(q3**2+q0**2-q1**2-q2**2)
    part_2_w1 = w2 + w0_o*(2*(q0*q2 + q1*q3))
    part_3_w1 = w0_o*(q0*q1*0.5*(q1*w2 - q2*w1 + w0*q3) + q0*q1*0.5*(-q0*w2 + q2*w0 + w1*q3)-q2*q3*0.5*(q0*w1 - q1*w0 + w2*q3)-q2*q3*0.5*(-q0*w0 - q1*w1 - w2*q2))
    part_4_w1 = tau_y_ctrl/I_y
    part_5_w1 = tau_y_per/I_y
    return part_1_w1*part_2_w1*(I_x-I_z)/I_y - part_3_w1 + part_4_w1 +part_5_w1

def f7_K(t, q0, q1, q2,q3, w0, w1, w2,w0_o,tau_z_ctrl,tau_z_per,I_x,I_y,I_z): #w3_dot
    part_1_w2 = w0 + w0_o*(q3**2+q0**2-q1**2-q2**2)
    part_2_w2 = w1 + w0_o*(2*(q0*q1 - q2*q3))
    part_3_w2 = w0_o*(q0*q2*0.5*(q1*w2 - q2*w1 + w0*q3) + q0*q2*0.5*(q0*w1 - q1*w0 + w2*q3)+ q1*q3*0.5*(-q0*w2 + q2*w0 + w1*q3)+ q1*q3*0.5*(-q0*w0 - q1*w1 - w2*q2))
    part_4_w2 = tau_z_ctrl/I_z
    part_5_w2 = tau_z_per/I_z
    return part_1_w2*part_2_w2*(I_x-I_y)/I_z - part_3_w2 + part_4_w2 + part_5_w2

def rk4_EKF_step(t, q0, q1, q2,q3, w0, w1, w2, h, w0_o,tau_x_ctrl,tau_x_per,tau_y_ctrl,tau_y_per,tau_z_ctrl,tau_z_per,I_x,I_y,I_z):
    #k1 = h * f1(x, y1, y2)
    k1_1 = h * f1_K(t, q0, q1, q2,q3, w0, w1, w2)
    k1_2 = h * f2_K(t, q0, q1, q2,q3, w0, w1, w2)
    k1_3 = h * f3_K(t, q0, q1, q2,q3, w0, w1, w2)
    k1_4 = h * f4_K(t, q0, q1, q2,q3, w0, w1, w2)
    k1_5 = h * f5_K(t, q0, q1, q2,q3, w0, w1, w2,w0_o, tau_x_ctrl,tau_x_per,I_x,I_y,I_z)
    k1_6 = h * f6_K(t, q0, q1, q2,q3, w0, w1, w2,w0_o, tau_y_ctrl,tau_y_per,I_x,I_y,I_z)
    k1_7 = h * f7_K(t, q0, q1, q2,q3, w0, w1, w2,w0_o, tau_z_ctrl,tau_z_per,I_x,I_y,I_z)
    
    k2_1 = h * f1_K(t + 0.5 * h, q0 + 0.5 * k1_1, q1 + 0.5 * k1_2, q2 + 0.5 * k1_3,
                  q3 + 0.5 * k1_4,w0 + 0.5 * k1_5, w1 + 0.5 * k1_6, w2 + 0.5 * k1_7)
    k2_2 = h * f2_K(t + 0.5 * h, q0 + 0.5 * k1_1, q1 + 0.5 * k1_2, q2 + 0.5 * k1_3,
                  q3 + 0.5 * k1_4,w0 + 0.5 * k1_5, w1 + 0.5 * k1_6, w2 + 0.5 * k1_7)
    k2_3 = h * f3_K(t + 0.5 * h, q0 + 0.5 * k1_1, q1 + 0.5 * k1_2, q2 + 0.5 * k1_3,
                  q3 + 0.5 * k1_4,w0 + 0.5 * k1_5, w1 + 0.5 * k1_6, w2 + 0.5 * k1_7)
    k2_4 = h * f4_K(t + 0.5 * h, q0 + 0.5 * k1_1, q1 + 0.5 * k1_2, q2 + 0.5 * k1_3,
                  q3 + 0.5 * k1_4,w0 + 0.5 * k1_5, w1 + 0.5 * k1_6, w2 + 0.5 * k1_7)
    k2_5 = h * f5_K(t + 0.5 * h, q0 + 0.5 * k1_1, q1 + 0.5 * k1_2, q2 + 0.5 * k1_3,
                  q3 + 0.5 * k1_4,w0 + 0.5 * k1_5, w1 + 0.5 * k1_6, w2 + 0.5 * k1_7,w0_o, tau_x_ctrl,tau_x_per,I_x,I_y,I_z)
    k2_6 = h * f6_K(t + 0.5 * h, q0 + 0.5 * k1_1, q1 + 0.5 * k1_2, q2 + 0.5 * k1_3,
                  q3 + 0.5 * k1_4,w0 + 0.5 * k1_5, w1 + 0.5 * k1_6, w2 + 0.5 * k1_7,w0_o, tau_y_ctrl,tau_y_per,I_x,I_y,I_z)
    k2_7 = h * f7_K(t + 0.5 * h, q0 + 0.5 * k1_1, q1 + 0.5 * k1_2, q2 + 0.5 * k1_3,
                  q3 + 0.5 * k1_4,w0 + 0.5 * k1_5, w1 + 0.5 * k1_6, w2 + 0.5 * k1_7,w0_o, tau_z_ctrl,tau_z_per,I_x,I_y,I_z)
    
    k3_1 = h * f1_K(t + 0.5 * h, q0 + 0.5 * k2_1, q1 + 0.5 * k2_2, q2 + 0.5 * k2_3,
                  q3 + 0.5 * k2_4, w0 + 0.5 * k2_5, w1 + 0.5 * k2_6, w2 + 0.5 * k2_7)
    k3_2 = h * f2_K(t + 0.5 * h, q0 + 0.5 * k2_1, q1 + 0.5 * k2_2, q2 + 0.5 * k2_3,
                  q3 + 0.5 * k2_4, w0 + 0.5 * k2_5, w1 + 0.5 * k2_6, w2 + 0.5 * k2_7)
    k3_3 = h * f3_K(t + 0.5 * h, q0 + 0.5 * k2_1, q1 + 0.5 * k2_2, q2 + 0.5 * k2_3,
                  q3 + 0.5 * k2_4, w0 + 0.5 * k2_5, w1 + 0.5 * k2_6, w2 + 0.5 * k2_7)
    k3_4 = h * f4_K(t + 0.5 * h, q0 + 0.5 * k2_1, q1 + 0.5 * k2_2, q2 + 0.5 * k2_3,
                  q3 + 0.5 * k2_4, w0 + 0.5 * k2_5, w1 + 0.5 * k2_6, w2 + 0.5 * k2_7)

    k3_5 = h * f5_K(t + 0.5 * h, q0 + 0.5 * k2_1, q1 + 0.5 * k2_2, q2 + 0.5 * k2_3,
                  q3 + 0.5 * k2_4, w0 + 0.5 * k2_5, w1 + 0.5 * k2_6, w2 + 0.5 * k2_7, w0_o, tau_x_ctrl,
                  tau_x_per,I_x,I_y,I_z)
    k3_6 = h * f6_K(t + 0.5 * h, q0 + 0.5 * k2_1, q1 + 0.5 * k2_2, q2 + 0.5 * k2_3,
                  q3 + 0.5 * k2_4, w0 + 0.5 * k2_5, w1 + 0.5 * k2_6, w2 + 0.5 * k2_7, w0_o, tau_y_ctrl,
                  tau_y_per,I_x,I_y,I_z)
    k3_7 = h * f7_K(t + 0.5 * h, q0 + 0.5 * k2_1, q1 + 0.5 * k2_2, q2 + 0.5 * k2_3,
                  q3 + 0.5 * k2_4, w0 + 0.5 * k2_5, w1 + 0.5 * k2_6, w2 + 0.5 * k2_7, w0_o, tau_z_ctrl,
                  tau_z_per,I_x,I_y,I_z)

    
    k4_1 = h * f1_K(t + 0.5 * h, q0 + 0.5 * k3_1, q1 + 0.5 * k3_2, q2 + 0.5 * k3_3,
                  q3 + 0.5 * k3_4, w0 + 0.5 * k3_5, w1 + 0.5 * k3_6, w2 + 0.5 * k3_7)
    k4_2 = h * f2_K(t + 0.5 * h, q0 + 0.5 * k3_1, q1 + 0.5 * k3_2, q2 + 0.5 * k3_3,
                  q3 + 0.5 * k3_4, w0 + 0.5 * k3_5, w1 + 0.5 * k3_6, w2 + 0.5 * k3_7)
    k4_3 = h * f3_K(t + 0.5 * h, q0 + 0.5 * k3_1, q1 + 0.5 * k3_2, q2 + 0.5 * k3_3,
                  q3 + 0.5 * k3_4, w0 + 0.5 * k3_5, w1 + 0.5 * k3_6, w2 + 0.5 * k3_7)
    k4_4 = h * f4_K(t + 0.5 * h, q0 + 0.5 * k3_1, q1 + 0.5 * k3_2, q2 + 0.5 * k3_3,
                  q3 + 0.5 * k3_4, w0 + 0.5 * k3_5, w1 + 0.5 * k3_6, w2 + 0.5 * k3_7)
    k4_5 = h * f5_K(t + 0.5 * h, q0 + 0.5 * k3_1, q1 + 0.5 * k3_2, q2 + 0.5 * k3_3,
                  q3 + 0.5 * k3_4, w0 + 0.5 * k3_5, w1 + 0.5 * k3_6, w2 + 0.5 * k3_7, w0_o, tau_x_ctrl,
                  tau_x_per,I_x,I_y,I_z)
    k4_6 = h * f6_K(t + 0.5 * h, q0 + 0.5 * k3_1, q1 + 0.5 * k3_2, q2 + 0.5 * k3_3,
                  q3 + 0.5 * k3_4, w0 + 0.5 * k3_5, w1 + 0.5 * k3_6, w2 + 0.5 * k3_7, w0_o, tau_y_ctrl,
                  tau_y_per,I_x,I_y,I_z)
    k4_7 = h * f7_K(t + 0.5 * h, q0 + 0.5 * k3_1, q1 + 0.5 * k3_2, q2 + 0.5 * k3_3,
                  q3 + 0.5 * k3_4, w0 + 0.5 * k3_5, w1 + 0.5 * k3_6, w2 + 0.5 * k3_7, w0_o, tau_z_ctrl,
                  tau_z_per,I_x,I_y,I_z)
   

    q0_new = q0 + (k1_1 + 2 * k2_1 + 2 * k3_1 + k4_1) / 6    
    q1_new = q1 + (k1_2 + 2 * k2_2 + 2 * k3_2 + k4_2) / 6
    q2_new = q2 + (k1_3 + 2 * k2_3 + 2 * k3_3 + k4_3) / 6
    q3_new = q3 + (k1_4 + 2 * k2_4 + 2 * k3_4 + k4_4) / 6

    q = [q0_new, q1_new, q2_new,q3_new]
    q = q / np.linalg.norm(q)
    
    w0_new = w0 + (k1_5 + 2 * k2_5 + 2 * k3_5 + k4_5) / 6
    w1_new = w1 + (k1_6 + 2 * k2_6 + 2 * k3_6 + k4_6) / 6
    w2_new = w2 + (k1_7 + 2 * k2_7 + 2 * k3_7 + k4_7) / 6
    w = [w0_new, w1_new, w2_new]
    return q, w


def mod_nolineal(x,u,deltat, b,h):
    
    w0_O = 0.00163
    tt = 0
    
    if  1-x[0]**2-x[1]**2-x[2]**2 < 0:
        x[0:3] = x[0:3] / np.linalg.norm(x[0:3])
        q3s_rot = 0

    else:
        q3s_rot = np.sqrt(1-x[0]**2-x[1]**2-x[2]**2)
        
    I_x = 0.037
    I_y = 0.036
    I_z = 0.006
    
    b_norm = np.linalg.norm(np.hstack((b[0],b[1],b[2])))
    tau_x_ctrl = ((b[0]*u[2]-b[2]*u[0])/b_norm)*b[2] - ((b[1]*u[0]-b[0]*u[1])/b_norm)*b[1]
    tau_y_ctrl = ((b[2]*u[1]-b[1]*u[2])/b_norm)*-b[2] + ((b[1]*u[0]-b[0]*u[1])/b_norm)*b[0]
    tau_z_ctrl = ((b[2]*u[1]-b[1]*u[2])/b_norm)*b[1] - ((b[0]*u[2]-b[2]*u[0])/b_norm)*b[0]

    for j in range(int(deltat/h)):
        t, q0, q1, q2, q3, w0, w1, w2 = tt, x[0], x[1], x[2], q3s_rot, x[3], x[4], x[5]
        q_rot, w_new = rk4_EKF_step(t, q0, q1, q2,q3, w0, w1, w2,h,w0_O, tau_x_ctrl,0,tau_y_ctrl,0,tau_z_ctrl,0,I_x,I_y,I_z)
    
    q_rot_trunc = q_rot[0:3]
    x_new = np.hstack((np.transpose(q_rot_trunc), np.transpose(w_new)))
    
    return x_new, q_rot[3]

#%% Filtro de kalman lineal y sus ecuaciones/matrices

# Matriz C o H, que representa las variables de medicion derivadas respecto al estado

def R_k(sigma_m, sigma_s):
    R_11 = np.eye(3) * sigma_m**2
    R_12 = np.zeros((3,3))
    R_21 = R_12
    R_22 = np.eye(3) * sigma_s**2
    
    R_1 = np.hstack((R_11, R_12))
    R_2 = np.hstack((R_21,R_22))
    
    R = np.vstack((R_1,R_2))
    
    return R

# Matriz de covarianza del ruido del proceso se subdivide en Q y G
def Q_k(sigma_w,sigma_bias,deltat):
    Q_11 = np.eye(3) * (sigma_w**2*deltat + 1/3 * sigma_bias**2*deltat**3)
    Q_12 = np.eye(3) * -(1/2 * sigma_bias**2 * deltat**2)
    Q_21 = Q_12
    Q_22 = np.eye(3) * (sigma_bias**2 * deltat)
    
    Q_1 = np.hstack((Q_11, Q_12))
    Q_2 = np.hstack((Q_21,Q_22))
    
    Q = np.vstack((Q_1,Q_2))
    
    return Q

def P_k_prior(F_k, P_ki, Q_k):
    P_k_priori = np.dot(np.dot(F_k,P_ki),np.transpose(F_k)) + Q_k
    return P_k_priori #MATRIZ P PRIORI CON P INICIAL DADO, DESPUES SE ACTUALIZA


def k_kalman(R_k, P_k_priori, H_mat):
    K_k_izq =  np.dot(P_k_priori,np.transpose(H_mat))
    K_k_der = np.linalg.inv(R_k + np.dot(np.dot(H_mat,P_k_priori),np.transpose(H_mat)))
    
    K_k = np.dot(K_k_izq,K_k_der)
    return K_k #GANANCIA DE KALMAN


def P_posteriori(K_k,H_k,P_k_priori):
   I = np.identity(6)
   P_k_pos = np.dot(I - np.dot(K_k,H_k),P_k_priori)
   return P_k_pos #SACAR MATRIZ P POSTERIORI ACTUALIZADA

# def kalman_lineal(A, B, x, u, b, s, P_ki, ruido, sigma):

def kalman_lineal(A, B, x, u, b_eci, b, P_ki, ruido, bias,deltat,w,hh):
# def kalman_lineal(A, B, C, x, u, b, b_eci, P_ki, ruido, ruido_ss,deltat,s):
    
    C = H_EKF(b_eci, b_eci)
    [x_priori,q3s_rot] = mod_lineal_disc(x, u, deltat, hh, A, B)
    # x_priori = np.dot(A,x) - np.dot(B,u)
    q_priori = np.hstack((x_priori[0:3], q3s_rot))
    # q_priori = np.hstack((x_priori[0:3], np.sqrt(1-np.linalg.norm(x_priori[0:3])**2)))
    # print("q priori obtenido por kalman:",q_priori,"\n")
    w_priori = x_priori[3:6]

    # aa = np.eye(6,6)
    # Q_ki = np.vstack((0.25*aa[0:3,:],0.01*aa[3:6,:]))
    
    Q_ki = Q_k(5e-3, 3e-4,deltat)
    
    P_k_priori = P_k_prior(A,P_ki,Q_ki)
    H_k = C
    # H_k = H_k_bar(b[0], b[1], b[2], s[0], s[1], s[2])
    # z_sensor = np.hstack((b[0],b[1],b[2], s[0], s[1], s[2]))
    z_sensor = np.hstack((b[0],b[1],b[2],w[0],w[1],w[2]))
    z_modelo = np.dot(H_k,x)
    y= z_sensor - z_modelo
    R = R_k(ruido, bias)
    # R = R_k(ruido, ruido_ss)
    K_k = k_kalman(R,P_k_priori,H_k)
    print(C)
    print("zsensor:",z_sensor)
    print("zmodelo:",z_modelo)

    delta_x = np.dot(K_k,y)
    delta_q_3 = delta_x[0:3]
    delta_w = delta_x[3:6]
    delta_q = np.hstack((delta_q_3, np.sqrt(1-np.linalg.norm(delta_q_3)**2)))
    # print("delta_q obtenido por kalman:",delta_q,"\n")
    
    q_posteriori = quat_mult(delta_q,q_priori)
    # print("q posteriori multi:",q_posteriori,"\n")
    w_posteriori = w_priori + delta_w
    
    # aa = q_priori + delta_q
    # q_posterior = aa / np.linalg.norm(aa)
    # print("q posteriori con suma:",q_posterior,"\n")
    
    P_ki = P_posteriori(K_k, H_k, P_k_priori)
    
    return q_posteriori, w_posteriori, P_ki


def rotacion_v(q, b_i, sigma):
    
    B_quat = [b_i[0],b_i[1],b_i[2],0]
    inv_q_b = inv_q(q)
    B_body = quat_mult(quat_mult(q,B_quat),inv_q_b)
    B_body_n = np.array([B_body[0],B_body[1],B_body[2]])
    B_magn = simulate_magnetometer_reading(B_body_n, sigma)
    
    return B_magn


#%% Restricciones del magnetorquer (restriccion de la accion de control)

def torquer(u_PD_NL,lim):
    
    if u_PD_NL[0]>lim: 
        u_PD_NL[0] = lim
    else:
        u_PD_NL[0] = u_PD_NL[0]

    if u_PD_NL[1]>lim:
        u_PD_NL[1] = lim

    else:
        u_PD_NL[1] = u_PD_NL[1]

    if u_PD_NL[2]>lim:
        u_PD_NL[2] = lim
    else:
        u_PD_NL[2] = u_PD_NL[2]

    if u_PD_NL[0]<-lim: 
        u_PD_NL[0] = -lim

    else:
        u_PD_NL[0] = u_PD_NL[0]

    if u_PD_NL[1]<-lim:
        u_PD_NL[1] = -lim

    else:
        u_PD_NL[1] = u_PD_NL[1]

    if u_PD_NL[2]<-lim:
        u_PD_NL[2] = -lim

    else:
        u_PD_NL[2] = u_PD_NL[2]
    
    return u_PD_NL

#%% Funcion para obtener el Mean Square Error

def RPY_MSE(t, q0_disc, q1_disc, q2_disc, q3_disc,q0_control, q1_control, q2_control, q3_control):
    
    q_kalman = np.vstack((q0_disc, q1_disc, q2_disc, q3_disc))
    q_kalman_t = np.transpose(q_kalman)
    RPY_kalman = []
    
    q_control = np.vstack((q0_control, q1_control, q2_control, q3_control))
    q_control_t = np.transpose(q_control)
    RPY_control = []
    
    for i in range(len(t)):
        RPY_EKF_id = quaternion_to_euler(q_kalman_t[i, :])
        RPY_kalman.append(RPY_EKF_id)
        RPY_control_id = quaternion_to_euler(q_control_t[i, :])
        RPY_control.append(RPY_control_id)
        
    RPY_kalman = np.array(RPY_kalman)
    RPY_control = np.array(RPY_control)
    
    restas_roll = []
    restas_pitch = []
    restas_yaw = []
    
    for i in range(len(t)):
        dif_roll = abs(RPY_control[i,0] - RPY_kalman[i,0])
        cuad_roll = dif_roll**2
        
        dif_pitch = abs(RPY_control[i,1] - RPY_kalman[i,1])
        cuad_pitch = dif_pitch**2
        
        dif_yaw = abs(RPY_control[i,2] - RPY_kalman[i,2])
        cuad_yaw = dif_yaw**2
        
        restas_roll.append(cuad_roll)
        restas_pitch.append(cuad_pitch)
        restas_yaw.append(cuad_yaw)
    
    restas_roll = np.array(restas_roll)
    sumatoria_roll = np.sum(restas_roll)
    mse_roll = sumatoria_roll / len(restas_roll)
    
    restas_pitch = np.array(restas_pitch)
    sumatoria_pitch = np.sum(restas_pitch)
    mse_pitch = sumatoria_pitch / len(restas_pitch)
    
    restas_yaw = np.array(restas_yaw)
    sumatoria_yaw = np.sum(restas_yaw)
    mse_yaw = sumatoria_yaw / len(restas_yaw)
    
    return RPY_kalman, RPY_control, mse_roll, mse_pitch, mse_yaw

#%%
def cuat_MSE_NL(q0,q1,q2,q3,w0,w1,w2,q0_nl,q1_nl,q2_nl,q3_nl,w0_nl,w1_nl,w2_nl):

    restas_q0_nl = []
    restas_q1_nl = []
    restas_q2_nl = []
    restas_q3_nl = []
    restas_w0_nl = []
    restas_w1_nl = []
    restas_w2_nl = []

    for i in range(len(q0_nl)):
            # Verificar si alguno de los elementos en la posición actual es NaN
        if math.isnan(q0[i]) or math.isnan(q0_nl[i]) or \
           math.isnan(q1[i]) or math.isnan(q1_nl[i]) or \
           math.isnan(q2[i]) or math.isnan(q2_nl[i]) or \
           math.isnan(q3[i]) or math.isnan(q3_nl[i]) or \
           math.isnan(w0[i]) or math.isnan(w0_nl[i]) or \
           math.isnan(w1[i]) or math.isnan(w1_nl[i]) or \
           math.isnan(w2[i]) or math.isnan(w2_nl[i]):
            break
        dif_q0_nl = abs(q0[i] - q0_nl[i])
        cuad_q0_nl = dif_q0_nl**2
        
        dif_q1_nl = abs(q1[i] - q1_nl[i])
        cuad_q1_nl = dif_q1_nl**2
        
        dif_q2_nl = abs(q2[i] - q2_nl[i])
        cuad_q2_nl = dif_q2_nl**2
        
        dif_q3_nl = abs(q3[i] - q3_nl[i])
        cuad_q3_nl = dif_q3_nl**2
        
        dif_w0_nl = abs(w0[i] - w0_nl[i])
        cuad_w0_nl = dif_w0_nl**2
        
        dif_w1_nl = abs(w1[i] - w1_nl[i])
        cuad_w1_nl = dif_w1_nl**2
        
        dif_w2_nl = abs(w2[i] - w2_nl[i])
        cuad_w2_nl = dif_w2_nl**2
        
        restas_q0_nl.append(cuad_q0_nl)
        restas_q1_nl.append(cuad_q1_nl)
        restas_q2_nl.append(cuad_q2_nl)
        restas_q3_nl.append(cuad_q3_nl)
        restas_w0_nl.append(cuad_w0_nl)
        restas_w1_nl.append(cuad_w1_nl)
        restas_w2_nl.append(cuad_w2_nl)
        
    restas_q0_nl = np.array(restas_q0_nl)
    sumatoria_q0_nl = np.sum(restas_q0_nl)
    mse_q0_nl = sumatoria_q0_nl / len(restas_q0_nl)
    # mse_q0_nl = sumatoria_q0_nl / 1

    restas_q1_nl = np.array(restas_q1_nl)
    sumatoria_q1_nl = np.sum(restas_q1_nl)
    mse_q1_nl = sumatoria_q1_nl / len(restas_q1_nl)
    # mse_q1_nl = sumatoria_q1_nl / 1

    restas_q2_nl = np.array(restas_q2_nl)
    sumatoria_q2_nl = np.sum(restas_q2_nl)
    mse_q2_nl = sumatoria_q2_nl / len(restas_q2_nl)
    # mse_q2_nl = sumatoria_q2_nl / 1

    restas_q3_nl = np.array(restas_q3_nl)
    sumatoria_q3_nl = np.sum(restas_q3_nl)
    mse_q3_nl = sumatoria_q3_nl / len(restas_q3_nl)
    # mse_q3_nl = sumatoria_q3_nl / 1

    restas_w0_nl = np.array(restas_w0_nl)
    sumatoria_w0_nl = np.sum(restas_w0_nl)
    mse_w0_nl = sumatoria_w0_nl / len(restas_w0_nl)
    # mse_w0_nl = sumatoria_w0_nl / 1

    restas_w1_nl = np.array(restas_w1_nl)
    sumatoria_w1_nl = np.sum(restas_w1_nl)
    mse_w1_nl = sumatoria_w1_nl / len(restas_w1_nl)
    # mse_w1_nl = sumatoria_w1_nl / 1

    restas_w2_nl = np.array(restas_w2_nl)
    sumatoria_w2_nl = np.sum(restas_w2_nl)
    mse_w2_nl = sumatoria_w2_nl / len(restas_w2_nl)
    # mse_w2_nl = sumatoria_w2_nl / 1
    
    MSE_cuat = np.array([mse_q0_nl,mse_q1_nl,mse_q2_nl,mse_q3_nl])
    MSE_omega = np.array([mse_w0_nl,mse_w1_nl,mse_w2_nl])
    
    return MSE_cuat, MSE_omega

#%%
def mod_lineal_disc_bar(x,u,deltat, h,A_discrete,B_discrete):
    
    
    for i in range(int(deltat/h)):
        x_k_1 = np.dot(A_discrete,x) - np.dot(B_discrete,u)
        q_rot = x_k_1[0:3]
        w_new = x_k_1[3:6]
    
        if  1-q_rot[0]**2-q_rot[1]**2-q_rot[2]**2 < 0:
            q_rot = q_rot / np.linalg.norm(q_rot)
            x_new = np.hstack((np.transpose(q_rot), np.transpose(w_new)))
            q3s_rot = 0
    
        else:
            x_new = np.hstack((np.transpose(q_rot), np.transpose(w_new)))
            q3s_rot = np.sqrt(1-q_rot[0]**2-q_rot[1]**2-q_rot[2]**2)
        
    return x_new, q3s_rot

#%% filtro de kalman discreto basado en Baroni (2019)

# Matriz omega capaz de entregarme el cuaternion a priori al multiplicarlo con un cuaternion inicial
def omega(w0,w1,w2,deltat): 
    w = np.array([w0,w1,w2])
    omega_1 = np.array([np.cos(0.5*np.linalg.norm(w)*deltat), 
                         (-np.sin(0.5*np.linalg.norm(w)*deltat)*w2)/np.linalg.norm(w),
                         (np.sin(0.5*np.linalg.norm(w)*deltat)*w1)/np.linalg.norm(w),
                         (np.sin(0.5*np.linalg.norm(w)*deltat)*w0)/np.linalg.norm(w)])
    
    omega_2 = np.array([(np.sin(0.5*np.linalg.norm(w)*deltat)*w2)/np.linalg.norm(w), 
                         np.cos(0.5*np.linalg.norm(w)*deltat),
                         (-np.sin(0.5*np.linalg.norm(w)*deltat)*w0)/np.linalg.norm(w),
                         (np.sin(0.5*np.linalg.norm(w)*deltat)*w1)/np.linalg.norm(w)])
    
    omega_3 = np.array([(-np.sin(0.5*np.linalg.norm(w)*deltat)*w1)/np.linalg.norm(w), 
                        (np.sin(0.5*np.linalg.norm(w)*deltat)*w0)/np.linalg.norm(w),
                         np.cos(0.5*np.linalg.norm(w)*deltat),
                         (np.sin(0.5*np.linalg.norm(w)*deltat)*w2)/np.linalg.norm(w)])
    
    omega_4 = np.array([(-np.sin(0.5*np.linalg.norm(w)*deltat)*w0)/np.linalg.norm(w), 
                        (-np.sin(0.5*np.linalg.norm(w)*deltat)*w1)/np.linalg.norm(w),
                        (-np.sin(0.5*np.linalg.norm(w)*deltat)*w2)/np.linalg.norm(w),
                         np.cos(0.5*np.linalg.norm(w)*deltat)])
    
    omega_f = np.array([omega_1,omega_2,omega_3,omega_4])
    
    return omega_f

# Matriz de transicion, util para conocer la matriz de covarianza a priori
def state_mat(w0,w1,w2,deltat):
    w = np.array([w0,w1,w2])
    skew_w_1 = np.array([0, -w2, w1])
    skew_w_2 = np.array([w2,0,-w0])
    skew_w_3 = np.array([-w1,w0,0])
    skew_w = np.array([skew_w_1,skew_w_2,skew_w_3])
    state_11 = np.eye(3) - skew_w* (np.sin(np.linalg.norm(w)*deltat))/np.linalg.norm(w) + np.dot(skew_w,skew_w)* (1-np.cos(np.linalg.norm(w)*deltat))/np.linalg.norm(w)**2
    state_12 = skew_w*(1-np.cos(np.linalg.norm(w)*deltat))/np.linalg.norm(w)**2 - np.eye(3)*deltat - np.dot(skew_w,skew_w) * (np.linalg.norm(w)*deltat - np.sin(np.linalg.norm(w)*deltat))/np.linalg.norm(w)**3   
    state_21 = np.zeros((3,3))
    state_22 = np.eye(3)
    
    state_1 = np.hstack((state_11,state_12))
    state_2 = np.hstack((state_21,state_22))
    
    state = np.vstack((state_1,state_2))
    
    return state   

def G_k(delta_T):
    G_11 = -np.eye(3)
    G_12 = np.zeros((3,3), dtype=int)
    G_21 = G_12
    G_22 = np.eye(3)
    
    G_1 = np.hstack((G_11, G_12))
    G_2 = np.hstack((G_21,G_22))
    
    G = np.vstack((G_1,G_2))
    
    return G

# Obtencion de la matriz de covarianza a priori
def P_k_priori(state_mat, P_k, G_k,Q_k):
    P_k_p = np.dot(state_mat,np.dot(P_k,np.transpose(state_mat))) + np.dot(G_k,np.dot(Q_k,np.transpose(G_k)))
    return P_k_p

# Implementacion del kalman en funcion
def kalman_baroni(q, w, bias_priori, deltaT, P_ki, Bi, vsuni, sigma_ww, sigma_bias, sigma_mm, sigma_ss):
    
    x = np.hstack((q[0:3],bias_priori))
    w_plus = w - bias_priori
    q_k_priori = np.dot(omega(w_plus[0],w_plus[1],w_plus[2],deltaT),q)
    state_matrix = state_mat(w_plus[0],w_plus[1],w_plus[2],deltaT)
    Q = Q_k(sigma_ww, sigma_bias, deltaT)
    G = G_k(deltaT)
    
    P_k_pr = P_k_priori(state_matrix,P_ki,G,Q)
    # H = H_k_bar(Bi[0], Bi[1], Bi[2], vsuni[0], vsuni[1], vsuni[2])
    H = H_k_bar(Bi[0], Bi[1], Bi[2], vsuni[0], vsuni[1], vsuni[2])
    R = R_k(sigma_mm, sigma_ss)
    K_k = k_kalman(R,P_k_pr,H)
    z_sensor = np.hstack((Bi[0],Bi[1],Bi[2],vsuni[0],vsuni[1],vsuni[2]))
    z_modelo = np.dot(H,x)
    y= z_sensor - z_modelo
    
    delta_x = np.dot(K_k,y)
    delta_q_3 = delta_x[0:3]
    delta_bias = delta_x[3:6]
    
    delta_q = np.hstack((delta_q_3, np.sqrt(1-np.linalg.norm(delta_q_3)**2)))
    q_posteriori = quat_mult(delta_q,q_k_priori)
    bias_posteriori = bias_priori - delta_bias
    
    P_k_pos = P_posteriori(K_k, H, P_k_pr)

    return q_posteriori, bias_posteriori, P_k_pos
