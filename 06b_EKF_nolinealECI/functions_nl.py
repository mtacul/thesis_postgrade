# -*- coding: utf-8 -*-
"""
Created on Wed May  8 01:17:39 2024

@author: nachi
"""
import numpy as np
import control as ctrl

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

#%%inversa de un cuaternion

def inv_q(q):
    inv_q = np.array([-q[0],-q[1],-q[2],q[3]])
    return inv_q

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
def simulate_gyros_reading(w,ruido,s_bias):
    # np.random.seed(42)  # Puedes cambiar el número 42 a cualquier otro número entero

    #aplicar el ruido del sensor
    noise = np.random.normal(0, ruido, 1)
    
    # bias = np.random.normal(0,s_bias,1)
    bias_x = np.random.normal(0,abs(s_bias[0]),1)
    bias_y = np.random.normal(0,abs(s_bias[1]),1)    
    bias_z = np.random.normal(0,abs(s_bias[2]),1)    
    
    #Simular la medicion del giroscopio
    # measurement = w + noise + bias
    w_x = w[0] + noise + bias_x
    w_y = w[1] + noise + bias_y
    w_z = w[2] + noise + bias_z
    
    measurement = np.array([w_x,w_y,w_z])
    return measurement

def rotacion_v(q, b_i):
    
    B_quat = [b_i[0],b_i[1],b_i[2],0]
    inv_q_b = inv_q(q)
    B_body = quat_mult(quat_mult(q,B_quat),inv_q_b)
    B_body_n = np.array([B_body[0],B_body[1],B_body[2]])
    
    return B_body_n

# #%% Modelo lineal discreto

# # Matriz linealizada de la funcion dinamica no lineal derivada respecto al vector 
# # estado en el punto de equilibrio x = [0,0,0,0,0,0] (tres primeros cuaterniones y
# # tres componentes de velocidad angular)
# def A_PD(I_x,I_y,I_z,w0_O, w0,w1,w2):
#     A1 = np.array([0, 0.5*w2, -0.5*w1, 0.5, 0,0])
#     A2 = np.array([-0.5*w2,0,0.5*w0,0,0.5,0])
#     A3 = np.array([0.5*w1,-0.5*w0,0,0,0,0.5])
#     A4 = np.array([6*w0_O**2*(I_x-I_y), 0, 0, 0, w2*(I_y-I_z)/I_x, w1*(I_y-I_z)/I_x])
#     A5 = np.array([0, 6*w0_O**2*(I_z-I_y), 0, w2*(I_x-I_z)/I_y,0, (w0+w0_O)*(I_x-I_z)/I_y + I_y*w0_O])
#     A6 = np.array([0, 0, 0, w1*(I_y-I_x)/I_z, (w0+w0_O)*(I_y-I_x)/I_z - I_z*w0_O, 0])
    
#     A_k = np.array([A1,A2,A3,A4,A5,A6])
    
#     return A_k    


# #Matriz linealizada de la accion de control derivada respecto al vector estado
# # en el punto de equilibrio x = [0,0,0,0,0,0]
# def B_PD(I_x,I_y,I_z,B_magnet):
#     b_norm = np.linalg.norm(B_magnet)
#     B123 = np.zeros((3,3))
#     B4 = np.array([(-(B_magnet[2]**2)-B_magnet[1]**2)/(b_norm*I_x), B_magnet[1]*B_magnet[0]/(b_norm*I_x), B_magnet[2]*B_magnet[0]/(b_norm*I_x)])
#     B5 = np.array([B_magnet[0]*B_magnet[1]/(b_norm*I_y), (-B_magnet[2]**2-B_magnet[0]**2)/(b_norm*I_y), B_magnet[2]*B_magnet[1]/(b_norm*I_y)])
#     B6 = np.array([B_magnet[0]*B_magnet[2]/(b_norm*I_z), B_magnet[1]*B_magnet[2]/(b_norm*I_z), (-B_magnet[1]**2-B_magnet[0]**2)/(b_norm*I_z)])
    
#     B_k = np.vstack((B123,B4,B5,B6))
#     #B_k = np.array([B123,B4,B5,B6])

#     return B_k

# # Obtencion de las matrices A y B de manera discreta
# def A_B(I_x,I_y,I_z,w0_O,w0,w1,w2,deltat,h,b_body, s_body):
    
#     A =A_PD(I_x,I_y,I_z,w0_O, w0,w1,w2)
#     B = B_PD(I_x,I_y,I_z,b_body)
    
#     # Define an identity matrix for C and a zero matrix for D to complete state-space model
#     # C = np.eye(6)  # Assuming a 6x6 identity matrix for C
#     C = H_k_bar(b_body[0],b_body[1],b_body[2], s_body[0],s_body[1],s_body[2])
#     D = np.zeros((6, 3))  # Assuming D has the same number of rows as A and the same number of columns as B

#     # Create the continuous state-space model
#     sys_continuous = ctrl.StateSpace(A, B, C, D)

#     # Discretize the system
#     sys_discrete = ctrl.c2d(sys_continuous, deltat*h, method='zoh')

#     # Extract the discretized A and B matrices
#     A_discrete = sys_discrete.A
#     B_discrete = sys_discrete.B
#     C_discrete = sys_discrete.C
    
#     return A,B,C,A_discrete,B_discrete,C_discrete

# def mod_lineal_disc(x,u,deltat, h,A_discrete,B_discrete):
        
#     for i in range(int(1/h)):
#         x_k_1 = np.dot(A_discrete,x) - np.dot(B_discrete,u)
#         q_rot = x_k_1[0:3]
#         w_new = x_k_1[3:6]
    
#         if  1-q_rot[0]**2-q_rot[1]**2-q_rot[2]**2 < 0:
#             q_rot = q_rot / np.linalg.norm(q_rot)
#             x_new = np.hstack((np.transpose(q_rot), np.transpose(w_new)))
#             q3s_rot = 0
    
#         else:
#             x_new = np.hstack((np.transpose(q_rot), np.transpose(w_new)))
#             q3s_rot = np.sqrt(1-q_rot[0]**2-q_rot[1]**2-q_rot[2]**2)
        
#     return x_new, q3s_rot

#Funciones ecuaciones no lineales, en conjunto con su resolucion en RK4
#%% Modelo no lineal continuo

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


def mod_nolineal(x,u,deltat, b,h,tt,I_x,I_y,I_z,w0_O):
    
    
    if  1-x[0]**2-x[1]**2-x[2]**2 < 0:
        x[0:3] = x[0:3] / np.linalg.norm(x[0:3])
        q3s_rot = 0

    else:
        q3s_rot = np.sqrt(1-x[0]**2-x[1]**2-x[2]**2)
        
    
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
    Q_11 = np.eye(3) * (sigma_w**2*deltat + 1/3 * sigma_bias**2 * deltat**3)
    Q_12 = np.eye(3) * -(1/2 * sigma_bias**2 * deltat**2)
    Q_21 = Q_12
    Q_22 = np.eye(3) * (sigma_bias**2 * deltat)
    
    Q_1 = np.hstack((Q_11, Q_12))
    Q_2 = np.hstack((Q_21,Q_22))
    
    Q = np.vstack((Q_1,Q_2))
    
    return Q

# Obtencion de la matriz de covarianza a priori
def P_k_priori(state_mat, P_k, G_k,Q_k):
    P_k_p = np.dot(state_mat,np.dot(P_k,np.transpose(state_mat))) + np.dot(G_k,np.dot(Q_k,np.transpose(G_k)))
    return P_k_p

def H_k_bar(b0,b1,b2,s0,s1,s2):

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

def k_kalman(R_k, P_k_priori, H_mat):
    K_k_izq =  np.dot(P_k_priori,np.transpose(H_mat))
    K_k_der = np.linalg.inv(R_k + np.dot(np.dot(H_mat,P_k_priori),np.transpose(H_mat)))
    
    K_k = np.dot(K_k_izq,K_k_der)
    return K_k #GANANCIA DE KALMAN

def P_posteriori(K_k,H_k,P_k_priori,R_k):
   I = np.identity(6)
   P_k_pos = np.dot(np.dot(I - np.dot(K_k,H_k),P_k_priori),np.transpose(I - np.dot(K_k,H_k))) + np.dot(np.dot(K_k,R_k),np.transpose(K_k))
   return P_k_pos #SACAR MATRIZ P POSTERIORI ACTUALIZADA

# Implementacion del kalman en funcion
def kalman_baroni(q, w, bias_priori, deltaT, P_ki, b_body_med, s_body_med,b_body_est, s_body_est,  sigma_ww, sigma_bias, sigma_mm, sigma_ss):
    
    x = np.hstack((q[0:3],bias_priori))
    w_plus = w - bias_priori
    q_k_priori = np.dot(omega(w_plus[0],w_plus[1],w_plus[2],deltaT),q)
    state_matrix = state_mat(w_plus[0],w_plus[1],w_plus[2],deltaT)
    Q = Q_k(sigma_ww, sigma_bias, deltaT)
    G = G_k(deltaT)
    
    P_k_pr = P_k_priori(state_matrix,P_ki,G,Q)
    H = H_k_bar(b_body_med[0], b_body_med[1],b_body_med[2], s_body_med[0], s_body_med[1], s_body_med[2])
    R = R_k(sigma_mm, sigma_ss)
    K_k = k_kalman(R,P_k_pr,H)
    z_sensor = np.hstack((b_body_med[0],b_body_med[1],b_body_med[2],s_body_med[0],s_body_med[1],s_body_med[2]))
    z_modelo = np.hstack((b_body_est[0],b_body_est[1],b_body_est[2],s_body_est[0],s_body_est[1],s_body_est[2]))
    y= z_sensor - z_modelo
    # print(H)
    # print("zsensor:",z_sensor)
    # print("zmodelo:",z_modelo)
    # print("y:",y)

    delta_x = np.dot(K_k,y)
    # print("deltax:",delta_x)
    delta_q_3 = delta_x[0:3]
    delta_bias = delta_x[3:6]
    
    delta_q = np.hstack((delta_q_3, np.sqrt(1-np.linalg.norm(delta_q_3)**2)))
    # delta_q = np.hstack((delta_q_3, 1))
    q_posteriori = quat_mult(delta_q,q_k_priori)
    # print("q_posteriori:",q_posteriori)

    bias_posteriori = bias_priori - delta_bias
    
    P_k_pos = P_posteriori(K_k, H, P_k_pr, R)

    return q_posteriori, bias_posteriori, P_k_pos, w_plus