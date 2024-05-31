# -*- coding: utf-8 -*-
"""
Created on Fri May 10 20:22:29 2024

@author: nachi
"""
import numpy as np
import control as ctrl
import math

#%%
# Inversa de un cuaternion
def inv_q(q):
    inv_q = np.array([-q[0],-q[1],-q[2],q[3]])
    return inv_q


# Multiplicacion de cuaterniones
def quat_mult(qk_priori,dqk):
    dqk_n = dqk 
    
# Realizar la multiplicación de cuaterniones
    result = np.array([
    qk_priori[3]*dqk_n[0] + qk_priori[0]*dqk_n[3] + qk_priori[1]*dqk_n[2] - qk_priori[2]*dqk_n[1],  # Componente i
    qk_priori[3]*dqk_n[1] + qk_priori[1]*dqk_n[3] + qk_priori[2]*dqk_n[0] - qk_priori[0]*dqk_n[2],  # Componente j
    qk_priori[3]*dqk_n[2] + qk_priori[2]*dqk_n[3] + qk_priori[0]*dqk_n[1] - qk_priori[1]*dqk_n[0],  # Componente k
    qk_priori[3]*dqk_n[3] - qk_priori[0]*dqk_n[0] - qk_priori[1]*dqk_n[1] - qk_priori[2]*dqk_n[2]   # Componente escalar
    ])
    return result


# Simular magnetometro con ruido
def simulate_magnetometer_reading(B_body, ruido):
    # np.random.seed(42)  # Puedes cambiar el número 42 a cualquier otro número entero

    # Simular el ruido gaussiano
    noise = np.random.normal(0, ruido, 1)
        
    # Simular la medición del magnetómetro con ruido
    measurement = B_body + noise

    return measurement


# Rotación usando cuaterniones
def rotacion_v(q, b_i, sigma):
    
    B_quat = [b_i[0],b_i[1],b_i[2],0]
    inv_q_b = inv_q(q)
    B_body = quat_mult(quat_mult(q,B_quat),inv_q_b)
    B_body_n = np.array([B_body[0],B_body[1],B_body[2]])
    B_magn = simulate_magnetometer_reading(B_body_n, sigma)
    
    return B_magn


#%%
# Matriz linealizada de la funcion dinamica no lineal derivada respecto al vector 
# estado en el punto de equilibrio x = [0,0,0,0,0,0] (tres primeros cuaterniones y
# tres componentes de velocidad angular)
def A_PD(I_x,I_y,I_z,w0_O, w0,w1,w2):
    A1 = np.array([0, 0.5*w2, -0.5*w1, 0.5, 0,0])
    A2 = np.array([-0.5*w2,0,0.5*w0,0,0.5,0])
    A3 = np.array([0.5*w1,-0.5*w0,0,0,0,0.5])
    A4 = np.array([6*w0_O**2*(I_x-I_y), 0, 0, 0, w2*(I_y-I_z)/I_x, w1*(I_y-I_z)/I_x])
    A5 = np.array([0, 6*w0_O**2*(I_z-I_y), 0, w2*(I_x-I_z)/I_y,0, (w0+w0_O)*(I_x-I_z)/I_y + I_y*w0_O])
    A6 = np.array([0, 0, 0, w1*(I_y-I_x)/I_z, (w0+w0_O)*(I_y-I_x)/I_z - I_z*w0_O, 0])
    
    A_k = np.array([A1,A2,A3,A4,A5,A6])
    
    return A_k    


#Matriz linealizada de la accion de control derivada respecto al vector estado
# en el punto de equilibrio x = [0,0,0,0,0,0]
def B_PD(I_x,I_y,I_z,B_magnet):
    b_norm = np.linalg.norm(B_magnet)
    B123 = np.zeros((3,3))
    B4 = np.array([(-(B_magnet[2]**2)-B_magnet[1]**2)/(b_norm*I_x), B_magnet[1]*B_magnet[0]/(b_norm*I_x), B_magnet[2]*B_magnet[0]/(b_norm*I_x)])
    B5 = np.array([B_magnet[0]*B_magnet[1]/(b_norm*I_y), (-B_magnet[2]**2-B_magnet[0]**2)/(b_norm*I_y), B_magnet[2]*B_magnet[1]/(b_norm*I_y)])
    B6 = np.array([B_magnet[0]*B_magnet[2]/(b_norm*I_z), B_magnet[1]*B_magnet[2]/(b_norm*I_z), (-B_magnet[1]**2-B_magnet[0]**2)/(b_norm*I_z)])
    
    B_k = np.vstack((B123,B4,B5,B6))
    #B_k = np.array([B123,B4,B5,B6])

    return B_k


# Matriz de ganancia
def K(Kp_x, Kp_y, Kp_z, Kd_x, Kd_y, Kd_z):
    K_gain = np.hstack((np.diag([Kp_x,Kp_y,Kp_z]), np.diag([Kd_x,Kd_y,Kd_z])))
    return K_gain


# Matriz H, que representa las mediciones derivadas respecto al vector estado x 
# (q0,q1,q2,biasx,biasy,biasz)
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


# Obtencion de las matrices A y B de manera discreta
def A_B(I_x,I_y,I_z,w0_O,w0,w1,w2,deltat,h,b_orbit,b_body, s_body):
    
    A =A_PD(I_x,I_y,I_z,w0_O, w0,w1,w2)
    B = B_PD(I_x,I_y,I_z,b_orbit)
    
    # Define an identity matrix for C and a zero matrix for D to complete state-space model
    # C = np.eye(6)  # Assuming a 6x6 identity matrix for C
    C = H_k_bar(b_body[0],b_body[1],b_body[2], s_body[0],s_body[1],s_body[2])
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

#%% Modelo lineal continuo

# funcion de la ecuacion xDot = Ax - Bu 
def dynamics(A, x, B, u):
    return np.dot(A, x) - np.dot(B, u)


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
    return part_1_w2*part_2_w2*(I_y-I_x)/I_z - part_3_w2 + part_4_w2 + part_5_w2

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
    tt = 2
    
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

#%% Mean Square Error para cuaterniones y velocidad angular

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