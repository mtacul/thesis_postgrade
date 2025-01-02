# -*- coding: utf-8 -*-
"""
Created on Sat May 11 23:33:44 2024

@author: nachi
"""
import numpy as np
import control as ctrl
import math
from scipy.optimize import minimize
from sigfig import round

# inversa de un cuaternion
def inv_q(q):
    inv_q = np.array([-q[0],-q[1],-q[2],q[3]])
    return inv_q

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

# Funciones para imposicion de ruido en magnetometro
def simulate_magnetometer_reading(B_eci, ruido):
    # np.random.seed(42)  # Puedes cambiar el número 42 a cualquier otro número entero

    # Simular el ruido gaussiano
    noise = np.random.normal(0, ruido, 1)
        
    # Simular la medición del magnetómetro con ruido
    measurement = B_eci + noise


    return measurement

# Rotacion utilizando cuaterniones
def rotacion_v(q, b_i, sigma):
    
    B_quat = [b_i[0],b_i[1],b_i[2],0]
    inv_q_b = inv_q(q)
    B_body = quat_mult(quat_mult(q,B_quat),inv_q_b)
    B_body_n = np.array([B_body[0],B_body[1],B_body[2]])
    B_magn = simulate_magnetometer_reading(B_body_n, sigma)
    
    return B_magn

# Funcion para generar realismo del giroscopio
def simulate_gyros_reading(w,ruido,bias):
    # np.random.seed(42)  # Puedes cambiar el número 42 a cualquier otro número entero

    #aplicar el ruido del sensor
    noise = np.random.normal(0, ruido, 1)
        
    #Simular la medicion del giroscopio
    measurement = w + noise + bias
        
    return measurement

# Matriz linealizada de la funcion dinamica no lineal derivada respecto al vector 
# estado en el punto de equilibrio x = [0,0,0,0,0,0] (tres primeros cuaterniones y
# tres componentes de velocidad angular)
def A_PD(I_x,I_y,I_z,w0_O, w0,w1,w2, I_s0_x, I_s1_y, I_s2_z, w_s0,w_s1,w_s2, J_x, J_y, J_z):
    A1 = np.array([0, 0.5*w2, -0.5*w1, 
                   0.5, 0,0,
                   0,0,0])
    A2 = np.array([-0.5*w2,0,0.5*w0,
                   0,0.5,0,
                   0,0,0])
    A3 = np.array([0.5*w1,-0.5*w0,0,
                   0,0,0.5,
                   0,0,0])
    
    A4 = np.array([6*w0_O**2*(I_z-I_y), 0, (2*w0_O*w_s2*I_s2_z) / (J_x-I_s0_x),
                   0, -w_s2*I_s2_z/(J_x-I_s0_x), w_s1*I_s1_y/(J_x-I_s0_x),
                   0,0,0])
    A5 = np.array([0, 6*w0_O**2*(I_z-I_x) + 2*w0_O**2*(J_x-J_z) / (J_y-I_s1_y) + (2*w0_O*w_s0*I_s0_x) / (J_y-I_s1_y), 0,
                   -w_s2*I_s2_z/(J_y-I_s1_y),0, w0_O + w0_O*(J_x-J_z)/(J_y-I_s1_y) + w_s0*I_s0_x/(J_y-I_s1_y),
                   0,0,-w0_O*I_s2_z/(J_y-I_s1_y)])
    A6 = np.array([0, 0, -2*w0_O**2*(J_x-J_y) / (J_z-I_s2_z) - (2*w0_O*w_s0*I_s0_x) / (J_z-I_s2_z),
                   -w_s1*(J_x-J_y)/(J_z-I_s2_z) , -w0_O + w0_O*(J_x-J_y)/(J_z-I_s2_z) + w_s0*I_s0_x/(J_z-I_s2_z), 0,
                   0,-w0_O*I_s1_y/(J_z-I_s2_z),0])
    
    A7 = np.array([0,0,-2*w0_O*w_s2*I_s2_z/(J_x-I_s0_x),
                   0,w_s2*I_s2_z/(J_x-I_s0_x),-w_s1*I_s1_y/(J_x-I_s0_x), 
                   0,0,0])
    A8 = np.array([0, -2*w0_O**2*(J_x-J_z) / (J_y-I_s1_y) - (2*w0_O*w_s0*I_s0_x) / (J_y-I_s1_y), 0,
                   w_s2*I_s2_z/(J_y-I_s1_y),0, -w0_O - w0_O*(J_x-J_z)/(J_y-I_s1_y) - w_s0*I_s0_x/(J_y-I_s1_y),
                   0,0,w0_O*I_s2_z/(J_y-I_s1_y)])
    A9 = np.array([0, 0, 2*w0_O**2*(J_x-J_y) / (J_z-I_s2_z) + (2*w0_O*w_s0*I_s0_x) / (J_z-I_s2_z),
                   w_s1*(J_x-J_y)/(J_z-I_s2_z) , w0_O - w0_O*(J_x-J_y)/(J_z-I_s2_z) - w_s0*I_s0_x/(J_z-I_s2_z), 0,
                   0,w0_O*I_s1_y/(J_z-I_s2_z),0])
    A_k = np.array([A1,A2,A3,A4,A5,A6,A7,A8,A9])
    
    return A_k     


#Matriz linealizada de la accion de control derivada respecto al vector de entrada
# en el punto de equilibrio x = [0,0,0,0,0,0]
def B_PD(w0_O,I_s0_x,I_s1_y,I_s2_z,J_x,J_y,J_z):
    B123 = np.zeros((3,3))
    B4 = np.array([1/(J_x-I_s0_x),0,0])
    B5 = np.array([0,1/(J_y-I_s1_y),0])
    B6 = np.array([0,0,1/(J_z-I_s2_z)])
    B7 = np.array([1/I_s0_x - 1/(J_x-I_s0_x), 0, 0])
    B8 = np.array([0, 1/I_s1_y - 1/(J_y-I_s1_y), 0])
    B9 = np.array([0,0, 1/I_s2_z - 1/(J_z-I_s2_z)])
    B_k = np.vstack((B123,B4,B5,B6,B7,B8,B9))
    #B_k = np.array([B123,B4,B5,B6])

    return B_k


# Matriz de ganancia
def K(Kp_x, Kp_y, Kp_z, Kd_x, Kd_y, Kd_z):
    K_gain = np.hstack((np.diag([Kp_x,Kp_y,Kp_z]), np.diag([Kd_x,Kd_y,Kd_z])))
    return K_gain


# Matriz H, que representa las mediciones derivadas respecto al vector estado x 
# (q0,q1,q2,biasx,biasy,biasz)
def H_k_bar(b0,b1,b2,s0,s1,s2,I_s0_x,I_s1_y,I_s2_z):
    
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
    H_13 = np.zeros((3,3))

    H_21 = 2*skew_s
    H_22 = H_12
    H_23 = H_13
    
    H_31 = np.zeros((3,3))
    H_32 = np.zeros((3,3))
    diag = np.array([I_s0_x , I_s1_y, I_s2_z])
    H_33 = np.diag(diag)

    H_1 = -np.hstack((H_11, H_12,H_13))
    H_2 = -np.hstack((H_21,H_22,H_23))
    H_3 = np.hstack((H_31,H_32,H_33))
    H = np.vstack((H_1,H_2,H_3))
    
    return H

# Obtencion de las matrices A y B de manera discreta
def A_B_kalman(I_x,I_y,I_z,w0_O, w0,w1,w2, I_s0_x, I_s1_y, I_s2_z, w_s0,w_s1,w_s2, J_x, J_y, J_z,deltat,h,b_orbit,b_body, s_body):
    
    A =A_PD(I_x,I_y,I_z,w0_O, w0,w1,w2, I_s0_x, I_s1_y, I_s2_z, w_s0,w_s1,w_s2, J_x, J_y, J_z)
    B = B_PD(w0_O,I_s0_x,I_s1_y,I_s2_z,J_x,J_y,J_z)
    # # Definición de C
    # C = np.hstack([np.eye(6), np.zeros((6, 3))])
    
    # # Construcción de Aa
    # Aa = np.block([
    #     [A, np.zeros((9, 6))],
    #     [C, np.zeros((6, 6))]
    # ])
    # Ba = np.vstack([B, np.zeros((6, 3))])

    # Define an identity matrix for C and a zero matrix for D to complete state-space model
    # C = np.eye(9)  # Assuming a 6x6 identity matrix for C
    C = H_k_bar(b_body[0],b_body[1],b_body[2], s_body[0],s_body[1],s_body[2],I_s0_x,I_s1_y,I_s2_z)
    D = np.zeros((9, 3))  # Assuming D has the same number of rows as A and the same number of columns as B
    # Ca = np.eye(15)
    # Da = np.zeros((15,3))
    
    # Create the continuous state-space model
    # sys_continuous = ctrl.StateSpace(Aa, Ba, Ca, Da)
    sys_continuous = ctrl.StateSpace(A, B, C, D)

    # A_discrete, B_discrete, _, _, _ = cont2discrete((Aa, Ba, Ca, Da), deltat, method='zoh')

    # Discretize the system
    sys_discrete = ctrl.c2d(sys_continuous, h, method='zoh')

    # Extract the discretized A and B matrices
    A_discrete = sys_discrete.A
    B_discrete = sys_discrete.B
    C_discrete = sys_discrete.C
    
    # return Aa,Ba,C,A_discrete,B_discrete,C_discrete
    return A,B,C,A_discrete,B_discrete,C_discrete

# Obtencion de las matrices A y B de manera discreta
def A_B(I_x,I_y,I_z,w0_O, w0,w1,w2, I_s0_x, I_s1_y, I_s2_z, w_s0,w_s1,w_s2, J_x, J_y, J_z,deltat,h,b_orbit,b_body, s_body):
    
    A =A_PD(I_x,I_y,I_z,w0_O, w0,w1,w2, I_s0_x, I_s1_y, I_s2_z, w_s0,w_s1,w_s2, J_x, J_y, J_z)
    B = B_PD(w0_O,I_s0_x,I_s1_y,I_s2_z,J_x,J_y,J_z)
    # # Definición de C
    # C = np.hstack([np.eye(6), np.zeros((6, 3))])
    
    # # Construcción de Aa
    # Aa = np.block([
    #     [A, np.zeros((9, 6))],
    #     [C, np.zeros((6, 6))]
    # ])
    # Ba = np.vstack([B, np.zeros((6, 3))])

    # Define an identity matrix for C and a zero matrix for D to complete state-space model
    C = np.eye(9)  # Assuming a 6x6 identity matrix for C
    # C = H_k_bar(b_body[0],b_body[1],b_body[2], s_body[0],s_body[1],s_body[2],I_s0_x,I_s1_y,I_s2_z)
    D = np.zeros((9, 3))  # Assuming D has the same number of rows as A and the same number of columns as B
    # Ca = np.eye(15)
    # Da = np.zeros((15,3))
    
    # Create the continuous state-space model
    # sys_continuous = ctrl.StateSpace(Aa, Ba, Ca, Da)
    sys_continuous = ctrl.StateSpace(A, B, C, D)

    # A_discrete, B_discrete, _, _, _ = cont2discrete((Aa, Ba, Ca, Da), deltat, method='zoh')

    # Discretize the system
    sys_discrete = ctrl.c2d(sys_continuous, h, method='zoh')

    # Extract the discretized A and B matrices
    A_discrete = sys_discrete.A
    B_discrete = sys_discrete.B
    # C_discrete = sys_discrete.C
    
    # return Aa,Ba,C,A_discrete,B_discrete,C_discrete
    return A,B,A_discrete,B_discrete

#%% Modelo lineal continuo

# funcion de la ecuacion xDot = Ax - Bu 
def dynamics(A, x, B, u):
    return np.dot(A, x) + np.dot(B, u)


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

    # Update components of ws
    ws0_new = x[6] + (k1[6] + 2 * k2[6] + 2 * k3[6] + k4[6]) / 6
    ws1_new = x[7] + (k1[7] + 2 * k2[7] + 2 * k3[7] + k4[7]) / 6
    ws2_new = x[8] + (k1[8] + 2 * k2[8] + 2 * k3[8] + k4[8]) / 6
    ws_new = np.array([ws0_new, ws1_new, ws2_new])
    return q_new_real, w_new, ws_new


def mod_lineal_cont(x,u,deltat,h,A,B):
    x_new = x
    for j in range(int(deltat/h)):
        q_rot,w_new,ws_new = rk4_step_PD(dynamics, x_new, A, B, u, h)
        if  1-q_rot[0]**2-q_rot[1]**2-q_rot[2]**2 < 0:
            q_rot = q_rot / np.linalg.norm(q_rot)
            x_new = np.hstack((np.transpose(q_rot), np.transpose(w_new),np.transpose(ws_new)))
            q3s_rot = 0

        else:
            x_new = np.hstack((np.transpose(q_rot), np.transpose(w_new),np.transpose(ws_new)))
            q3s_rot = np.sqrt(1-q_rot[0]**2-q_rot[1]**2-q_rot[2]**2)
                
    return x_new, q3s_rot

#%% Modelo lineal discreto

def mod_lineal_disc(x,u,deltat, h,A_discrete,B_discrete):
    x_new = x
    for i in range(int(deltat/h)):
        x_k_1 = np.dot(A_discrete,x_new) + np.dot(B_discrete,u)
        q_rot = x_k_1[0:3]
        w_new = x_k_1[3:6]
        ws_new = x_k_1[6:9]
    
        if  1-q_rot[0]**2-q_rot[1]**2-q_rot[2]**2 < 0:
            q_rot = q_rot / np.linalg.norm(q_rot)
            x_new = np.hstack((np.transpose(q_rot), np.transpose(w_new), np.transpose(ws_new)))
            q3s_rot = 0
    
        else:
            x_new = np.hstack((np.transpose(q_rot), np.transpose(w_new), np.transpose(ws_new)))
            q3s_rot = np.sqrt(1-q_rot[0]**2-q_rot[1]**2-q_rot[2]**2)
        
    return x_new, q3s_rot
#%% Filtro de kalman lineal y sus ecuaciones/matrices

def R_k(sigma_m, sigma_s):
    R_11 = np.eye(3) * sigma_m**2
    R_12 = np.zeros((3,3))
    R_13 = R_12
    
    R_21 = R_12
    R_22 = np.eye(3) * sigma_s**2
    R_23 = R_12
    
    R_31 = R_12
    R_32 = R_12
    R_33 = np.eye(3)
    
    R_1 = np.hstack((R_11, R_12,R_13))
    R_2 = np.hstack((R_21,R_22, R_23))
    R_3 = np.hstack((R_31,R_32, R_33))
    
    R = np.vstack((R_1,R_2,R_3))
    
    return R

# Matriz de covarianza del ruido del proceso se subdivide en Q
def Q_k(sigma_w,sigma_bias,deltat):
    Q_11 = np.eye(3) * (sigma_w**2*deltat + 1/3 * sigma_bias**2*deltat**3)
    Q_12 = np.eye(3) * -(1/2 * sigma_bias**2 * deltat**2)
    Q_13 = np.zeros((3,3))
    
    Q_21 = Q_12
    Q_22 = np.eye(3) * (sigma_bias**2 * deltat)
    Q_23 = np.zeros((3,3))
    
    Q_31 = Q_23
    Q_32 = Q_23
    Q_33 = Q_23
    
    Q_1 = np.hstack((Q_11, Q_12,Q_13))
    Q_2 = np.hstack((Q_21,Q_22,Q_23))
    Q_3 = np.hstack((Q_31,Q_32,Q_33))
    
    Q = np.vstack((Q_1,Q_2,Q_3))
    
    return Q

def P_k_prior(F_k, P_ki, Q_k):
    P_k_priori = np.dot(np.dot(F_k,P_ki),np.transpose(F_k)) + Q_k
    return P_k_priori #MATRIZ P PRIORI CON P INICIAL DADO, DESPUES SE ACTUALIZA


def k_kalman(R_k, P_k_priori, H_mat):
    K_k_izq =  np.dot(P_k_priori,np.transpose(H_mat))
    K_k_der = np.linalg.inv(R_k + np.dot(np.dot(H_mat,P_k_priori),np.transpose(H_mat)))
    
    K_k = np.dot(K_k_izq,K_k_der)
    return K_k #GANANCIA DE KALMAN


def P_posteriori(K_k,H_k,P_k_priori,R_k):
   I = np.identity(9)
   P_k_pos = np.dot(np.dot(I - np.dot(K_k,H_k),P_k_priori),np.transpose(I - np.dot(K_k,H_k))) + np.dot(np.dot(K_k,R_k),np.transpose(K_k))
   return P_k_pos #SACAR MATRIZ P POSTERIORI ACTUALIZADA


def kalman_lineal(A, B, C, x, u, b_orbit,b_real, s_orbit,s_real, P_ki, sigma_b, sigma_s,deltat,hh,h,I_s0_x,I_s1_y,I_s2_z,sigma_bb,sigma_ss):
    
    H_k = C
    [x_priori,q3s_rot] = mod_lineal_disc(x, u, deltat, hh, A, B) #para disc
    
    q_priori = np.hstack((x_priori[0:3], q3s_rot))
    w_priori = x_priori[3:6]
    ws_priori = x_priori[6:9]
    
    Q_ki = Q_k(5e-3, 3e-4,deltat)
    
    P_k_priori = P_k_prior(A,P_ki,Q_ki)
    R = R_k(sigma_bb, sigma_ss)
    K_k = k_kalman(R,P_k_priori,H_k)
    
    b_est = rotacion_v(q_priori, b_orbit,sigma_b)
    s_est = rotacion_v(q_priori, s_orbit,sigma_s)
    h_est = np.array([x_priori[6]/I_s0_x-x_priori[3], x_priori[7]/I_s1_y-x_priori[4], x_priori[8]/I_s2_z-x_priori[5]])

    z_sensor = np.hstack((b_real[0],b_real[1],b_real[2], s_real[0], s_real[1], s_real[2],h[0],h[1],h[2]))
    z_modelo = np.hstack((b_est[0],b_est[1],b_est[2], s_est[0], s_est[1], s_est[2],h_est[0],h_est[1],h_est[2]))
    y= z_sensor - z_modelo
    
    delta_x = np.dot(K_k,y)
    delta_q_3 = delta_x[0:3]
    delta_w = delta_x[3:6]
    delta_ws = delta_x[6:9]
    q3_delta =  np.sqrt(1-delta_q_3[0]**2-delta_q_3[1]**2-delta_q_3[2]**2)
    delta_q = np.hstack((delta_q_3, q3_delta))
    delta_qn = delta_q / np.linalg.norm(delta_q)

    q_posteriori = quat_mult(delta_qn,q_priori)
    # print("q posteriori multi:",q_posteriori,"\n")
    w_posteriori = w_priori + delta_w
    ws_posteriori = ws_priori + delta_ws
    
    P_ki = P_posteriori(K_k, H_k, P_k_priori,R)
    
    return q_posteriori, w_posteriori, P_ki,K_k, ws_posteriori

#%%

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

def torquer(u_PD_NL,lim):
    # Número de cifras significativas
    sig = 2
    u_PD_NL = np.array([round(x, sigfigs=sig) for x in u_PD_NL])

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

#%% Optimizacion del K

# Variables globales para manejar la solución
optimal_x = None
found_solution = False

def eigenvalue_constraint(x, A, B):
    eigs = []
    global found_solution, optimal_x
    K = np.hstack([np.diag(x[:3]), np.diag(x[3:6]),np.diag(x[6:])])  # Crear matriz de control K
    A_prim = A + B @ K
    eigenvalues = np.linalg.eigvals(A_prim)
    c = np.abs(eigenvalues) - 1  # Asegurarse de que todos los valores propios son menores que 1 en magnitud
    eigs.append(eigenvalues)
    if np.all(np.abs(eigenvalues) < 1):
        found_solution = True
        optimal_x = x  # Guardar la solución
        raise StopIteration("Found a solution with all eigenvalues having magnitude less than 1.")  # Lanzar una excepción para detener la optimización

    return c
    
def objective_function(x):
    return np.sqrt(np.sum(x**2))
    
    
def opt_K(A_discrete,B_discrete,deltat,h,x0):
    global found_solution, optimal_x
    
    found_solution = False
    optimal_x = None
    constraints = {'type': 'ineq', 'fun': eigenvalue_constraint, 'args': (A_discrete, B_discrete)}
    
    for i in range(600):
        print(i)
        random_adjustment = np.random.rand(len(x0))*100
        current_x0 = x0 + random_adjustment
        try:
            res = minimize(objective_function, current_x0, method='SLSQP', constraints=[constraints])
        except StopIteration as e:
            print(e)
            break  # Detener la iteración si se encuentra una solución válida
    
    if found_solution:
        print("Optimal solution found with all eigenvalues having magnitude less than 1:", optimal_x)
    else:
        print("No solution found with all eigenvalues having magnitude less than 1.")

    return optimal_x

