# -*- coding: utf-8 -*-
"""
Created on Sat May 11 23:33:44 2024

@author: nachi
"""
import numpy as np
import control as ctrl
import math
from scipy.optimize import minimize

# Funcion para generar realismo del giroscopio
def simulate_gyros_reading(w,ruido,bias):
    # np.random.seed(42)  # Puedes cambiar el número 42 a cualquier otro número entero

    #aplicar el ruido del sensor
    noise = np.random.normal(0, ruido, 1)
        
    #Simular la medicion del giroscopio
    measurement = w + noise + bias
        
    return measurement

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
def H_Velez(Bref,Bmagn):
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



# Obtencion de las matrices A y B de manera discreta
def A_B(I_x,I_y,I_z,w0_O,w0,w1,w2,deltat,h,b_body, b_orbit):
    
    A =A_PD(I_x,I_y,I_z,w0_O, w0,w1,w2)
    B = B_PD(I_x,I_y,I_z,b_orbit)
    
    # Define an identity matrix for C and a zero matrix for D to complete state-space model
    # C = np.eye(6)  # Assuming a 6x6 identity matrix for C
    C = H_Velez(b_orbit, b_body)
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


def P_posteriori(K_k,H_k,P_k_priori,R_k):
   I = np.identity(6)
   P_k_pos = np.dot(np.dot(I - np.dot(K_k,H_k),P_k_priori),np.transpose(I - np.dot(K_k,H_k))) + np.dot(np.dot(K_k,R_k),np.transpose(K_k))
   return P_k_pos #SACAR MATRIZ P POSTERIORI ACTUALIZADA

# y = np.array([0,0,0,0,0,0])

def kalman_lineal(A, B, C, x, u, b, b_eci, P_ki, sigma_m, sigma_ss,deltat,hh):
# def kalman_lineal(A, B, C, x, u, b, b_eci, P_ki, ruido, ruido_ss,deltat,s):
    
    H_k = C
    [x_priori,q3s_rot] = mod_lineal_disc(x, u, deltat, hh, A, B)
    q_priori = np.hstack((x_priori[0:3], q3s_rot))

    # print("q priori obtenido por kalman:",q_priori,"\n")
    w_priori = x_priori[3:6]
    
    Q_ki = Q_k(5e-3, 3e-4,deltat)
    
    P_k_priori = P_k_prior(A,P_ki,Q_ki)
    
    z_sensor = np.hstack((b[0],b[1],b[2], w_priori[0], w_priori[1], w_priori[2]))
    z_modelo = np.dot(H_k,x)
    y_03 = np.cross(b,b_eci) / (np.linalg.norm(b) * np.linalg.norm(b_eci))
    print(y_03)
    y_06 = z_sensor[3:6] - z_modelo[3:6]
    y = np.hstack((y_03,y_06)) 
    
    R = R_k(sigma_m, sigma_ss)
    K_k = k_kalman(R,P_k_priori,H_k)
    # print("zsensor:",z_sensor)
    # print("zmodelo:",z_modelo)
    # print("Y:",y)
    
    delta_x = np.dot(K_k,y)
    # print("deltax",delta_x)
    delta_q_3 = delta_x[0:3]
    delta_w = delta_x[3:6]
    q3_delta =  np.sqrt(1-delta_q_3[0]**2-delta_q_3[1]**2-delta_q_3[2]**2)
    delta_q = np.hstack((delta_q_3, q3_delta))
    delta_qn = delta_q / np.linalg.norm(delta_q)
    # print("deltaq",delta_qn)

    # print("delta_q obtenido por kalman:",delta_q,"\n")
    
    q_posteriori = quat_mult(delta_qn,q_priori)
    # print("q posteriori multi:",q_posteriori,"\n")
    w_posteriori = w_priori + delta_w

    
    P_ki = P_posteriori(K_k, H_k, P_k_priori,R)
    
    return q_posteriori, w_posteriori, P_ki,K_k

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

#%% Optimizacion del K

# Variables globales para manejar la solución
optimal_x = None
found_solution = False

def eigenvalue_constraint(x, A, B):
    global found_solution, optimal_x
    K = np.hstack([np.diag(x[:3]), np.diag(x[3:])])  # Crear matriz de control K
    A_prim = A - B @ K
    eigenvalues = np.linalg.eigvals(A_prim)
    c = np.abs(eigenvalues) - 1  # Asegurarse de que todos los valores propios son menores que 1 en magnitud

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
    
    for i in range(1000):
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
