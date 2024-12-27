# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 20:54:59 2024

@author: nachi
"""

import functions_03_rw
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_are, solve_continuous_are
import control as ctrl


deltat = 2
# limite =  5762*69
limite =  5762*1
t = np.arange(0, limite, deltat)

#%% Parámetros geométricos y orbitales dados

w0_O = 0.00163

I_x = 0.037
I_y = 0.036
I_z = 0.006

m_s0 = 0.06 #kg
m_s1 = m_s0
m_s2 = m_s0

b_0 = 0.05
b_1 = 0.05
b_2 = 0.15

I_s0_x = 0.005
I_s1_x = I_s0_x
I_s2_x = I_s0_x

I_s0_y = 0.005
I_s1_y = I_s0_y
I_s2_y = I_s0_y

I_s0_z = 0.004
I_s1_z = I_s0_z
I_s2_z = I_s0_z	

J_x = I_x + I_s0_x + I_s1_x + I_s2_x + m_s1*b_1**2 + m_s2*b_2**2
J_y = I_y + I_s0_y + I_s1_y + I_s2_y + m_s0*b_0**2 + m_s2*b_2**2
J_z = I_z + I_s0_z + I_s1_z + I_s2_z + m_s0*b_0**2 + m_s1*b_1**2

# _mod de modelo
hh = deltat

deltat = 2
bi_orbit = [-8.991298983741396e-06,
-5.907147530097531e-06,
-4.5687558794566606e-05]

b_body_i = [-8.991298983741396e-06,
-5.907147530097531e-06,
-4.5687558794566606e-05]

si_orbit = [0.776906,
0.57818,
-0.249251]
s_body_i = [0.776906,
0.57818,
-0.249251]


[A,B,A_discrete,B_discrete] = functions_03_rw.A_B(I_x,I_y,I_z,w0_O,0,0,0 , I_s0_x, I_s1_y, I_s2_z, 0,0,0, J_x, J_y, J_z, deltat, hh, bi_orbit,b_body_i, s_body_i)

# # Índices correspondientes a las filas y columnas
# indices_0 = [0, 3, 6]
# indices_1 = [1, 4, 7]
# indices_2 = [2, 5, 8]

# # Seleccionar la primera columna y las filas deseadas
# B_discrete_3x1_0 = B_discrete[indices_0, 0]
# B_discrete_3x1_1 = B_discrete[indices_1, 1]
# B_discrete_3x1_2 = B_discrete[indices_2, 2]

# # Convertir a una matriz columna (3x1)
# B_discrete_0 = B_discrete_3x1_0.reshape(-1, 1)# Extraer submatriz de 3x3
# B_discrete_1 = B_discrete_3x1_1.reshape(-1, 1)# Extraer submatriz de 3x3
# B_discrete_2 = B_discrete_3x1_2.reshape(-1, 1)# Extraer submatriz de 3x3

# A_discrete_0 = A_discrete[np.ix_(indices_0, indices_0)]
# A_discrete_1 = A_discrete[np.ix_(indices_1, indices_1)]
# A_discrete_2 = A_discrete[np.ix_(indices_2, indices_2)]

C = np.eye(3)
D = np.zeros((3, 1))  # Assuming D has the same number of rows as A and the same number of columns as B

# Índices correspondientes a las filas y columnas
indices_0 = [0, 3, 6]
indices_1 = [1, 4, 7]
indices_2 = [2, 5, 8]

# Seleccionar la primera columna y las filas deseadas
B_3x1_0 = B[indices_0, 0]
B_3x1_1 = B[indices_1, 1]
B_3x1_2 = B[indices_2, 2]

# Convertir a una matriz columna (3x1)
B_0 = B_3x1_0.reshape(-1, 1)# Extraer submatriz de 3x3
B_1 = B_3x1_1.reshape(-1, 1)# Extraer submatriz de 3x3
B_2 = B_3x1_2.reshape(-1, 1)# Extraer submatriz de 3x3

A_0 = A[np.ix_(indices_0, indices_0)]
A_1 = A[np.ix_(indices_1, indices_1)]
A_2 = A[np.ix_(indices_2, indices_2)]




# sys_continuous_0 = ctrl.StateSpace(A_3x3_0, B_3x1_0, C, D)
# sys_continuous_1 = ctrl.StateSpace(A_3x3_1, B_3x1_1, C, D)
# sys_continuous_2 = ctrl.StateSpace(A_3x3_2, B_3x1_2, C, D)

# # A_discrete, B_discrete, _, _, _ = cont2discrete((Aa, Ba, Ca, Da), deltat, method='zoh')

# # Discretize the system
# sys_discrete_0 = ctrl.c2d(sys_continuous_0, hh, method='zoh')
# sys_discrete_1 = ctrl.c2d(sys_continuous_1, hh, method='zoh')
# sys_discrete_2 = ctrl.c2d(sys_continuous_2, hh, method='zoh')

# # Extract the discretized A and B matrices
# A_discrete_0 = sys_discrete_0.A
# A_discrete_1 = sys_discrete_1.A
# A_discrete_2 = sys_discrete_2.A

# B_discrete_0 = sys_discrete_0.B
# B_discrete_1 = sys_discrete_1.B
# B_discrete_2 = sys_discrete_2.B


#%% Control LQR en cada matriz

# Definir las matrices Q y R del coste del LQR
diag_Q1 = np.array([10, 100, 1000])*100
# diag_Q = np.array([10, 10, 10, 0.1, 0.1, 0.1, 10, 10, 10])*1
diag_R1 = np.array([1])
Q1 = np.diag(diag_Q1)
R1 = np.diag(diag_R1)

diag_Q2 = np.array([10, 1, 0.01])*10000
# diag_Q = np.array([10, 10, 10, 0.1, 0.1, 0.1, 10, 10, 10])*1
diag_R2 = np.array([10000])
Q2 = np.diag(diag_Q2)
R2 = np.diag(diag_R2)

diag_Q3 = np.array([10, 10, 0.1])*10000
# diag_Q = np.array([10, 10, 10, 0.1, 0.1, 0.1, 10, 10, 10])*1
diag_R3 = np.array([10000])
Q3 = np.diag(diag_Q3)
R3 = np.diag(diag_R3)


K1, P1, eigenvalues1 = ctrl.lqr(A_0, B_0, Q1, R1)
K2, P2, eigenvalues2 = ctrl.lqr(A_1, B_1, Q2, R2)
K3, P3, eigenvalues3 = ctrl.lqr(A_2, B_2, Q3, R3)

# K1, P1, eigenvalues1 = ctrl.dlqr(A_discrete_0, B_discrete_0, Q1, R1)
# K2, P2, eigenvalues2 = ctrl.dlqr(A_discrete_1, B_discrete_1, Q2, R2)
# K3, P3, eigenvalues3 = ctrl.dlqr(A_discrete_2, B_discrete_2, Q3, R3)

K1_1 = np.array([K1[0][0],K1[0][1],K1[0][2]])
# K1_1 = np.diag(K1_1)

K2_1 = np.array([K2[0][0],K2[0][1],K2[0][2]])
# K2_1 = np.diag(K2_1)

K3_1 = np.array([K3[0][0],K3[0][1],K3[0][2]])
# K3_1 = np.diag(K3_1)

Kk1 = np.array([K1_1[0],0,0,K1_1[1],0,0,K1_1[2],0,0])
Kk2 = np.array([0,K2_1[0],0,0,K2_1[1],0,0,K2_1[2],0])
Kk3 = np.array([0,0,K3_1[0],0,0,K3_1[1],0,0,K3_1[2]])

K = np.vstack((Kk1,Kk2,Kk3))
asad,vect = np.linalg.eig(A-B@K)
asad1,vect1 = np.linalg.eig(A_0-B_0@K1)
asad2,vect2 = np.linalg.eig(A_1-B_1@K2)
asad3,vect3 = np.linalg.eig(A_2-B_2@K3)
# asad,vect = np.linalg.eig(A_discrete-B_discrete@K)
# asad1,vect1 = np.linalg.eig(A_discrete_0-B_discrete_0@K1)
# asad2,vect2 = np.linalg.eig(A_discrete_1-B_discrete_1@K2)
# asad3,vect3 = np.linalg.eig(A_discrete_2-B_discrete_2@K3)

q = np.array([0.0789,0.0941,0.0789,0.9893])
w = np.array([0.0001, 0.0001, 0.0001])
ws = np.array([0.00001, 0.00001, 0.00001])

q0_real = [q[0]]
q1_real = [q[1]]
q2_real = [q[2]]
q3_real = [q[3]]
w0_real = [w[0]]
w1_real = [w[1]]
w2_real = [w[2]]
w0s_real = [ws[0]]
w1s_real = [ws[1]]
w2s_real = [ws[2]]
q_real = np.array([q0_real[-1],q1_real[-1],q2_real[-1],q3_real[-1]])
w_body = np.array([w0_real[-1], w1_real[-1], w2_real[-1]])
w_gyros = functions_03_rw.simulate_gyros_reading(w_body, 0,0)
ws_real = np.array([w0s_real[-1], w1s_real[-1], w2s_real[-1]])
x_real = np.hstack((np.transpose(q_real[:3]), np.transpose(w_gyros), np.transpose(ws_real)))

for i in range(len(t)-1):
    print(t[i+1])
    # print(x_real)
    # print(np.dot(-K*10,x_real))
    

    u_est = np.dot(-K,x_real)
    # print(u_est)
    u_est = functions_03_rw.torquer(u_est,100000)
    
    [xx_new_d, qq3_new_d] = functions_03_rw.mod_lineal_disc(
        x_real, u_est, deltat, hh, A_discrete,B_discrete)
    
    # print(u_est)
    # print(xx_new_d)
    # input()
    x_real = xx_new_d
    w_gyros = functions_03_rw.simulate_gyros_reading(x_real[3:6],0,0)
    ws_real = x_real[6:9]
    q0_real.append(xx_new_d[0])
    q1_real.append(xx_new_d[1])
    q2_real.append(xx_new_d[2])
    q3_real.append(qq3_new_d)
    w0_real.append(w_gyros[0])
    w1_real.append(w_gyros[1])
    w2_real.append(w_gyros[2])
    w0s_real.append(ws_real[0])
    w1s_real.append(ws_real[1])
    w2s_real.append(ws_real[2])



fig0, axes0 = plt.subplots(nrows=1, ncols=2, figsize=(16, 4))

axes0[0].plot(t, q0_real, label='q0 modelo')
axes0[0].plot(t, q1_real, label='q1 modelo')
axes0[0].plot(t, q2_real, label='q2 modelo')
axes0[0].plot(t, q3_real, label='q3 modelo')
axes0[0].set_xlabel('Tiempo [s]')
axes0[0].set_ylabel('cuaternion [-]')
axes0[0].legend()
axes0[0].set_title('cuaterniones obtenidos por el modelo de control lineal discreto')
axes0[0].grid()
# axes0[0].set_ylim(-1, 1)  # Ajusta los límites en el eje Y

axes0[1].plot(t, w0_real, label='w0 modelo')
axes0[1].plot(t, w1_real, label='w1 modelo')
axes0[1].plot(t, w2_real, label='w2 modelo')
axes0[1].set_xlabel('Tiempo [s]')
axes0[1].set_ylabel('velocidad angular [rad/s]')
axes0[1].legend()
axes0[1].set_title('velocidades angulares obtenidos por el modelo de control lineal discreto')
axes0[1].grid()

plt.tight_layout()
plt.show()

#%% Control LQR

# # Definir las matrices Q y R del coste del LQR
# diag_Q = np.array([1, 1, 1, 1, 1, 1, 0.1, 0.1, 0.1])*1
# # diag_Q = np.array([10, 10, 10, 0.1, 0.1, 0.1, 10, 10, 10])*1
# diag_R = np.array([diag_Q[6],diag_Q[7],diag_Q[8]])
# Q = np.diag(diag_Q)
# R = np.diag(diag_R)

# print(A_discrete.shape)
# print(B_discrete.shape)
# print(Q.shape)
# print(R.shape)



# # # Resolver la ecuación de Riccati
# # X,L,G=control.dare(A_discrete,B_discrete,Q,R)
# P = solve_discrete_are(A_discrete, B_discrete, Q, R)

# # # Calcular la matriz de retroalimentación K
# K = np.linalg.inv(B_discrete.T @ P @ B_discrete + R) @ (B_discrete.T @ P @ A_discrete)

# # P = solve_continuous_are(A,B,Q,R)

# # K, P, eigenvalues = control.lqr(A, B, Q, R)
# # K, P, eigenvalues = control.dlqr(A_discrete, B_discrete, Q, R)