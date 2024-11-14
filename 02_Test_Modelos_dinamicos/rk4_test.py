# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 22:38:17 2024

@author: nachi
"""


import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

A = np.array([
    [0, 0, 0, 0.5, 0, 0],
    [0, 0, 0, 0, 0.5, 0],
    [0, 0, 0, 0, 0, 0.5],
    [-4.78242e-07, 0, 0, 0, 0, 0],
    [0, -4.94183e-07, 0, 0, 0, 0.00303361],
    [0, 0, 0, 0, -0.00135833, 0]
])

def system(t, x):
    return A @ x

x0 = np.array([0, 0, 0, 0.001, 0, 0])  

t_span = (0, 10000)
solution = solve_ivp(system, t_span, x0, method='RK45', t_eval=np.linspace(0, 10000, 1000000))

plt.figure(figsize=(10, 6))
for i in range(6):
    plt.plot(solution.t, solution.y[i], label=f'x{i+1}')
plt.xlabel('Tiempo')
plt.ylabel('Valores de x')
# plt.xlim(0,100)
plt.title('Solución de $\dot{x} = A \cdot x$')
plt.legend()
plt.grid()
plt.show()

#%%

# funcion de la ecuacion xDot = Ax + Bu 
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

    return q_new_real, w_new

# def mod_lineal_cont(x,u,h,A,B):
#     q_rot,w_new = rk4_step_PD(dynamics, x, A, B, u, h)
#     x_new = np.hstack((np.transpose(q_rot), np.transpose(w_new)))
#     q3s_rot = np.sqrt(1-q_rot[0]**2-q_rot[1]**2-q_rot[2]**2)
                
#     return x_new, q3s_rot

def mod_lineal_cont(x,u,deltat,h,A,B):
    x_new = x
    for j in range(int(deltat/h)):
        q_rot,w_new = rk4_step_PD(dynamics, x_new, A, B, u, h)
        if  1-q_rot[0]**2-q_rot[1]**2-q_rot[2]**2 < 0:
            q_rot = q_rot / np.linalg.norm(q_rot)
            x_new = np.hstack((np.transpose(q_rot), np.transpose(w_new)))
            q3s_rot = 0

        else:
            x_new = np.hstack((np.transpose(q_rot), np.transpose(w_new)))
            q3s_rot = np.sqrt(1-q_rot[0]**2-q_rot[1]**2-q_rot[2]**2)
                
    return x_new, q3s_rot

deltat = 1
hh = 0.1
B = np.zeros((6,3))
t = np.arange(0, t_span[1], deltat)

x0_q3 = np.sqrt(1-x0[0]**2-x0[1]**2-x0[2]**2)

q0 = [x0[0]]
q1 = [x0[1]]
q2 = [x0[2]]
q3 = [x0_q3]
w0 = [x0[3]]
w1 = [x0[4]]
w2 = [x0[5]]

for i in range(len(t)-1):
    qq = np.array([q0[-1],q1[-1],q2[-1],q3[-1]])
    ww = np.array([w0[-1],w1[-1],w2[-1]])
    xx = np.hstack((np.transpose(qq[:3]), np.transpose(ww)))
    uu = np.array([0,0,0])
    [xx_new, qq3_new] = mod_lineal_cont(xx, uu, deltat,hh, A, B)
    
    q0.append(xx_new[0])
    q1.append(xx_new[1])
    q2.append(xx_new[2])
    q3.append(qq3_new)
    w0.append(xx_new[3])
    w1.append(xx_new[4])
    w2.append(xx_new[5])
    
fig0, axes0 = plt.subplots(nrows=1, ncols=2, figsize=(16, 4))

axes0[0].plot(t[0:len(q0)], q0[0:len(q0)], label='q0 lineal')
axes0[0].plot(t[0:len(q0)], q1[0:len(q0)], label='q1 lineal')
axes0[0].plot(t[0:len(q0)], q2[0:len(q0)], label='q2 lineal')
axes0[0].plot(t[0:len(q0)], q3[0:len(q0)], label='q3 lineal')
axes0[0].set_xlabel('Tiempo [s]')
axes0[0].set_ylabel('cuaternión [-]')
axes0[0].legend()
axes0[0].set_title('cuaterniones lineales continuos MT')
axes0[0].grid()
# axes0[0].set_xlim(0, 2)  # Ajusta los límites en el eje x

axes0[1].plot(t[0:len(q0)], w0[0:len(q0)], label='w0 lineal')
axes0[1].plot(t[0:len(q0)], w1[0:len(q0)], label='w1 lineal')
axes0[1].plot(t[0:len(q0)], w2[0:len(q0)], label='w2 lineal')
axes0[1].set_xlabel('Tiempo [s]')
axes0[1].set_ylabel('velocidad angular [rad/s]')
axes0[1].legend()
axes0[1].set_title('velocidades angulares lineales continuos MT')
axes0[1].grid()
# axes0[1].set_xlim(0, 2)  # Ajusta los límites en el eje x

plt.tight_layout()
plt.show()