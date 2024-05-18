# -*- coding: utf-8 -*-
"""
Created on Fri May 17 21:07:38 2024

@author: nachi
"""
import functions_03
import numpy as np
import control as ctrl
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

# %%
archivo_csv = "Vectores_orbit_ECI.csv"

# Leer el archivo CSV en un DataFrame de pandas
df = pd.read_csv(archivo_csv)

# Convertir el DataFrame a un array de NumPy
array_datos = df.to_numpy()

t_aux = array_datos[:, 0]
Bx_orbit = array_datos[:, 1]
By_orbit = array_datos[:, 2]
Bz_orbit = array_datos[:, 3]
Bx_IGRF = array_datos[:, 4]
By_IGRF = array_datos[:, 5]
Bz_IGRF = array_datos[:, 6]
vx_sun_orbit = array_datos[:, 7]
vy_sun_orbit = array_datos[:, 8]
vz_sun_orbit = array_datos[:, 9]
vsun_x = array_datos[:, 10]
vsun_y = array_datos[:, 11]
vsun_z = array_datos[:, 12]
#%%
w0_O = 0.00163
# 5600 aprox es una orbita
deltat = 2
limite =  5602
t = np.arange(0, limite, deltat)

I_x = 0.037
I_y = 0.036
I_z = 0.006

w0_eq = 0
w1_eq = 0
w2_eq = 0

sigma_ss = 0.036
sigma_b = 1e-6

#%%
q= np.array([0,0.7071,0,0.7071])
# q= np.array([0,0,0,1])
w = np.array([0.0001, 0.0001, 0.0001])
# q_est = np.array([0.00985969, 0.70703804, 0.00985969, 0.70703804])


q0_real = [q[0]]
q1_real = [q[1]]
q2_real = [q[2]]
q3_real = [q[3]]
w0_real = [w[0]]
w1_real = [w[1]]
w2_real = [w[2]]
q_real = np.array([q0_real[-1],q1_real[-1],q2_real[-1],q3_real[-1]])
w_body = np.array([w0_real[-1], w1_real[-1], w2_real[-1]])
w_gyros = functions_03.simulate_gyros_reading(w_body, 0,0)
x_real = np.hstack((np.transpose(q_real[:3]), np.transpose(w_gyros)))

bi_orbit = [Bx_orbit[0],By_orbit[0],Bz_orbit[0]]
b_body_i = functions_03.rotacion_v(q_real, bi_orbit, 1e-6)

si_orbit = [vx_sun_orbit[0],vy_sun_orbit[0],vz_sun_orbit[0]]
s_body_i = functions_03.rotacion_v(q_real, si_orbit, 0.036)
hh =0.01

[A,B,C,A_discrete,B_discrete,C_discrete] = functions_03.A_B(I_x, I_y, I_z, w0_O, w0_eq, w1_eq, w2_eq, deltat, hh, b_body_i, s_body_i)


# x0 = np.array([ -10 ,   20, -30,  -20,
#   -40, -100])

# optimal_x = functions_03.opt_K(A_discrete, B_discrete, deltat, hh, x0)
# K = np.hstack([np.diag(optimal_x[:3]), np.diag(optimal_x[3:])])

B_matrices = [B_discrete]
#%%
np.random.seed(42)
for i in range(len(t)-1):
    print(t[i+1])
    
    u_est = np.array([15,15,15])
    # u_est = np.dot(K,x_real)

    b_orbit = [Bx_orbit[i+1],By_orbit[i+1],Bz_orbit[i+1]]
    b_body = functions_03.rotacion_v(q_real, b_orbit, sigma_b)

    vsun_orbit = [vx_sun_orbit[i+1],vy_sun_orbit[i+1],vz_sun_orbit[i+1]]
    s_body = functions_03.rotacion_v(q_real, vsun_orbit, sigma_ss)


    [A,B,C,A_discrete,B_discrete,C_discrete] = functions_03.A_B(I_x, I_y, I_z, w0_O, w0_eq, w1_eq, w2_eq, deltat, hh, b_body, s_body)
    # asasd
    B_matrices.append(B_discrete)
    
    [xx_new_d, qq3_new_d] = functions_03.mod_lineal_disc(
        x_real, u_est, deltat, hh, A_discrete,B_discrete)
    
    # optimal_x = functions_03.opt_K(A_discrete, B_discrete, deltat, hh, x0)
    # K = np.hstack([np.diag(optimal_x[:3]), np.diag(optimal_x[3:])])
    
    x_real = xx_new_d

    q0_real.append(xx_new_d[0])
    q1_real.append(xx_new_d[1])
    q2_real.append(xx_new_d[2])
    q3_real.append(qq3_new_d)
    w0_real.append(xx_new_d[3])
    w1_real.append(xx_new_d[4])
    w2_real.append(xx_new_d[5])

    # q_real = np.array([q0_real[-1], q1_real[-1], q2_real[-1], q3_real[-1]])
    # www_real = np.array([w0_real[-1], w1_real[-1], w2_real[-1]])
    # ww_real = functions_03.simulate_gyros_reading(www_real, 0,0)

Bs_a = np.array(B_matrices)

#%%

import numpy as np
from scipy.linalg import solve_discrete_are

# Define matrix A
A = np.array([
    [1.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e-02, 0.00000000e+00, 0.00000000e+00],
    [0.00000000e+00, 1.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e-02, 1.46229111e-07],
    [0.00000000e+00, 8.97330778e-17, 1.00000000e+00, 0.00000000e+00, -2.81446667e-08, 1.00000000e-02],
    [3.18828000e-10, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00, 0.00000000e+00],
    [0.00000000e+00, -9.56484000e-09, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 2.92458222e-05],
    [0.00000000e+00, 2.69199234e-14, 0.00000000e+00, 0.00000000e+00, -5.62893333e-06, 1.00000000e+00]
])

# Define weight matrices for LQR
Q = np.eye(A.shape[0])
R = np.eye(3)  # Assuming B has 3 columns, matching dimensions

# Define time intervals and corresponding B matrices
time_intervals = np.arange(0, 5600 + 2, 2)  # Intervals from 0 to 5600 seconds with a step of 2 seconds

# Calculate K for each interval
K_matrices = []
for B in B_matrices:
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P
    K_matrices.append(K)

def piecewise_controller(x, current_time, time_intervals, K_matrices):
    # Determine the current interval
    interval_index = np.searchsorted(time_intervals, current_time) - 1
    if interval_index < 0:
        interval_index = 0
    elif interval_index >= len(K_matrices):
        interval_index = len(K_matrices) - 1
    # Use the corresponding K matrix
    K_current = K_matrices[interval_index]
    return -K_current @ x

# Example function to get current time
def get_current_time():
    # In a real application, replace with actual time retrieval logic
    return np.random.choice(time_intervals)  # Replace with actual time retrieval

# Example control loop
state = np.random.rand(6)  # Replace with actual state
current_time = get_current_time()
control_signal = piecewise_controller(state, current_time, time_intervals, K_matrices)
print(control_signal)

    
# %%
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
