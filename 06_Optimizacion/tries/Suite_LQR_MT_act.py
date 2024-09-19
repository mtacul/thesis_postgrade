# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 22:09:38 2024

@author: nachi
"""
import functions_06
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_are
from scipy.signal import welch
#%%

def suite_act(lim):
    
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
    
    deltat = 2
    # limite =  5762*69
    limite =  5762*4
    t = np.arange(0, limite, deltat)
    
    #%% Parámetros geométricos y orbitales dados
    
    w0_O = 0.00163
    
    I_x = 0.037
    I_y = 0.036
    I_z = 0.006
    
    #%% Caracteristicas del sensor
    sigma_ss = 0.1
    sigma_ss = np.sin(sigma_ss*np.pi/180)
    sigma_b = 1e-9

#%% Condiciones iniciales reales y estimadas

    # q= np.array([0,0.7071,0,0.7071])
    # q= np.array([0,0,0,1])
    q = np.array([0.0789,0.0941,0.0789,0.9893])
    w = np.array([0.0001, 0.0001, 0.0001])
    # q_est = np.array([0.00985969, 0.70703804, 0.00985969, 0.70703804])
    # q_est= np.array([0.0120039,0.0116517,0.0160542,0.999731])
    q_est = np.array([0.0789,0.0941,0.0789,0.9893])
    
    q0_est = [q_est[0]]
    q1_est = [q_est[1]]
    q2_est = [q_est[2]]
    q3_est = [q_est[3]]
    w0_est = [w[0]]
    w1_est = [w[1]]
    w2_est = [w[2]]
    
    q0_real = [q[0]]
    q1_real = [q[1]]
    q2_real = [q[2]]
    q3_real = [q[3]]
    w0_real = [w[0]]
    w1_real = [w[1]]
    w2_real = [w[2]]
    q_real = np.array([q0_real[-1],q1_real[-1],q2_real[-1],q3_real[-1]])
    w_body = np.array([w0_real[-1], w1_real[-1], w2_real[-1]])
    w_gyros = functions_06.simulate_gyros_reading(w_body, 0,0)
    x_real = np.hstack((np.transpose(q_real[:3]), np.transpose(w_gyros)))
    
    bi_orbit = [Bx_orbit[0],By_orbit[0],Bz_orbit[0]]
    b_body_i = functions_06.rotacion_v(q_real, bi_orbit, 1e-6)
    
    si_orbit = [vx_sun_orbit[0],vy_sun_orbit[0],vz_sun_orbit[0]]
    s_body_i = functions_06.rotacion_v(q_real, si_orbit, 0.036)
    hh =0.01
    # print('1')
    #%% Obtencion de un B_prom representativo
    
    [A,B,C,A_discrete,B_discrete,C_discrete] = functions_06.A_B(I_x, I_y, I_z, w0_O, 0, 0, 0, deltat, hh, bi_orbit,b_body_i, s_body_i)
    B_matrices = [B_discrete]
    
    for i in range(len(t[0:2882])-1):
    
        # print(t[i+1])
            
        b_orbit = [Bx_orbit[i+1],By_orbit[i+1],Bz_orbit[i+1]]
    
        [A,B,C,A_discrete,B_discrete,C_discrete] = functions_06.A_B(I_x, I_y, I_z, w0_O, 0, 0, 0, deltat, hh,b_orbit, np.zeros(3), np.zeros(3))
    
        B_matrices.append(B_discrete)
    
    Bs_a = np.array(B_matrices)
    
    B_concanate = []    
    for ii in range(len(Bs_a[0,:,0])):
        for jj in range(len(Bs_a[0,0,:])):
            B_concanate.append(np.sum(Bs_a[:,ii,jj]) / len(Bs_a[:,ii,jj]))
    
    B_prom = np.vstack((B_concanate[0:3],B_concanate[3:6],B_concanate[6:9],B_concanate[9:12],B_concanate[12:15],B_concanate[15:18]))

    #%% Control LQR
    
    # Definir las matrices Q y R del coste del LQR
    # diag_Q = np.array([100, 1000000, 10000, 0.1, 0.1, 0.10, 0.01, 10, 10])*10000
    # diag_R = np.array([0.1,0.1,0.1])*100000
    diag_Q = np.array([100, 10, 100, 0.1, 0.1, 0.1])*1000000
    diag_R = np.array([0.1,0.1,0.1])*100000
    
    Q = np.diag(diag_Q)
    R = np.diag(diag_R)
    
    # Resolver la ecuación de Riccati
    P = solve_discrete_are(A_discrete, B_prom, Q, R)
    
    # Calcular la matriz de retroalimentación K
    K = np.linalg.inv(B_prom.T @ P @ B_prom + R) @ (B_prom.T @ P @ A_discrete)
    
    #%% Simulacion dinamica de actitud
    
    diagonal_values = np.array([0.5**2, 0.5**2, 0.5**2, 0.1**2, 0.1**2, 0.1**2])
    P_ki = np.diag(diagonal_values)
    np.random.seed(42)
    
    for i in range(len(t)-1):
        # print(t[i+1])
        q_est = np.array([q0_est[-1], q1_est[-1], q2_est[-1], q3_est[-1]])
        w_est = np.array([w0_est[-1], w1_est[-1], w2_est[-1]])
        x_est = np.hstack((np.transpose(q_est[:3]), np.transpose(w_est)))
        u_est = np.dot(-K,x_est)
        u_est = functions_06.torquer(u_est,lim)
    
        [xx_new_d, qq3_new_d] = functions_06.mod_lineal_disc(
            x_real, u_est, deltat, hh, A_discrete,B_prom)
        
        x_real = xx_new_d
        w_gyros = functions_06.simulate_gyros_reading(x_real[3:6],0,0)
        q0_real.append(xx_new_d[0])
        q1_real.append(xx_new_d[1])
        q2_real.append(xx_new_d[2])
        q3_real.append(qq3_new_d)
        w0_real.append(w_gyros[0])
        w1_real.append(w_gyros[1])
        w2_real.append(w_gyros[2])
    
        q_real = np.array([q0_real[-1], q1_real[-1], q2_real[-1], q3_real[-1]])
    
        b_orbit = [Bx_orbit[i+1],By_orbit[i+1],Bz_orbit[i+1]]
        b_body_med = functions_06.rotacion_v(q_real, b_orbit,sigma_b)
        
        s_orbit = [vx_sun_orbit[i+1],vy_sun_orbit[i+1],vz_sun_orbit[i+1]]
        s_body_med = functions_06.rotacion_v(q_real, s_orbit,sigma_ss)
    
        [A,B,C,A_discrete,B_discrete,C_discrete] = functions_06.A_B(I_x, I_y, I_z, w0_O, 0, 0, 0, deltat, hh,b_orbit, b_body_med, s_body_med)
        
        if sigma_ss == 0 or sigma_b ==0:
            [q_posteriori, w_posteriori, P_k_pos,K_k] = functions_06.kalman_lineal(A_discrete, B_prom,C_discrete, x_est, u_est, b_orbit,b_body_med, s_orbit,s_body_med, P_ki, sigma_b,sigma_ss, deltat,hh,1, 1)
        else:
            [q_posteriori, w_posteriori, P_k_pos,K_k] = functions_06.kalman_lineal(A_discrete, B_prom,C_discrete, x_est, u_est, b_orbit,b_body_med, s_orbit,s_body_med, P_ki, sigma_b,sigma_ss, deltat,hh,sigma_b,sigma_ss)
    
        q0_est.append(q_posteriori[0])
        q1_est.append(q_posteriori[1])
        q2_est.append(q_posteriori[2])
        q3_est.append(q_posteriori[3])
        w0_est.append(w_posteriori[0])
        w1_est.append(w_posteriori[1])
        w2_est.append(w_posteriori[2])
    
        P_ki = P_k_pos
        
    [MSE_cuat, MSE_omega]  = functions_06.cuat_MSE_NL(q0_real, q1_real, q2_real, q3_real, w0_real, w1_real, w2_real, q0_est, q1_est, q2_est, q3_est, w0_est, w1_est, w2_est)   
    [RPY_all_est,RPY_all_id,mse_roll,mse_pitch,mse_yaw] = functions_06.RPY_MSE(t, q0_est, q1_est, q2_est, q3_est, q0_real, q1_real, q2_real, q3_real)   
          
    #%% Tratamiento de datos
    
    Roll_low_pass_1 = RPY_all_id[:,0]
    Pitch_low_pass_1 = RPY_all_id[:,1]
    Yaw_low_pass_1 = RPY_all_id[:,2]
    Roll_1 = RPY_all_est[:,0]
    Pitch_1 = RPY_all_est[:,1]
    Yaw_1 = RPY_all_est[:,2]
    RPY_1 = np.transpose(np.vstack((Roll_1,Pitch_1,Yaw_1)))
    norms_RPY_1 = []
    for j in range(len(RPY_1)):
        norm_RPY_1 = np.linalg.norm(RPY_1[j,:])
        norms_RPY_1.append(norm_RPY_1)
    norms_RPY_1 = np.array(norms_RPY_1)
    RPY_low_pass_1 = np.transpose(np.vstack((Roll_low_pass_1,Pitch_low_pass_1,Yaw_low_pass_1)))
    norms_RPY_low_pass_1 = []
    for j in range(len(RPY_low_pass_1)):
        norm_RPY_low_pass_1 = np.linalg.norm(RPY_low_pass_1[j,:])
        norms_RPY_low_pass_1.append(norm_RPY_low_pass_1)
    norms_RPY_low_pass_1 = np.array(norms_RPY_low_pass_1)
    
    #%% Calculo del Jitter
    
    Roll_high_pass_1 = functions_06.high_pass_filter(Roll_1, 10, len(t))
    Pitch_high_pass_1 = functions_06.high_pass_filter(Pitch_1, 10, len(t))
    Yaw_high_pass_1 = functions_06.high_pass_filter(Yaw_1, 10, len(t))
    
    frequencies_R_1, psd_R_1 = welch(Roll_high_pass_1, len(t), nperseg=1024)
    frequencies_P_1, psd_P_1 = welch(Pitch_high_pass_1, len(t), nperseg=1024)
    frequencies_Y_1, psd_Y_1 = welch(Yaw_high_pass_1, len(t), nperseg=1024)
    
    # psd_R_R_1 =[]
    # psd_P_R_1 =[]
    # psd_Y_R_1 =[]
    
    # for i in range(len(frequencies_R_1)):
    #     psd_R_r_1 = np.real(psd_R_1[i])
    #     psd_P_r_1 = np.real(psd_P_1[i])
    #     psd_Y_r_1 = np.real(psd_Y_1[i])
    #     psd_R_R_1.append(psd_R_r_1)
    #     psd_P_R_1.append(psd_P_r_1)
    #     psd_Y_R_1.append(psd_Y_r_1)
    
    # psd_R_R_1 = np.array(psd_R_R_1)
    # psd_P_R_1 = np.array(psd_P_R_1)
    # psd_Y_R_1 = np.array(psd_Y_R_1)
    
    # Definir los anchos de banda deseados
    bandwidth_1 = (0, 10000)  # Ancho de banda 1 en Hz
    
    # Calcular la PSD dentro de los anchos de banda específicos
    indices_bandwidth_1_R = np.where((frequencies_R_1 >= bandwidth_1[0]) & (frequencies_R_1 <= bandwidth_1[1]))
    psd_bandwidth_1_R = np.trapz(psd_R_1[indices_bandwidth_1_R], frequencies_R_1[indices_bandwidth_1_R])
    
    indices_bandwidth_1_P = np.where((frequencies_P_1 >= bandwidth_1[0]) & (frequencies_P_1 <= bandwidth_1[1]))
    psd_bandwidth_1_P = np.trapz(psd_P_1[indices_bandwidth_1_P], frequencies_P_1[indices_bandwidth_1_P])
    
    indices_bandwidth_1_Y = np.where((frequencies_Y_1 >= bandwidth_1[0]) & (frequencies_Y_1 <= bandwidth_1[1]))
    psd_bandwidth_1_Y = np.trapz(psd_Y_1[indices_bandwidth_1_Y], frequencies_Y_1[indices_bandwidth_1_Y])
    
    psd_RPY_1 = np.array([psd_bandwidth_1_R,psd_bandwidth_1_P,psd_bandwidth_1_Y])
    norm_psd_RPY_1 = np.linalg.norm(psd_RPY_1)

    #%% Calculo del tiempo de asentamiento

    settling_band_R = 7
    settling_band_P = 7
    settling_band_Y = 7

    settling_error_sup_R = np.full(len(t_aux),settling_band_R)
    settling_error_inf_R = np.full(len(t_aux),-settling_band_R)

    settling_error_sup_P = np.full(len(t_aux),settling_band_P)
    settling_error_inf_P = np.full(len(t_aux),-settling_band_P)

    settling_error_sup_Y = np.full(len(t_aux),settling_band_Y)
    settling_error_inf_Y = np.full(len(t_aux),-settling_band_Y)

    settling_time_indices_R_1 = []
    start_index_R_1 = None
    settling_time_indices_P_1 = []
    start_index_P_1 = None
    settling_time_indices_Y_1 = []
    start_index_Y_1 = None


    for i in range(len(Roll_low_pass_1)):
        if Roll_low_pass_1[i] <= settling_error_sup_R[i] and Roll_low_pass_1[i] >= settling_error_inf_R[i]:
            if start_index_R_1 is None:
                start_index_R_1 = i
        else:
            if start_index_R_1 is not None:
                settling_time_indices_R_1.append((start_index_R_1, i - 1))
                start_index_R_1 = None

    if start_index_R_1 is not None:
        settling_time_indices_R_1.append((start_index_R_1, len(Roll_low_pass_1) - 1))

    if settling_time_indices_R_1:
        settling_times_R_1 = []
        for start, end in settling_time_indices_R_1:
            settling_times_R_1.append((t_aux[start], t_aux[end]))
    else:
        print("La señal no entra en la banda de asentamiento.")
        

    for i in range(len(Pitch_low_pass_1)):
        if Pitch_low_pass_1[i] <= settling_error_sup_P[i] and Pitch_low_pass_1[i] >= settling_error_inf_P[i]:
            if start_index_P_1 is None:
                start_index_P_1 = i
        else:
            if start_index_P_1 is not None:
                settling_time_indices_P_1.append((start_index_P_1, i - 1))
                start_index_P_1 = None

    if start_index_P_1 is not None:
        settling_time_indices_P_1.append((start_index_P_1, len(Pitch_low_pass_1) - 1))

    if settling_time_indices_P_1:
        settling_times_P_1 = []
        for start, end in settling_time_indices_P_1:
            settling_times_P_1.append((t_aux[start], t_aux[end]))
    else:
        print("La señal no entra en la banda de asentamiento.")


    for i in range(len(Yaw_low_pass_1)):
        if Yaw_low_pass_1[i] <= settling_error_sup_Y[i] and Yaw_low_pass_1[i] >= settling_error_inf_Y[i]:
            if start_index_Y_1 is None:
                start_index_Y_1 = i
        else:
            if start_index_Y_1 is not None:
                settling_time_indices_Y_1.append((start_index_Y_1, i - 1))
                start_index_Y_1 = None

    if start_index_Y_1 is not None:
        settling_time_indices_Y_1.append((start_index_Y_1, len(Yaw_low_pass_1) - 1))

    if settling_time_indices_Y_1:
        settling_times_Y_1 = []
        for start, end in settling_time_indices_Y_1:
            settling_times_Y_1.append((t_aux[start], t_aux[end]))
    else:
        print("La señal no entra en la banda de asentamiento.")


    settling_time_indices_norm_1 = []
    start_index_norm_1 = None
    upper_limit = settling_band_R
    lower_limit = -settling_band_R

    for i in range(len(norms_RPY_low_pass_1)):
        if lower_limit <= norms_RPY_low_pass_1[i] <= upper_limit:
            if start_index_norm_1 is None:
                start_index_norm_1 = i
        else:
            if start_index_norm_1 is not None:
                settling_time_indices_norm_1.append((start_index_norm_1, i - 1))
                start_index_norm_1 = None

    if start_index_norm_1 is not None:
        settling_time_indices_norm_1.append((start_index_norm_1, len(norms_RPY_low_pass_1) - 1))

    if settling_time_indices_norm_1:
        settling_times_norm_1 = []
        for start, end in settling_time_indices_norm_1:
            settling_times_norm_1.append((t_aux[start], t_aux[end]))
    else:
        print("La señal no entra en la banda de asentamiento.")
        
    #%% Exactitud de apuntamiento

    time_R_1 = np.array(settling_times_R_1[-1])
    data_R_1 = Roll_1[int(time_R_1[0]/2):int(time_R_1[1]/2)]
    sigma_R_1 = np.std(data_R_1)
    accuracy_R_1 = 3*sigma_R_1

    time_P_1 = np.array(settling_times_P_1[-1])
    data_P_1 = Pitch_1[int(time_P_1[0]/2):int(time_P_1[1]/2)]
    sigma_P_1 = np.std(data_P_1)
    accuracy_P_1 = 3*sigma_P_1

    time_Y_1 = np.array(settling_times_Y_1[-1])
    data_Y_1= Yaw_1[int(time_Y_1[0]/2):int(time_Y_1[1]/2)]
    sigma_Y_1 = np.std(data_Y_1)
    accuracy_Y_1 = 3*sigma_Y_1

    accuracy_RPY_1 = np.array([accuracy_R_1,accuracy_P_1,accuracy_Y_1])
    
    # normas exactitud de apuntamiento
    ti_1 = np.array([time_R_1[0],time_P_1[0],time_Y_1[0]])
    norm_time = np.linalg.norm(ti_1)
    time_norm_1 = np.array([np.linalg.norm(ti_1),t_aux[-1]])
    data_norm_1= norms_RPY_1[int(time_norm_1[0]/2):int(time_norm_1[1]/2)]

    # Calcular media y desviación estándar
    sigma_norm_1 = np.std(data_norm_1)
    accuracy_norm_1 = 3*sigma_norm_1
    accuracy_norms = np.array([accuracy_norm_1]) 
    
    # print("El tiempo de asentamiento en RPY es:",time_R_1[0],"[s]","\n",time_P_1[0],"[s]","\n",time_Y_1[0],"[s]")
    #%% Graficas

    fig0, ax = plt.subplots(figsize=(15,5))  # Crea un solo set de ejes
    
    # Graficar los tres conjuntos de datos en la misma gráfica
    ax.plot(t, norms_RPY_1, label='magnetorquer')
    
    # Configurar etiquetas, leyenda y grid
    ax.set_xlabel('Tiempo [s]', fontsize=18)
    ax.set_ylabel('Error en ángulo de orientación [°]', fontsize=18)
    ax.legend(fontsize=18)
    ax.grid()
    
    # Ajustar límites del eje X
    ax.set_xlim(0, 30000)
    
    # Ajustar el tamaño de las etiquetas de los ticks
    ax.tick_params(axis='both', which='major', labelsize=18)
    
    plt.tight_layout()
    
    # Guardar la gráfica como archivo SVG
    # plt.savefig('norm2.svg', format='svg')
    
    # Mostrar la gráfica
    plt.show()
    
    fig0, axes0 = plt.subplots(nrows=3, ncols=1, figsize=(13, 8))

    axes0[0].plot(t, Roll_1, label= {'magnetorquer'})
    axes0[0].set_xlabel('Tiempo [s]')
    axes0[0].set_ylabel('Roll [°]')
    axes0[0].legend()
    axes0[0].grid()
    axes0[0].set_xlim(0, 30000)  # Ajusta los límites en el eje Y

    axes0[1].plot(t, Pitch_1, label={'magnetorquer'},color='orange')
    axes0[1].set_xlabel('Tiempo [s]')
    axes0[1].set_ylabel('Pitch [°]')
    axes0[1].legend()
    axes0[1].grid()
    axes0[1].set_xlim(0, 30000) # Ajusta los límites en el eje Y
    # axes0[1].set_ylim(-20, -5)  # Ajusta los límites en el eje Y
    
    axes0[2].plot(t, Yaw_1, label={'magnetorquer'},color='green')
    axes0[2].set_xlabel('Tiempo [s]')
    axes0[2].set_ylabel('Yaw [°]')
    axes0[2].legend()
    axes0[2].grid()
    axes0[2].set_xlim(0, 30000)  # Ajusta los límites en el eje Y

    plt.tight_layout()
    
    # Save the plot as an SVG file
    # plt.savefig('plot.svg', format='svg')

    # Show the plot (optional)
    plt.show()
    
    fig0, axes0 = plt.subplots(nrows=3, ncols=1, figsize=(13, 8))

    axes0[0].plot(t, Roll_low_pass_1, label= {'magnetorquer'})
    axes0[0].set_xlabel('Tiempo [s]')
    axes0[0].set_ylabel('Roll [°]')
    axes0[0].legend()
    axes0[0].grid()
    axes0[0].set_xlim(0, 30000)  # Ajusta los límites en el eje Y

    axes0[1].plot(t, Pitch_low_pass_1, label={'magnetorquer'},color='orange')
    axes0[1].set_xlabel('Tiempo [s]')
    axes0[1].set_ylabel('Pitch [°]')
    axes0[1].legend()
    axes0[1].grid()
    axes0[1].set_xlim(0, 30000) # Ajusta los límites en el eje Y
    # axes0[1].set_ylim(-20, -5)  # Ajusta los límites en el eje Y
    
    axes0[2].plot(t, Yaw_low_pass_1, label={'magnetorquer'},color='green')
    axes0[2].set_xlabel('Tiempo [s]')
    axes0[2].set_ylabel('Yaw [°]')
    axes0[2].legend()
    axes0[2].grid()
    axes0[2].set_xlim(0, 30000)  # Ajusta los límites en el eje Y

    plt.tight_layout()
    
    # Save the plot as an SVG file
    # plt.savefig('plot.svg', format='svg')

    # Show the plot (optional)
    plt.show()
    
    return time_R_1[0], time_P_1[0], time_Y_1[0]