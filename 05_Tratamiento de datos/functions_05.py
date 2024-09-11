import numpy as np
import math
from scipy.signal import butter, lfilter


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