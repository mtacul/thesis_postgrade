# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 08:55:55 2024

@author: nachi
"""

from Suite_LQR import suite_sim
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from scipy.optimize import minimize

def save_res_txt(x,funi,file_name,text):
    with open(file_name, 'a') as f:
        # Escribir valores de x y resultados de la simulación
        f.write(f"x {text}: {x}\n")
        f.write(f"f {text}: {funi}\n")
        f.write(f"------------------------\n")
        
# Función para escribir en archivo .txt
def save_to_txt(x, acc, psd, time, pot_b, masa_b, vol_b, pot_ss, masa_ss, vol_ss, pot_act, masa_act, vol_act, P_i, funi,file_name):
    with open(file_name, 'a') as f:
        # Escribir valores de x y resultados de la simulación
        f.write(f"x: {x}\n")
        f.write(f"acc: {acc}, psd: {psd}, time: {time}\n")
        f.write(f"pot_b: {pot_b}, masa_b: {masa_b}, vol_b: {vol_b}\n")
        f.write(f"pot_ss: {pot_ss}, masa_ss: {masa_ss}, vol_ss: {vol_ss}\n")
        f.write(f"pot_act: {pot_act}, masa_act: {masa_act}, vol_act: {vol_act}\n")
        f.write(f"Pesos: {P_i}\n")
        f.write(f"funcion de costo: {funi}\n")
        f.write(f"------------------------\n")
        
        
# Definir la función objetivo con dos argumentos
def objective(x, *args):
    # Desempaquetar los argumentos adicionales
    type_act, S_A_both, type_rend,P_i,filename = args[:5]  
    
    # Determinar los valores de std_sensor_sol, std_magnetometros y lim según S_A_both
    if S_A_both == 0:
        std_sensor_sol, std_magnetometros = x
        P_i = [P_i[0],P_i[1],P_i[2],
                P_i[3],P_i[4],0, 
                P_i[6],P_i[7],0,   
                0,0,0]
        lim = args[5]
        
    elif S_A_both == 1:
        lim = x
        std_sensor_sol = args[5]
        std_magnetometros = args[6]  # Valores fijos o iniciales
        
        P_i = [P_i[0],P_i[1],P_i[2],
                0,0,0, 
                0,0,0,   
                P_i[9],P_i[10],0]
        
    elif S_A_both == 2:
        std_sensor_sol, std_magnetometros, lim = x
        P_i = P_i
    # Invocar la simulación según type_act
    acc, psd, time,pot_b, masa_b, vol_b,pot_ss, masa_ss, vol_ss, pot_act, masa_act, vol_act = suite_sim(std_sensor_sol, std_magnetometros, lim, type_act, S_A_both)
    

    # Seleccionar el valor a retornar según type_rend
    if type_rend == 'acc':   
        funi = P_i[0]*acc**2 + P_i[3]*pot_b**2 + P_i[4]*masa_b**2 + P_i[5]*vol_b**2 +  P_i[6]*pot_ss**2 + P_i[7]*masa_ss**2 + P_i[8]*vol_ss**2 + P_i[9]*pot_act**2 + P_i[10]*masa_act**2 + P_i[11]*vol_act**2 
        # print(time)
        # Guardar los resultados en un archivo .txt
        save_to_txt(x, acc, psd, time, pot_b, masa_b, vol_b, pot_ss, masa_ss, vol_ss, pot_act, masa_act, vol_act,P_i,funi,filename)
        return funi
    
    elif type_rend == 'psd':
        funi = P_i[1]*psd**2 + P_i[3]*pot_b**2 + P_i[4]*masa_b**2 + P_i[5]*vol_b**2 +  P_i[6]*pot_ss**2 + P_i[7]*masa_ss**2 + P_i[8]*vol_ss**2 + P_i[9]*pot_act**2 + P_i[10]*masa_act**2 + P_i[11]*vol_act**2
        # Guardar los resultados en un archivo .txt
        save_to_txt(x, acc, psd, time, pot_b, masa_b, vol_b, pot_ss, masa_ss, vol_ss, pot_act, masa_act, vol_act,P_i,funi,filename)
        return funi
      
    elif type_rend == 'time':
        funi = P_i[2]*time**2 + P_i[3]*pot_b**2 + P_i[4]*masa_b**2 + P_i[5]*vol_b**2 +  P_i[6]*pot_ss**2 + P_i[7]*masa_ss**2 + P_i[8]*vol_ss**2 + P_i[9]*pot_act**2 + P_i[10]*masa_act**2 + P_i[11]*vol_act**2
        # Guardar los resultados en un archivo .txt
        save_to_txt(x, acc, psd, time, pot_b, masa_b, vol_b, pot_ss, masa_ss, vol_ss, pot_act, masa_act, vol_act,P_i,funi,filename)
        return funi

    elif type_rend =='acc_time':
        funi = P_i[0]*acc**2 + P_i[2]*time**2 + P_i[3]*pot_b**2 + P_i[4]*masa_b**2 + P_i[5]*vol_b**2 +  P_i[6]*pot_ss**2 + P_i[7]*masa_ss**2 + P_i[8]*vol_ss**2 + P_i[9]*pot_act**2 + P_i[10]*masa_act**2 + P_i[11]*vol_act**2
        # Guardar los resultados en un archivo .txt
        save_to_txt(x, acc, psd, time, pot_b, masa_b, vol_b, pot_ss, masa_ss, vol_ss, pot_act, masa_act, vol_act,P_i,funi,filename)
        return funi

    elif type_rend =='acc_psd':
        funi = P_i[0]*acc**2 + P_i[1]*psd**2 + P_i[3]*pot_b**2 + P_i[4]*masa_b**2 + P_i[5]*vol_b**2 +  P_i[6]*pot_ss**2 + P_i[7]*masa_ss**2 + P_i[8]*vol_ss**2 + P_i[9]*pot_act**2 + P_i[10]*masa_act**2 + P_i[11]*vol_act**2
        # Guardar los resultados en un archivo .txt
        save_to_txt(x, acc, psd, time, pot_b, masa_b, vol_b, pot_ss, masa_ss, vol_ss, pot_act, masa_act, vol_act,P_i,funi,filename)
        return funi        

    elif type_rend =='psd_time':
        funi = P_i[1]*psd**2 + P_i[2]*time**2 + P_i[3]*pot_b**2 + P_i[4]*masa_b**2 + P_i[5]*vol_b**2 +  P_i[6]*pot_ss**2 + P_i[7]*masa_ss**2 + P_i[8]*vol_ss**2 + P_i[9]*pot_act**2 + P_i[10]*masa_act**2 + P_i[11]*vol_act**2
        # Guardar los resultados en un archivo .txt
        save_to_txt(x, acc, psd, time, pot_b, masa_b, vol_b, pot_ss, masa_ss, vol_ss, pot_act, masa_act, vol_act,P_i,funi,filename)
        return funi  

    elif type_rend =='all':
        funi = P_i[0]*acc**2 + P_i[1]*psd**2 + P_i[2]*time**2 + P_i[3]*pot_b**2 + P_i[4]*masa_b**2 + P_i[5]*vol_b**2 +  P_i[6]*pot_ss**2 + P_i[7]*masa_ss**2 + P_i[8]*vol_ss**2 + P_i[9]*pot_act**2 + P_i[10]*masa_act**2 + P_i[11]*vol_act**2
        # Guardar los resultados en un archivo .txt
        save_to_txt(x, acc, psd, time, pot_b, masa_b, vol_b, pot_ss, masa_ss, vol_ss, pot_act, masa_act, vol_act,P_i,funi,filename)
        return funi  
    
    
file_result = "optimizacion.txt"
   
# Definir valores de type_act, S_A_both, type_rend y Pesos P_i
type_act = 0  # 0: magnetorquer, 1: rueda de reacción
S_A_both = 0  # 0: solo sensor, 1: solo actuador, 2: ambos
type_rend = 'acc'  # Puede ser 'acc', 'psd', 'time', 'acc_time', 'acc_psd', 'psd_time' y 'all'

P_i = [1,1,1,   #Pesos: 0,1,2 -> acc,psd,time
        1,1,0,   #Pesos: 3,4,5 -> pot, masa y vol magnetometro
        1,1,0,   #Pesos: 6,7,8 -> pot, masa y vol sensor de sol
        1,1,0]   #Pesos: 9,10,11 -> pot, masa y vol actuador

filename = f"resultados_typeact{type_act}_saboth{S_A_both}_typerend{type_rend}.txt"
# x = [0.01,0.012e-9]
# x = 0.25
lim = 5
# std_ss,std_magn = [0.01,0.012e-9]

argss = (type_act, S_A_both, type_rend, P_i, filename,lim)
# argss = (type_act, S_A_both, type_rend, P_i, filename,std_ss,std_magn)
# argss = (type_act, S_A_both, type_rend, P_i, filename)

# f = objective(x,*argss)



# Definir los límites de los valores de std_sensor_sol y std_magnetometros
# bnds = ((0.01, 1.67), (0.012e-9, 3e-9), (0.29, 15))
bnds = ((0.01, 1.67), (0.012e-9, 3e-9))

# Crear una grilla de 20x20 en los rangos especificados
std_sensor_sol_vals = np.linspace(bnds[0][0], bnds[0][1], 20)
std_magnetometros_vals = np.linspace(bnds[1][0], bnds[1][1], 20)
# lim_vals = np.linspace(bnds[2][0], bnds[2][1], 6)

# Inicializar una matriz para almacenar los valores de f
f_values = np.zeros((20, 20))

# Inicializar una matriz 3D para almacenar los valores de f (20x20x20)
# f_values = np.zeros((20, 20, 20))

# Evaluar la función objetivo en cada punto de la grilla tridimensional
# for i, std_sensor_sol in enumerate(std_sensor_sol_vals):
#     for j, std_magnetometros in enumerate(std_magnetometros_vals):
#         for k, lim in enumerate(lim_vals):
#             x = [std_sensor_sol, std_magnetometros, lim]
#             f_values[i, j, k] = objective(x, *argss)  # `objective` es tu función objetivo

# Guardar los resultados en un archivo CSV
# np.savetxt("f_values_3D.csv", f_values.reshape(-1, f_values.shape[-1]), delimiter=",", fmt="%.8f")

print("Valores de la función guardados en 'f_values_3D.csv'")

# Evaluar la función objetivo en cada punto de la grilla
for i, std_sensor_sol in enumerate(std_sensor_sol_vals):
    for j, std_magnetometros in enumerate(std_magnetometros_vals):
        x = [std_sensor_sol, std_magnetometros]
        f_values[i, j] = objective(x, *argss)

np.savetxt("f_values.csv", f_values, delimiter=",", fmt="%.8f")

#%%
import plotly.graph_objects as go
import numpy as np
# Cargar el archivo CSV en una matriz (array)
# f_valuess = np.loadtxt('f_values.csv', delimiter=',')
f_valuess = np.loadtxt('f_values.csv', delimiter=',')

# Crear la grilla
std_sensor_sol_vals = np.linspace(0.01, 1.67, 20)
std_magnetometros_vals = np.linspace(0.012e-9, 3e-9, 20)
lim_vals = np.linspace(0.29, 70, 20)

X, Y = np.meshgrid(std_sensor_sol_vals, std_magnetometros_vals)
Z = f_valuess  # f_values es el resultado de tu función objetivo para cada punto de la grilla

# Crear la gráfica de superficie
# fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])
fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, cmin=np.min(Z), cmax=6)])

# Etiquetas de los ejes
fig.update_layout(scene=dict(
                    xaxis_title='std_sensor_sol',
                    yaxis_title='std_magnetometros',
                    zaxis_title='f(x)'),
                  title='Interactive 3D Surface Plot')

# Guardar como archivo HTML
fig.write_html("grafico_interactivo.html")

# Imprimir un mensaje para abrir el archivo
print("Gráfico guardado como 'grafico_interactivo.html'. Ábrelo en tu navegador.")

#%%
# import plotly.graph_objects as go
# import numpy as np

# # Cargar el archivo CSV en una matriz (array 3D)
# f_valuess = np.loadtxt('f_values_3D.csv', delimiter=',')

# # Crear las grillas 3D
# std_sensor_sol_vals = np.linspace(0.01, 1.67, 20)
# std_magnetometros_vals = np.linspace(0.012e-9, 3e-9, 20)
# lim_vals = np.linspace(0.29, 70, 6)

# # Crear la grilla 3D con 'np.meshgrid'
# X, Y, Z = np.meshgrid(std_sensor_sol_vals, std_magnetometros_vals, lim_vals, indexing='ij')

# # Aplanar las matrices para tener los puntos en forma de lista
# X_flat = X.flatten()
# Y_flat = Y.flatten()
# Z_flat = Z.flatten()
# f_flat = f_valuess.flatten()

# # Definir los límites para la barra de colores
# color_min = np.min(f_flat)  # O establece un valor personalizado
# color_max = 6  # O establece un valor personalizado

# # Crear una lista de texto que muestre la información para el hover
# hover_text = [f"std_sensor_sol: {x:.2f}<br>std_magnetometros: {y:.2e}<br>lim: {z:.2f}<br>f(x): {f:.4f}" 
#               for x, y, z, f in zip(X_flat, Y_flat, Z_flat, f_flat)]

# # Crear una gráfica 3D interactiva de tipo 'Scatter3d' para los puntos de datos, con color basado en f(x)
# fig = go.Figure(data=[go.Scatter3d(
#     x=X_flat,  # Eje X: std_sensor_sol
#     y=Y_flat,  # Eje Y: std_magnetometros
#     z=Z_flat,  # Eje Z: lim
#     mode='markers',
#     marker=dict(
#         size=4,
#         color=f_flat,  # Cuarta dimensión: valor de f(x) representado con colores
#         colorscale='Viridis',  # Escala de colores para f(x)
#         colorbar=dict(title="f(x)"),  # Barra de color que representa f(x)
#         opacity=0.8,
#         cmin=color_min,  # Límite inferior para la barra de colores
#         cmax=color_max   # Límite superior para la barra de colores
#     ),
#     text=hover_text,  # Usar el texto creado para el hover
#     hoverinfo='text'  # Mostrar solo el texto definido
# )])

# # Actualizar las etiquetas de los ejes y el título
# fig.update_layout(scene=dict(
#                     xaxis_title='std_sensor_sol',
#                     yaxis_title='std_magnetometros',
#                     zaxis_title='lim'),
#                   title='Interactive 3D Scatter Plot with 4th Dimension (f(x) Color)')

# # Guardar como archivo HTML
# fig.write_html("grafico_interactivo_4D.html")

# # Imprimir un mensaje para abrir el archivo
# print("Gráfico 3D con 4ta dimensión guardado como 'grafico_interactivo_4D.html'. Ábrelo en tu navegador.")
