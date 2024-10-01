# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 02:17:44 2024

@author: nachi
"""

# import pandas as pd
# import os
# from tkinter import Tk

# # 1. Crear la carpeta "SLSQP" si no existe y cambiar el directorio a ella
# # carpeta = "SLSQP"
# carpeta = "Nelder-Mead"
# # carpeta = "TNC"
# # carpeta = "Powell"
# # carpeta = "L-BGFS-B"

# if not os.path.exists(carpeta):
#     os.makedirs(carpeta)
# os.chdir(carpeta)

# # 2. Usar tkinter para abrir un cuadro de diálogo y seleccionar un archivo
# # Ocultar la ventna de tkinter
# root = Tk()
# root.withdraw()

    
# # Combinaciones
# type_act_values = [0]  # Por ejemplo: magnetorquer(0) o rueda de reacción(1)
# S_A_both_values = [0]  # Diferentes combinaciones de sensor y actuador 0:sensor, 1:actuador, 2: ambos
# type_rend_values = ['acc'] 

# with open('optimizacion.txt') as mi_archivo:
#     arch = mi_archivo.read()

# line = f'resultados_typeact{type_act_values[0]}_saboth{S_A_both_values[0]}_typerend{type_rend_values[0]}.txt:'

# # Supongamos que 'arch' ya contiene el texto completo del archivo, dividido en líneas
# lineas = arch.split('\n')  # Divide 'arch' en una lista de líneas

# # Crear una lista para almacenar los valores de 'x'
# xs = []
# f_costs = []
# # Iterar sobre las líneas con un salto de 8 líneas
# for i in range(0, len(lineas), 3):
#     linea = lineas[i]
#     xs.append(linea.split())  # Divide la línea en elementos individuales


# # Iterar sobre las líneas con un salto de 8 líneas
# for i in range(1, len(lineas), 3):
#     linea_f = lineas[i]
#     f_costs.append(linea_f.split())  # Divide la línea en elementos individuales

# name = []
# valores_f = []
# for j in range(0, len(f_costs)):
#     if line == f_costs[j][1]:
#         valores_f.append(float(f_costs[j][2]))

#%%

import os
import pandas as pd
from tkinter import Tk
import matplotlib.pyplot as plt

# Lista de carpetas
# carpetas = ["SLSQP", "Nelder-Mead", "TNC", "Powell", "L-BFGS-B"]
carpetas = ["SLSQP", "Nelder-Mead", "TNC", "Powell"]
valores_f =  []
solver = []
# 1. Recorrer todas las carpetas, crearlas si no existen y cambiar al directorio de cada una
for carpeta in carpetas:
    if not os.path.exists(carpeta):
        os.makedirs(carpeta)
    os.chdir(carpeta)

    # 2. Usar tkinter para abrir un cuadro de diálogo y seleccionar un archivo
    root = Tk()
    root.withdraw()  # Ocultar la ventana de tkinter

    # Combinaciones
    type_act_values = [1]  # Por ejemplo: magnetorquer(0) o rueda de reacción(1)
    S_A_both_values = [2]  # Diferentes combinaciones de sensor y actuador 0: sensor, 1: actuador, 2: ambos
    type_rend_values = ['acc_time']

    # Verificar si el archivo "optimizacion.txt" existe en la carpeta
    if not os.path.exists('optimizacion.txt'):
        print(f"El archivo 'optimizacion.txt' no se encontró en la carpeta {carpeta}")
        os.chdir('..')  # Volver al directorio anterior
        continue

    with open('optimizacion.txt') as mi_archivo:
        arch = mi_archivo.read()

    line = f'resultados_typeact{type_act_values[0]}_saboth{S_A_both_values[0]}_typerend{type_rend_values[0]}.txt:'

    # Dividir el contenido del archivo en líneas
    lineas = arch.split('\n')

    # Crear listas para almacenar los valores de 'x' y 'f_cost'
    xs = []
    f_costs = []

    # Iterar sobre las líneas con saltos para extraer los valores
    for i in range(0, len(lineas), 3):
        linea = lineas[i]
        xs.append(linea.split())  # Divide la línea en elementos individuales

    for i in range(1, len(lineas), 3):
        linea_f = lineas[i]
        f_costs.append(linea_f.split())  # Divide la línea en elementos individuales

    # Filtrar los valores relevantes
    for j in range(len(f_costs)):
        if line == f_costs[j][1]:
            valores_f.append(float(f_costs[j][2]))
            solver.append(carpeta)
    # Aquí podrías hacer algo con los valores obtenidos (por ejemplo, guardarlos en un archivo, graficar, etc.)
    print(f"Procesado completado para la carpeta {carpeta}")

    # Volver al directorio principal antes de pasar a la siguiente carpeta
    os.chdir('..')

#%%

iteraciones = list(range(0,len(f_costs)))

fig0, ax = plt.subplots(figsize=(15,5))  # Crea un solo set de ejes

# Graficar los tres conjuntos de datos en la misma gráfica
ax.scatter(solver, valores_f, label= line)

# Configurar etiquetas, leyenda y grid
ax.set_xlabel('solver minimize', fontsize=18)
ax.set_ylabel('funcion objetivo', fontsize=18)
ax.legend(fontsize=18)
ax.grid()

# # Ajustar límites del eje X
# ax.set_ylim(0.35e7, 0.4e7)

# Ajustar el tamaño de las etiquetas de los ticks
ax.tick_params(axis='both', which='major', labelsize=18)

plt.tight_layout()

# Guardar la gráfica como archivo SVG
# plt.savefig('norm2.svg', format='svg')

# Mostrar la gráfica
plt.show()