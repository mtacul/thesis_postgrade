# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 19:44:40 2024

@author: nachi
"""
import matplotlib.pyplot as plt
import os
from tkinter import Tk

# 1. Crear la carpeta "SLSQP" si no existe y cambiar el directorio a ella
carpeta = "SLSQP"
# carpeta = "Nelder-Mead"
# carpeta = "TNC"
# carpeta = "Powell"
# carpeta = "L-BGFS-B"

if not os.path.exists(carpeta):
    os.makedirs(carpeta)
os.chdir(carpeta)

# 2. Usar tkinter para abrir un cuadro de diálogo y seleccionar un archivo
# Ocultar la ventna de tkinter
root = Tk()
root.withdraw()


# Definir valores de type_act, S_A_both, type_rend y Pesos P_i
type_act = 1  # 0: magnetorquer, 1: rueda de reacción
S_A_both = 0  # 0: solo sensor, 1: solo actuador, 2: ambos
type_rend = 'acc'  # Puede ser 'acc', 'psd', 'time', 'acc_time', 'acc_psd', 'psd_time' y 'all'


# Valores iniciales para las desviaciones estándar o lim según el caso
if S_A_both == 0:
    if type_act == 0:
        if type_rend == 'acc':
            with open('resultados_typeact0_saboth0_typerendacc.txt') as mi_archivo:
                arch = mi_archivo.read() # Acá leemos todo el contenido del archivo
        elif type_rend == 'time':
            with open('resultados_typeact0_saboth0_typerendtime.txt') as mi_archivo:
                arch = mi_archivo.read() # Acá leemos todo el contenido del archivo
        elif type_rend == 'acc_time':
            with open('resultados_typeact0_saboth0_typerendacc_time.txt') as mi_archivo:
                arch = mi_archivo.read() # Acá leemos todo el contenido del archivo
        elif type_rend == 'all':
            with open('resultados_typeact0_saboth0_typerendall.txt') as mi_archivo:
                arch = mi_archivo.read() # Acá leemos todo el contenido del archivo            
    elif type_act == 1:
        if type_rend == 'acc':
            with open('resultados_typeact1_saboth0_typerendacc.txt') as mi_archivo:
                arch = mi_archivo.read() # Acá leemos todo el contenido del archivo
        elif type_rend == 'time':
            with open('resultados_typeact1_saboth0_typerendtime.txt') as mi_archivo:
                arch = mi_archivo.read() # Acá leemos todo el contenido del archivo
        elif type_rend == 'acc_time':
            with open('resultados_typeact1_saboth0_typerendacc_time.txt') as mi_archivo:
                arch = mi_archivo.read() # Acá leemos todo el contenido del archivo
        elif type_rend == 'all':
            with open('resultados_typeact1_saboth0_typerendall.txt') as mi_archivo:
                arch = mi_archivo.read() # Acá leemos todo el contenido del archivo             
            
elif S_A_both == 1:
    if type_act == 0:
        if type_rend == 'acc':
            with open('resultados_typeact0_saboth1_typerendacc.txt') as mi_archivo:
                arch = mi_archivo.read() # Acá leemos todo el contenido del archivo
        elif type_rend == 'time':
            with open('resultados_typeact0_saboth1_typerendtime.txt') as mi_archivo:
                arch = mi_archivo.read() # Acá leemos todo el contenido del archivo
        elif type_rend == 'acc_time':
            with open('resultados_typeact0_saboth1_typerendacc_time.txt') as mi_archivo:
                arch = mi_archivo.read() # Acá leemos todo el contenido del archivo
        elif type_rend == 'all':
            with open('resultados_typeact0_saboth1_typerendall.txt') as mi_archivo:
                arch = mi_archivo.read() # Acá leemos todo el contenido del archivo        
    elif type_act == 1:
        if type_rend == 'acc':
            with open('resultados_typeact1_saboth1_typerendacc.txt') as mi_archivo:
                arch = mi_archivo.read() # Acá leemos todo el contenido del archivo
        elif type_rend == 'time':
            with open('resultados_typeact1_saboth1_typerendtime.txt') as mi_archivo:
                arch = mi_archivo.read() # Acá leemos todo el contenido del archivo
        elif type_rend == 'acc_time':
            with open('resultados_typeact1_saboth1_typerendacc_time.txt') as mi_archivo:
                arch = mi_archivo.read() # Acá leemos todo el contenido del archivo
        elif type_rend == 'all':
            with open('resultados_typeact1_saboth1_typerendall.txt') as mi_archivo:
                arch = mi_archivo.read() # Acá leemos todo el contenido del archivo
            
            
elif S_A_both == 2:
    if type_act == 0:
        if type_rend == 'acc':
            with open('resultados_typeact0_saboth2_typerendacc.txt') as mi_archivo:
                arch = mi_archivo.read() # Acá leemos todo el contenido del archivo
        elif type_rend == 'time':
            with open('resultados_typeact0_saboth2_typerendtime.txt') as mi_archivo:
                arch = mi_archivo.read() # Acá leemos todo el contenido del archivo
        elif type_rend == 'acc_time':
            with open('resultados_typeact0_saboth2_typerendacc_time.txt') as mi_archivo:
                arch = mi_archivo.read() # Acá leemos todo el contenido del archivo
        elif type_rend == 'all':
            with open('resultados_typeact0_saboth2_typerendall.txt') as mi_archivo:
                arch = mi_archivo.read() # Acá leemos todo el contenido del archivo         
    elif type_act == 1:
        if type_rend == 'acc':
            with open('resultados_typeact1_saboth2_typerendacc.txt') as mi_archivo:
                arch = mi_archivo.read() # Acá leemos todo el contenido del archivo
        elif type_rend == 'time':
            with open('resultados_typeact1_saboth2_typerendtime.txt') as mi_archivo:
                arch = mi_archivo.read() # Acá leemos todo el contenido del archivo
        elif type_rend == 'acc_time':
            with open('resultados_typeact1_saboth2_typerendacc_time.txt') as mi_archivo:
                arch = mi_archivo.read() # Acá leemos todo el contenido del archivo
        elif type_rend == 'all':
            with open('resultados_typeact1_saboth2_typerendall.txt') as mi_archivo:
                arch = mi_archivo.read() # Acá leemos todo el contenido del archivo


# Supongamos que 'arch' ya contiene el texto completo del archivo, dividido en líneas
lineas = arch.split('\n')  # Divide 'arch' en una lista de líneas

# Crear una lista para almacenar los valores de 'x'
valores_x = []
f_costs = []
# Iterar sobre las líneas con un salto de 8 líneas
for i in range(0, len(lineas), 8):
    linea = lineas[i]
    valores_x.append(linea.split())  # Divide la línea en elementos individuales


# Iterar sobre las líneas con un salto de 8 líneas
for i in range(6, len(lineas), 8):
    linea_f = lineas[i]
    f_costs.append(linea_f.split())  # Divide la línea en elementos individuales

valores_numericos_por_sublista = []  # Lista que contendrá sublistas de números

for sublista in valores_x:
    sublista_numeros = []  # Sublista para almacenar los valores numéricos de esta iteración
    for valor in sublista:
        # Limpiamos los corchetes u otros caracteres que no queremos
        valor = valor.replace('[', '').replace(']', '')  # Eliminamos corchetes si es necesario
        
        # Intentamos convertir cada valor a float
        try:
            num = float(valor)  # Si se puede convertir, lo guardamos en la sublista de números
            sublista_numeros.append(num)
        except ValueError:
            # Si hay un error (es decir, no se puede convertir a float), lo ignoramos
            pass
    
    # Añadimos la sublista con los números a la lista general
    valores_numericos_por_sublista.append(sublista_numeros)


valores_f = []

for sublista in f_costs:
    sublista_numeros = []  # Sublista para almacenar los valores numéricos de esta iteración
    for valor in sublista:
        # Limpiamos los corchetes u otros caracteres que no queremos
        valor = valor.replace('[', '').replace(']', '')  # Eliminamos corchetes si es necesario
        
        # Intentamos convertir cada valor a float
        try:
            num = float(valor)  # Si se puede convertir, lo guardamos en la sublista de números
            sublista_numeros.append(num)
        except ValueError:
            # Si hay un error (es decir, no se puede convertir a float), lo ignoramos
            pass
    
    # Añadimos la sublista con los números a la lista general
    valores_f.append(sublista_numeros)

iteraciones = list(range(0,len(valores_numericos_por_sublista)-1))

#%%

fig0, ax = plt.subplots(figsize=(15,5))  # Crea un solo set de ejes

# Graficar los tres conjuntos de datos en la misma gráfica
ax.scatter(iteraciones, valores_f, label='its_minimize')

# Configurar etiquetas, leyenda y grid
ax.set_xlabel('iteraciones [-]', fontsize=18)
ax.set_ylabel('funcion objetivo', fontsize=18)
ax.legend(fontsize=18)
ax.grid()

# # Ajustar límites del eje X
# ax.set_ylim(60000, 80000)

# Ajustar el tamaño de las etiquetas de los ticks
ax.tick_params(axis='both', which='major', labelsize=18)

plt.tight_layout()

# Guardar la gráfica como archivo SVG
# plt.savefig('norm2.svg', format='svg')

# Mostrar la gráfica
plt.show()