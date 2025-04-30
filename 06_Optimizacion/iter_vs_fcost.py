# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 19:44:40 2024

@author: nachi
"""
import matplotlib.pyplot as plt
import os
from tkinter import Tk

# 1. Crear la carpeta "SLSQP" si no existe y cambiar el directorio a ella
# carpeta = "SLSQP"
# carpeta = "Nelder-Mead"
# carpeta = "TNC"
# carpeta = "Powell"
# carpeta = "L-BFGS-B"
# carpeta = "articulo"
carpeta = "articulo_2"

if not os.path.exists(carpeta):
    os.makedirs(carpeta)
os.chdir(carpeta)

# 2. Usar tkinter para abrir un cuadro de diálogo y seleccionar un archivo
# Ocultar la ventna de tkinter
root = Tk()
root.withdraw()


# Definir valores de type_act, S_A_both, type_rend y Pesos P_i
type_act = 0  # 0: magnetorquer, 1: rueda de reacción
S_A_both = 2  # 0: solo sensor, 1: solo actuador, 2: ambos
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
MoPs = []
# Iterar sobre las líneas con un salto de 8 líneas
for i in range(0, len(lineas), 8):
    linea = lineas[i]
    valores_x.append(linea.split())  # Divide la línea en elementos individuales
    
for i in range(1, len(lineas), 8):
    linea_perf = lineas[i]
    MoPs.append(linea_perf.split())  # Divide la línea en elementos individuales

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

valores_MoPs = []  # Lista que contendrá sublistas de números

for linea in lineas[1::8]:  # Itera cada 8 líneas desde la segunda línea
    try:
        # Extraer los valores numéricos
        partes = linea.split(", ")  # Divide la línea por ", "
        acc = float(partes[0].split(": ")[1])   # Extrae acc
        psd = float(partes[1].split(": ")[1])   # Extrae psd
        time = float(partes[2].split(": ")[1])  # Extrae time

        # Guardar como una tupla (acc, psd, time)
        valores_MoPs.append([acc, psd, time])
    except (IndexError, ValueError) as e:
        print(f"Error al procesar la línea: {linea}, {e}")  # Para depuración

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

# Asegúrate de que los datos están en la forma correcta
x = iteraciones  # Eje X (índices)
y = [sublista[0] for sublista in valores_f]  # Eje Y (tomando el primer valor de cada sublista)
acc = [sublistaa[0] for sublistaa in valores_MoPs]
time = [sublistaaa[2] for sublistaaa in valores_MoPs]

# Encontrar valores mínimos
min_acc = min(acc)
min_time = min(time)
min_f = min(y)

max_acc = max(acc)
max_time = max(time)
max_f = max(y)

# Filtrar valores que no excedan 10 veces el mínimo y mantener los índices válidos
filtered_indices = [i for i in range(len(acc)) if acc[i] <= 10 * min_acc and time[i] <= 10 * min_time and y[i] <= 10 * min_f]

# Filtrar listas según los índices válidos
filtered_x = [x[i] for i in filtered_indices]
filtered_acc = [acc[i] for i in filtered_indices]
filtered_time = [time[i] for i in filtered_indices]
filtered_y = [y[i] for i in filtered_indices]

acc_norm = [a / max_acc for a in filtered_acc]
time_norm = [b / max_time for b in filtered_time]
y_norm = [c/max_f for c in filtered_y]

# Convertir los pares ordenados en cadenas de texto
pares_x = [f"({format(sublista[0], '.3g')} [°], {format(sublista[1], '.3g')} [T])" 
            if len(sublista) >= 2 else f"({format(sublista[0], '.3g')}, ?)" 
            for sublista in valores_numericos_por_sublista if len(sublista) > 0]

# pares_x = [f"({format(sublista[0], '.3g')} [Am2])" 
#             if len(sublista) >= 1 else f"({format(sublista[0], '.3g')})" 
#             for sublista in valores_numericos_por_sublista if len(sublista) > 0]

#%%
# Crear figura
fig, ax = plt.subplots(figsize=(10, 6))

# Graficar las tres funciones en la misma gráfica
ax.plot(filtered_x, filtered_y, marker='o', linestyle='-', label='F Cost', color='blue')
ax.plot(filtered_x, filtered_acc, marker='o', linestyle='-', label='Acc [°]', color='red')
ax.plot(filtered_x, filtered_time, marker='o', linestyle='-', label='Settling Time/200 [-]', color='green')

# Etiquetas de los ejes
# target.set_xlabel("X Values")
ax.set_ylabel("Values")
ax.set_title("Iterations vs F Cost, Acc, and Settling Time")
# ax.set_ylim(0,4)
ax.grid()

# Ajustar etiquetas del eje X
ax.set_xticks(range(0, len(x), 2))
# ax.set_xticks(range(0, len(x), 3))
ax.set_xticklabels([pares_x[i] for i in range(0, len(pares_x), 2)], rotation=45)
            
# ax.set_xticklabels([pares_x[i] for i in range(0, len(pares_x), 3)], rotation=45)

# Agregar leyenda
ax.legend()

# Ajustar diseño para evitar sobreposiciones
plt.tight_layout()

# Guardar el gráfico
# plt.savefig('case_1.pdf', format='pdf', bbox_inches='tight')
# plt.savefig('case_2.pdf', format='pdf', bbox_inches='tight')

# Mostrar el gráfico
plt.show()

# Crear figura
fig, ax = plt.subplots(figsize=(10, 6))



# Graficar las tres funciones en la misma gráfica
ax.plot(filtered_x, y_norm, marker='o', linestyle='-', label='F Cost', color='blue')
ax.plot(filtered_x, acc_norm, marker='o', linestyle='-', label='Acc [°]', color='red')
ax.plot(filtered_x, time_norm, marker='o', linestyle='-', label='Settling Time/200', color='green')

# Etiquetas de los ejes
# target.set_xlabel("X Values")
ax.set_ylabel("Values [-]")
ax.set_title("Iterations vs F Cost, Acc, and Settling Time normalized")
# ax.set_ylim(0,4)
ax.grid()

# Ajustar etiquetas del eje X
ax.set_xticks(range(0, len(x), 2))
# ax.set_xticks(range(0, len(x), 3))
ax.set_xticklabels([pares_x[i] for i in range(0, len(pares_x), 2)], rotation=45)
            
# ax.set_xticklabels([pares_x[i] for i in range(0, len(pares_x), 3)], rotation=45)

# Agregar leyenda
ax.legend()

# Ajustar diseño para evitar sobreposiciones
plt.tight_layout()

# Guardar el gráfico
# plt.savefig('case_1_norm.pdf', format='pdf', bbox_inches='tight')
# plt.savefig('case_2_norm.pdf', format='pdf', bbox_inches='tight')

# Mostrar el gráfico
plt.show()


#%%

# # Crear figura con 2 subgráficos (1 fila, 2 columnas)
# fig, axes = plt.subplots(3, 1, figsize=(10, 10))

# # Primer gráfico: x vs. y
# axes[0].plot(x, y, marker='o', linestyle='-')
# axes[0].set_xlabel("X Values")
# axes[0].set_ylabel("f cost values")
# # axes[0].set_ylim(2, 4)
# axes[0].set_title("Iterations vs F Cost")
# axes[0].grid()

# # Segundo gráfico: x vs. acc
# axes[1].plot(x, acc, marker='o', linestyle='-', color='red')
# axes[1].set_xlabel("X Values")
# axes[1].set_ylabel("acc values [°]")
# axes[1].set_title("Iterations vs Acc")
# axes[1].grid()

# # tercer gráfico: x vs. time
# axes[2].plot(x, time, marker='o', linestyle='-', color='green')
# axes[2].set_xlabel("X Values")
# axes[2].set_ylabel("Scaled time values [-]")
# # axes[2].set_ylim(-0.5, 0.5)
# axes[2].set_title("Iterations vs Settling time/200")
# axes[2].grid()

# # Ajustar etiquetas del eje X en ambos gráficos
# for ax in axes:
#     # ax.set_xticks(range(0, len(x), 2))
#     ax.set_xticks(range(0, len(x), 3))
#     # ax.set_xticklabels([pares_x[i] for i in range(0, len(pares_x), 2)], rotation=45)
#     ax.set_xticklabels([pares_x[i] for i in range(0, len(pares_x), 3)], rotation=45)


# # Ajustar diseño para evitar sobreposiciones
# plt.tight_layout()

# # Guardar el gráfico
# # plt.savefig('case_1.pdf', format='pdf', bbox_inches='tight')
# plt.savefig('case_2.pdf', format='pdf', bbox_inches='tight')

# # Mostrar el gráfico
# plt.show()

#%%

# fig0, ax = plt.subplots(figsize=(15,5))  # Crea un solo set de ejes

# # Graficar los tres conjuntos de datos en la misma gráfica
# ax.scatter(iteraciones, valores_f, label='its_minimize')

# # Configurar etiquetas, leyenda y grid
# ax.set_xlabel('iteraciones [-]', fontsize=18)
# ax.set_ylabel('funcion objetivo', fontsize=18)
# ax.legend(fontsize=18)
# ax.grid()

# # # Ajustar límites del eje X
# ax.set_ylim(0, 10)

# # Ajustar el tamaño de las etiquetas de los ticks
# ax.tick_params(axis='both', which='major', labelsize=18)

# plt.tight_layout()

# # Guardar la gráfica como archivo SVG
# # plt.savefig('norm2.svg', format='svg')

# # Mostrar la gráfica
# plt.show()

#%%
# # Crear el gráfico
# plt.figure(figsize=(12, 6))
# plt.plot(x, y, marker='o', linestyle='-')
# # plt.plot(x, acc, marker='o', linestyle='-',color='red')

# # Configurar etiquetas en el eje X cada 5 valores
# plt.xticks(range(0, len(x), 2), [pares_x[i] for i in range(0, len(pares_x), 2)], rotation=45)

# # plt.xticks(range(0, len(x), 3), [pares_x[i] for i in range(0, len(pares_x), 3)], rotation=45)

# # Etiquetas y título
# plt.xlabel("x values")
# plt.ylabel("f cost values")
# plt.title("Iteration results of each optimization step: Case 01")

# # Mostrar la cuadrícula
# plt.grid()
# plt.savefig('case_1.pdf', format='pdf', bbox_inches='tight')

# # Mostrar el gráfico
# plt.show()