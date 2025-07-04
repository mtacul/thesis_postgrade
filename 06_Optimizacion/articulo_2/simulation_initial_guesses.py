# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 13:46:53 2025

@author: nachi
"""
import numpy as np
from itertools import product

def generate_initial_guesses(bounds, num_points_per_dim=4):
    """
    Genera una lista de initial guesses equiespaciados dentro de los bounds.
    num_points_per_dim: cuántos puntos quieres por dimensión (3**3 = 27 si hay 3 variables).
    """
    grids = [np.linspace(low, high, num_points_per_dim) for (low, high) in bounds]
    return list(product(*grids))  # devuelve todas las combinaciones

bnds = ((0.01, 1.67), (0.012e-9, 3e-9), (0.29, 15))  # Para std_sensor_sol, std_magnetometros y lim
initial_guesses = generate_initial_guesses(bnds)

# # Leer el archivo y asociar cada initial_guess con su función de costo
# file_path = "resultados_typeact0_saboth2_typerendacc.txt"
# file_path_time = "resultados_typeact0_saboth2_typerendtime.txt"
# file_path_all = "resultados_typeact0_saboth2_typerendall.txt"

# # Guardaremos aquí los resultados encontrados
# guess_to_cost = {}
# guess_to_cost_time = {}
# guess_to_cost_all = {}

# with open(file_path, 'r') as file:
#     lines = file.readlines()
# with open(file_path_time, 'r') as file:
#     lines_time = file.readlines()
# with open(file_path_all, 'r') as file:
#     lines_all = file.readlines()    

# # Recorremos el archivo buscando 'x:' y 'funcion de costo:'
# for i in range(len(lines)):
#     if lines[i].startswith('x:'):
#         # Extraemos el guess
#         x_str = lines[i].split(':')[1].strip()
#         x_vals = [float(val.replace('e', 'E')) for val in x_str.strip('[]').split()]
        
#         # Buscamos la línea de función de costo que viene después
#         for j in range(i, len(lines)):
#             if "funcion de costo:" in lines[j]:
#                 f_val = float(lines[j].split(':')[1].strip())
#                 break
        
#         # Ahora, matcheamos el guess encontrado contra tus initial_guesses
#         for guess in initial_guesses:
#             # Si las coordenadas son muy parecidas (por errores numéricos pequeños)
#             if np.allclose(x_vals, guess, rtol=1e-3, atol=1e-6):
#                 guess_to_cost[tuple(guess)] = f_val
#                 break  # Ya lo encontramos, no seguir buscando
                
# for i in range(len(lines_time)):
#     if lines_time[i].startswith('x:'):
#         # Extraemos el guess
#         x_str_time = lines_time[i].split(':')[1].strip()
#         x_vals_time = [float(val.replace('e', 'E')) for val in x_str_time.strip('[]').split()]
        
#         # Buscamos la línea de función de costo que viene después
#         for j in range(i, len(lines_time)):
#             if "funcion de costo:" in lines_time[j]:
#                 f_val_time = float(lines_time[j].split(':')[1].strip())
#                 break
        
#         # Ahora, matcheamos el guess encontrado contra tus initial_guesses
#         for guess in initial_guesses:
#             # Si las coordenadas son muy parecidas (por errores numéricos pequeños)
#             if np.allclose(x_vals_time, guess, rtol=1e-3, atol=1e-6):
#                 guess_to_cost_time[tuple(guess)] = f_val_time
#                 break  # Ya lo encontramos, no seguir buscando

# for i in range(len(lines_all)):
#     if lines_all[i].startswith('x:'):
#         # Extraemos el guess
#         x_str_all = lines_all[i].split(':')[1].strip()
#         x_vals_all = [float(val.replace('e', 'E')) for val in x_str_all.strip('[]').split()]
        
#         # Buscamos la línea de función de costo que viene después
#         for j in range(i, len(lines_all)):
#             if "funcion de costo:" in lines_all[j]:
#                 f_val_all = float(lines_all[j].split(':')[1].strip())
#                 break
        
#         # Ahora, matcheamos el guess encontrado contra tus initial_guesses
#         for guess in initial_guesses:
#             # Si las coordenadas son muy parecidas (por errores numéricos pequeños)
#             if np.allclose(x_vals_all, guess, rtol=1e-3, atol=1e-6):
#                 guess_to_cost_all[tuple(guess)] = f_val_all
#                 break  # Ya lo encontramos, no seguir buscando

# # Imprimimos todo lo encontrado
# for guess, cost in guess_to_cost.items():
#     print(f"Guess: {guess} -> Función de costo: {cost}")
# for guess, cost in guess_to_cost_time.items():
#     print(f"Guess: {guess} -> Función de costo: {cost}")
# for guess, cost in guess_to_cost_all.items():
#     print(f"Guess: {guess} -> Función de costo: {cost}")
    
import ast
import numpy as np


# Ahora leemos los resultados
file_path_2 = "optimizacion.txt"
results = []

with open(file_path_2, 'r') as file:
    lines_2 = file.readlines()

for i in range(0, len(lines_2), 3):
    line_x_2 = lines_2[i].strip()
    line_f_2 = lines_2[i+1].strip()
    
    # Extraemos
    x_str_2 = line_x_2.split(':', 1)[1].strip()

    # Agregar comas entre números:
    x_str_fixed_2 = x_str_2.replace(' ', ', ')
    
    x_vals_2 = ast.literal_eval(x_str_fixed_2)
    
    f_val_2 = float(line_f_2.split(':', 1)[1].strip())
    
    results.append((x_vals_2, f_val_2))

results_acc = results[0:64]
results_time = results[64:64*2]
results_psd = results[64*2:64*3]

# Ahora imprimimos todo relacionando guess inicial y resultado
for idx, ((x, f), guess) in enumerate(zip(results_acc, initial_guesses)):
    print(f"Optimización {idx+1}:")
    print(f"  Initial guess: {guess}")
    print(f"  x óptimo: {x}")
    print(f"  Valor de función objetivo: {f}\n")
for idx, ((x, f), guess) in enumerate(zip(results_time, initial_guesses)):
    print(f"Optimización {idx+1}:")
    print(f"  Initial guess: {guess}")
    print(f"  x óptimo: {x}")
    print(f"  Valor de función objetivo: {f}\n")
for idx, ((x, f), guess) in enumerate(zip(results_psd, initial_guesses)):
    print(f"Optimización {idx+1}:")
    print(f"  Initial guess: {guess}")
    print(f"  x óptimo: {x}")
    print(f"  Valor de función objetivo: {f}\n")
#%%

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FuncFormatter

# Función para formatear los valores del eje Y
def y_formatter(y, _):
    return f'{y*1e9:.1f}'  # puedes ajustar los decimales



# Separamos los valores para graficar
initial_x = [g[0] for g in initial_guesses]
initial_y = [g[1] for g in initial_guesses]
initial_z = [g[2] for g in initial_guesses]

# optimal_x = [x[0] for x, _ in results_acc]
# optimal_y = [x[1] for x, _ in results_acc]
# optimal_z = [x[2] for x, _ in results_acc]

# optimal_x = [x[0] for x, _ in results_time]
# optimal_y = [x[1] for x, _ in results_time]
# optimal_z = [x[2] for x, _ in results_time]

optimal_x = [x[0] for x, _ in results_psd]
optimal_y = [x[1] for x, _ in results_psd]
optimal_z = [x[2] for x, _ in results_psd]


# Crear el gráfico
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Graficar
ax.scatter(initial_x, initial_y, initial_z, c='blue', label='Initial guesses')
ax.scatter(optimal_x, optimal_y, optimal_z, c='red', label='Optimal')

# Aplicar el formateador
ax.yaxis.set_major_formatter(FuncFormatter(y_formatter))

# Etiquetas
ax.set_xlabel('$\sigma_s$ [°]')
ax.set_ylabel('$\sigma_b$ [nT]')
ax.set_zlabel('$\tau$ [Am^2]')

ax.legend()
ax.grid(True)
# plt.title('Initial guesses and optimal values optimizing accuracy')
# plt.title('Initial guesses and optimal values optimizing agility')
plt.title('Initial guesses and optimal values optimizing jitter')
# plt.savefig('opt_acc.pdf', format='pdf')
# plt.savefig('opt_agility.pdf', format='pdf')
plt.savefig('opt_jitter.pdf', format='pdf')
plt.show()



#%%


# === GRÁFICO EN EL PLANO XZ ===
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111)
ax.scatter(initial_x, initial_z, c='blue', label='Initial guesses')
ax.scatter(optimal_x, optimal_z, c='red', label='Optimal')
ax.set_xlabel('$\sigma_s$ [°]', fontsize=19)
ax.set_ylabel(r'$\tau$ [Am$^2$]', fontsize=19)
ax.tick_params(labelsize=17)  # Tamaño de números en los ejes
ax.grid(True)
ax.legend(fontsize=16)        # Tamaño de la leyenda
ax.set_title('XZ plane', fontsize=18)
plt.tight_layout(rect=[0, 0, 1, 0.95])
# plt.savefig('opt_acc_2D_XZ.pdf', format='pdf')
# plt.savefig('opt_agility_2D_XZ.pdf', format='pdf')
plt.savefig('opt_jitter_2D_XZ.pdf', format='pdf')
plt.show()


# === GRÁFICO EN EL PLANO XY ===
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111)  # ← Esto también te faltaba
ax.scatter(initial_x, initial_y, c='blue', label='Initial guesses')
ax.scatter(optimal_x, optimal_y, c='red', label='Optimal')
ax.set_xlabel('$\sigma_s$ [°]', fontsize=19)
ax.set_ylabel('$\sigma_b$ [nT]',fontsize=19)
ax.tick_params(labelsize=17)  # Tamaño de números en los ejes
ax.grid(True)
ax.legend(fontsize=16)        # Tamaño de la leyenda
ax.set_title('XY plane', fontsize=18)
plt.tight_layout(rect=[0, 0, 1, 0.95])
# plt.savefig('opt_acc_2D_XY.pdf', format='pdf')
# plt.savefig('opt_agility_2D_XY.pdf', format='pdf')
plt.savefig('opt_jitter_2D_XY.pdf', format='pdf')
plt.show()


#%%
# fig, axs = plt.subplots(1, 2, figsize=(15, 6))  # Dos subplots horizontales

# # Plot en el plano XY
# axs[0].scatter(initial_x, initial_y, c='blue', label='Initial guesses')
# axs[0].scatter(optimal_x, optimal_y, c='red', label='Optimal')
# axs[0].set_xlabel('$\sigma_s$ [°]')
# axs[0].set_ylabel('$\sigma_b$ [nT]')
# axs[0].yaxis.set_major_formatter(FuncFormatter(y_formatter))
# axs[0].grid(True)
# axs[0].legend()
# axs[0].set_title('XY plane')

# # Plot en el plano YZ
# axs[1].scatter(initial_x, initial_z, c='blue', label='Initial guesses')
# axs[1].scatter(optimal_x, optimal_z, c='red', label='Optimal')
# axs[1].set_xlabel('$\sigma_s$ [°]')
# axs[1].set_ylabel('$\tau$ [Am$^2$]')
# # axs[1].xaxis.set_major_formatter(FuncFormatter(y_formatter))
# axs[1].grid(True)
# axs[1].legend()
# axs[1].set_title('XZ plane')

# # plt.suptitle('Initial guesses and optimal values optimizing accuracy')
# # plt.suptitle('Initial guesses and optimal values optimizing agiity')
# plt.suptitle('Initial guesses and optimal values optimizing jitter')
# plt.tight_layout(rect=[0, 0, 1, 0.95])  # Ajuste para suptítulo
# # plt.savefig('opt_acc_2D.pdf', format='pdf')
# # plt.savefig('opt_agility_2D.pdf', format='pdf')
# plt.savefig('opt_jitter_2D.pdf', format='pdf')
# plt.show()
