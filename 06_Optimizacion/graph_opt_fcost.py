# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 02:17:44 2024

@author: nachi
"""

from Suite_LQR import suite_sim
import matplotlib.pyplot as plt
import pandas as pd

       
# Definir la función objetivo con dos argumentos
def objective(x, *args):
    # Desempaquetar los argumentos adicionales
    type_act, S_A_both, type_rend,P_i = args  
    
    # Determinar los valores de std_sensor_sol, std_magnetometros y lim según S_A_both
    if S_A_both == 0:
        std_sensor_sol, std_magnetometros = x
        P_i[9] = 0
        P_i[10] = 0 
        P_i[11] = 0
        if type_act == 0:
            lim = 5
        elif type_act == 1:
            lim = 0.1255
    elif S_A_both == 1:
        lim = x
        std_sensor_sol, std_magnetometros = 0.68, 1e-9  # Valores fijos o iniciales
        for i in range(3, 9):
            P_i[i] = 0
        
    elif S_A_both == 2:
        std_sensor_sol, std_magnetometros, lim = x

    # Invocar la simulación según type_act
    acc, psd, time,pot_b, masa_b, vol_b,pot_ss, masa_ss, vol_ss, pot_act, masa_act, vol_act = suite_sim(std_sensor_sol, std_magnetometros, lim, type_act, S_A_both)
    
    # Seleccionar el valor a retornar según type_rend
    if type_rend == 'acc':
        return P_i[0]*acc**2 + P_i[3]*pot_b**2 + P_i[4]*masa_b**2 + P_i[5]*vol_b**2 +  P_i[6]*pot_ss**2 + P_i[7]*masa_ss**2 + P_i[8]*vol_ss**2 + P_i[9]*pot_act**2 + P_i[10]*masa_act**2 + P_i[11]*vol_act**2 
    elif type_rend == 'psd':
        return P_i[1]*psd**2 + P_i[3]*pot_b**2 + P_i[4]*masa_b**2 + P_i[5]*vol_b**2 +  P_i[6]*pot_ss**2 + P_i[7]*masa_ss**2 + P_i[8]*vol_ss**2 + P_i[9]*pot_act**2 + P_i[10]*masa_act**2 + P_i[11]*vol_act**2
    elif type_rend == 'time':
        return P_i[2]*time**2 + P_i[3]*pot_b**2 + P_i[4]*masa_b**2 + P_i[5]*vol_b**2 +  P_i[6]*pot_ss**2 + P_i[7]*masa_ss**2 + P_i[8]*vol_ss**2 + P_i[9]*pot_act**2 + P_i[10]*masa_act**2 + P_i[11]*vol_act**2
    elif type_rend =='acc_time':
        return P_i[0]*acc**2 + P_i[2]*time**2 + P_i[3]*pot_b**2 + P_i[4]*masa_b**2 + P_i[5]*vol_b**2 +  P_i[6]*pot_ss**2 + P_i[7]*masa_ss**2 + P_i[8]*vol_ss**2 + P_i[9]*pot_act**2 + P_i[10]*masa_act**2 + P_i[11]*vol_act**2
    elif type_rend =='acc_psd':
        return P_i[0]*acc**2 + P_i[1]*psd**2 + P_i[3]*pot_b**2 + P_i[4]*masa_b**2 + P_i[5]*vol_b**2 +  P_i[6]*pot_ss**2 + P_i[7]*masa_ss**2 + P_i[8]*vol_ss**2 + P_i[9]*pot_act**2 + P_i[10]*masa_act**2 + P_i[11]*vol_act**2
    elif type_rend =='psd_time':
        return P_i[1]*psd**2 + P_i[2]*time**2 + P_i[3]*pot_b**2 + P_i[4]*masa_b**2 + P_i[5]*vol_b**2 +  P_i[6]*pot_ss**2 + P_i[7]*masa_ss**2 + P_i[8]*vol_ss**2 + P_i[9]*pot_act**2 + P_i[10]*masa_act**2 + P_i[11]*vol_act**2
    elif type_rend =='all':
        return P_i[0]*acc**2 + P_i[1]*psd**2 + P_i[2]*time**2 + P_i[3]*pot_b**2 + P_i[4]*masa_b**2 + P_i[5]*vol_b**2 +  P_i[6]*pot_ss**2 + P_i[7]*masa_ss**2 + P_i[8]*vol_ss**2 + P_i[9]*pot_act**2 + P_i[10]*masa_act**2 + P_i[11]*vol_act**2

    
# Iterar sobre diferentes combinaciones de parámetros
type_act_values = [0, 1]  # Por ejemplo: magnetorquer(0) o rueda de reacción(1)
# type_act_values = [0]  
S_A_both_values = [0, 1, 2]  # Diferentes combinaciones de sensor y actuador 0:sensor, 1:actuador, 2: ambos
# S_A_both_values = [2]  
# type_rend_values = ['acc', 'time', 'acc_time', 'acc_psd', 'psd_time','all']  # Diferentes tipos de rendimiento
# type_rend_values = ['acc_psd','psd_time','all'] 
type_rend_values = ['acc', 'time', 'acc_time','all'] 

# hacer 'psd' por separado

# Pesos para la optimización
P_i = [1,1,1,   #Pesos: 0,1,2 -> acc,psd,time
       1,1,0,   #Pesos: 3,4,5 -> pot, masa y vol magnetometro
       1,1,0,   #Pesos: 6,7,8 -> pot, masa y vol sensor de sol
       1,1,0]   #Pesos: 9,10,11 -> pot, masa y vol actuador

caso = []
f_costs = []

# Bucle para probar diferentes configuraciones
for type_act in type_act_values:
    for S_A_both in S_A_both_values:
        for type_rend in type_rend_values:

            argss=(type_act, S_A_both, type_rend,P_i)
            
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
            
            # Iterar sobre las líneas con un salto de 7 líneas
            for i in range(0, len(lineas), 7):
                linea = lineas[i]
                # Suponiendo que los valores de 'x' están separados por espacios, los extraemos
                valores_x.append(linea.split())  # Divide la línea en elementos individuales
            
            # 'valores_x' ahora tiene todas las líneas con los valores de 'x'
            # print(valores_x)
            
            
            valores_numericos_por_sublista = []  # Lista que contendrá sublistas de números
            
            # Recorremos cada sublista en 'valores_x'
            for sublista in valores_x:
                sublista_numeros = []  # Sublista para almacenar los valores numéricos de esta iteración
                # print(sublista)
                
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
            caso.append(valores_numericos_por_sublista[-2])
            if S_A_both ==1:
                f_cost = objective(float(valores_numericos_por_sublista[-2][0]), *argss)
            else:
                f_cost = objective(valores_numericos_por_sublista[-2], *argss)
            f_costs.append(f_cost)
            print(f"simulacion para type_act={type_act}, S_A_both={S_A_both}, type_rend={type_rend}")

    # # Mostramos la lista general con las sublistas de números
    # print(valores_numericos_por_sublista)

#%%
# Nombre del archivo basado en las opciones seleccionadas
nombre_archivo = "best_rend.csv"

# Crear un diccionario con los datos
datos = {
    'f_costo': f_costs
}

# Crear un DataFrame de pandas a partir del diccionario
df_resultados = pd.DataFrame(datos)

# Guardar el DataFrame en un archivo CSV
df_resultados.to_csv(nombre_archivo, index=False)

#%%

archivo_csv = "best_rend.csv"

# Leer el archivo CSV en un DataFrame de pandas
df = pd.read_csv(archivo_csv)

# Convertir el DataFrame a un array de NumPy
array_datos = df.to_numpy()

f_costs = array_datos[:, 0]
iteraciones = list(range(0,len(f_costs)))

fig0, ax = plt.subplots(figsize=(15,5))  # Crea un solo set de ejes

# Graficar los tres conjuntos de datos en la misma gráfica
ax.scatter(iteraciones, f_costs, label='its_minimize')

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