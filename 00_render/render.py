# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 23:00:36 2024

@author: nachi
"""
# pip install raylib-py
import raylibpy as rp
import sys
import os
from ctypes import byref
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import welch
import numpy as np


#%%
archivo_csv = "Vectores_orbit_ECI.csv"

# Leer el archivo CSV en un DataFrame de pandas
df = pd.read_csv(archivo_csv)

# Convertir el DataFrame a un array de NumPy
array_datos = df.to_numpy()


q0_e2o = array_datos[:, 13]
q1_e2o = array_datos[:, 14]
q2_e2o = array_datos[:, 15]
q3_e2o = array_datos[:, 16]

# Nombre del archivo CSV
archivo_csv = "vectores_60k.csv"

# Leer el archivo CSV en un DataFrame de pandas
df = pd.read_csv(archivo_csv)

# Convertir el DataFrame a un array de NumPy
array_datos = df.to_numpy()

position = np.transpose(np.vstack((array_datos[:,1], array_datos[:,2], array_datos[:,3])))
velocity = np.transpose(np.vstack((array_datos[:,4], array_datos[:,5], array_datos[:,6])))


# Definir la lista de archivos CSV disponibles
archivos_disponibles = [
    "_sen3_act1_MT_LQR.csv",
    "_sen3_act2_MT_LQR.csv",
    "_sen3_act3_MT_LQR.csv",
    "control_good_magnet_o.csv"
]

# Mostrar el menú al usuario
print("Seleccione un archivo CSV para abrir:")
for i, archivo in enumerate(archivos_disponibles, 1):
    print(f"{i}. {archivo}")

# Obtener la elección del usuario
opcion = int(input("Ingrese el número de archivo deseado: "))

# Validar la opción del usuario
if 1 <= opcion <= len(archivos_disponibles):
    archivo_c_csv = archivos_disponibles[opcion - 1]
    
    # Leer el archivo CSV en un DataFrame de pandas
    df_c = pd.read_csv(archivo_c_csv)

    # Convertir el DataFrame a un array de NumPy
    array_datos_c = df_c.to_numpy()
    
    # Ahora puedes usar 'array_datos_c' en tu código
    print(f"Archivo seleccionado: {archivo_c_csv}")
else:
    print("Opción inválida. Por favor, ingrese un número válido.")
    
print("\n")

t_aux = array_datos_c[:1502,0]
q0_rot = array_datos_c[:,4]
q1_rot = array_datos_c[:,5]
q2_rot = array_datos_c[:,6]
q3_rot = array_datos_c[:,7]
# q0_rot = []
# q1_rot = []
# q2_rot = []
# q3_rot = []

# for i in range(len(t_aux)):
    
#     q_o2b = [q0_orbit[i],q1_orbit[i],q2_orbit[i],q3_orbit[i]]
#     q_e2o = [q0_e2o[i],q1_e2o[i],q2_e2o[i],q3_e2o[i]]
#     q_b2e = functions.quat_mult(q_e2o , q_o2b)
#     q0_rot.append(q_b2e[0])
#     q1_rot.append(q_b2e[1])
#     q2_rot.append(q_b2e[2])
#     q3_rot.append(q_b2e[3])

# q0_rot = np.array(q0_rot)
# q1_rot = np.array(q1_rot)
# q2_rot = np.array(q2_rot)
# q3_rot = np.array(q3_rot)

#%%

GLSL_VERSION = 330

def main():
    
    screenWidth = 1200
    screenHeight = 700

    rp.init_window(screenWidth, screenHeight, "raylib [shaders] example - simple shader mask")

    camera = rp.Camera()
    #camera.position  = rp.Vector3(0.0, 1.0, 2.0)
    # camera.position  = rp.Vector3(10, 10, 10)
    # camera.target  = rp.Vector3(0.0, 0.0, 0.0)
    camera.up  = rp.Vector3(0.0, 1.0, 0.0)
    camera.fovy  = 45.0
    camera.projection  = rp.CAMERA_PERSPECTIVE

    # torus = rp.gen_mesh_torus(0.3, 1, 16, 32)
    # model1 = rp.load_model_from_mesh(torus)
    cube = rp.gen_mesh_cube(1e-2, 1e-2, 3e-2)
    model1 = rp.load_model_from_mesh(cube)
    sphere = rp.gen_mesh_sphere(6.371, 16, 16)
    model2 = rp.load_model_from_mesh(sphere)
    
    texDiffuse = rp.load_texture("resources/Solid_yellow.png")
    texDiffuse2 = rp.load_texture("resources/earth albedo.png")
    model1.materials[0].maps[rp.MATERIAL_MAP_ALBEDO].texture  = texDiffuse
    model2.materials[0].maps[rp.MATERIAL_MAP_ALBEDO].texture  = texDiffuse2

    background = rp.load_texture("resources/space3.png")

    framesCounter = 0

    rp.disable_cursor()

    rp.set_target_fps(20)
    
    i = 0
    default_font = rp.get_font_default()
    

    while not rp.window_should_close():
        rp.update_camera(camera.byref, rp.CAMERA_FIRST_PERSON)
        framesCounter += 1
        Q = rp.Quaternion(q0_rot[i], q1_rot[i], q2_rot[i], q3_rot[i])


        # rp.set_shader_value(shader, shaderFrame, byref(rp.Int(framesCounter)), rp.SHADER_UNIFORM_INT)
        model1.transform = rp.quaternion_to_matrix(Q)
        with rp.drawing():
            rp.clear_background(rp.BLUE)
            rp.draw_texture(background, 0, 0, rp.WHITE)

            with rp.mode3d(camera):
                camera.position  = rp.Vector3(position[i][0]/1000 - 0.5, position[i][1]/1000+ 0.5, position[i][2]/1000- 0.5)
                camera.target  = rp.Vector3(position[i][0]/1000, position[i][1]/1000, position[i][2]/1000)

                rp.draw_model(model1, rp.Vector3(position[i][0]/1000, position[i][1]/1000, position[i][2]/1000), 1, rp.WHITE)
                rp.draw_model(model2, rp.Vector3(0, 0, 0), 1, rp.WHITE)
                #rp.draw_model_ex(model2, rp.Vector3(-0.5, 0.0, 0.0), Vector3(1.0, 1.0, 0.0), 50, Vector3(1.0, 1.0, 1.0), WHITE)
                # draw_model(model3, Vector3(0.0, 0.0, -1.5), 1, WHITE)
                
                rp.draw_grid(10, 1.0)

            # end_mode3d()

            # rp.draw_rectangle(16, 698, rp.measure_text(f"Frame: {framesCounter}", 20) + 8, 42, rp.BLUE)
            # rp.draw_text(f"Frame: {framesCounter}", 20, 700, 20, rp.WHITE)

            rp.draw_fps(10, 10)
            #rp.draw_text(str(Q),10,10,12,rp.WHITE)
            #rp.draw_text(str(t_aux[i]),10,10,12,rp.WHITE)
            rp.draw_text_ex(default_font,"El tiempo es: " + str(t_aux[i]) + "[s]",rp.Vector2(10, 40),12,5,rp.WHITE)
        # end_drawing()
        i = i+1
        if i > len(q0_rot):
            i = 0
            
    rp.unload_model(model1)
    # rp.unload_model(model2)
    # rp.unload_model(model3)

    rp.unload_texture(texDiffuse)
    # rp.unload_texture(texMask)

    # rp.unload_shader(shader)

    rp.close_window()

    return 0


if __name__ == "__main__":
    if len(sys.argv) >= 2 and isinstance(sys.argv[1], str):
        os.chdir(sys.argv[1])
    print("Working dir:", os.getcwd())
    sys.exit(main())