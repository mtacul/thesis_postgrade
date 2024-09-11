import os
import re

def rename_files(directory, extension):
    # Compilar una expresión regular que coincida con la fecha al inicio del nombre del archivo
    # Asumiendo que la fecha está en formato YYYY-MM-DD
    date_pattern = re.compile(r'_2')

    # Cambiar al directorio especificado
    os.chdir(directory)

    # Iterar sobre cada archivo en el directorio
    for filename in os.listdir():
        # Verificar si el archivo tiene la extensión deseada
        if filename.endswith(extension):
            # Buscar la fecha al inicio del nombre del archivo
            new_name = date_pattern.sub('', filename)
            # Renombrar el archivo si se encontró una fecha al inicio
            if new_name != filename:
                os.rename(filename, new_name)
                print(f'Renamed "{filename}" to "{new_name}"')

# Usar la función con el directorio y la extensión deseados
directory_path = r'C:\Users\nachi\Desktop\thesis_git\thesis_postgrade\07_Tratamiento de datos'
file_extension = '.csv'  # Cambiar por la extensión deseada
rename_files(directory_path, file_extension)
