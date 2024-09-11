# Sección 01: Obtencion de posicion, velocidad y vectores orbitales
# Datos_ECI_SGP4.py
Es el modulo de python para obtener los vectores posicion, velocidad, fuerzas magneticas y sol que afectan al satélite en el marco de referencia inercial (ECI).
### Librerias necesarias para utilizar el módulo
    pip install skyfield
    pip install numpy
    pip install matplotlib
    pip install sgp4
    pip install pandas

### Importacion de librerias:
    from skyfield.api import utc
    from datetime import datetime, timedelta
    from sgp4.earth_gravity import wgs84
    from sgp4.io import twoline2rv
    import pyIGRF
    import numpy as np
    import functions_01
    import matplotlib.pyplot as plt
    import pandas as pd

## Explicación por secciones
Se explica el codigo hecho en base a la separacion de celdas que presenta el codigo.
### Define la ubicación de tu archivo TLE y encuentra la condicion inicial del estado
    Para inicializar la simulación, se debe entregar un .txt con un TLE de un CubeSat. En el caso del ejemplo realizado se tiene un TLE llamado 'suchai.txt', el cual se carga mediante el comando básico de python 'open'.

    Posteriormente, para obtener el vector posicion y velocidad inicial del satélite, se requiere convertir las líneas del archivo en un objeto Satrec utilizando el comando "twoline2rv". Tambien debe entregarse una fecha inicial mediante el datetime de python para la primera propagación utilizando a propagacion SGP4.

    Luego, se convierte este primer vector posicion y velocidad en coordenada GPS (latitud[°], longitud[°] y altitud[m]) utilizando la funcion propia 'eci2lla', para así obtener la fuerza magnetica inicial con la cual esta interactuando el satélite utilizando la libreria pyIGRF (cuya explicación se presenta por proveedor al final de este documento).

    Además, utilizando la funcion propia 'datetime_to_jd2000', se transforma la fecha inicial a Julian Date 2000. Con esto, se puede obtener el vector sol mediante la funcion 'sun_vector'. Finalmente se entrega la variable del tiempo a propagar la simulacion orbital a diferentes pasos de tiempo.


### Listas donde guardar las variables ECI y body
    Se guardan en listas las variables de posicion, velocidad, latitud, longitud, altitud, fuerzas magneticas en x, fuerzas magneticas en y, fuerzas magneticas en z, la norma de cada componente de las fuerzas magneticas y su símil en los vectores sol.

### Propagacion SGP4 y obtencion de vectores magneticos y sol en ECI a través del tiempo
    Se propaga a traves del tiempo con un delta_t a seleccion hasta el tiempo de propagacion seleccionado. Esta propagación sigue los mismos pasos de la condicion inicial dentro del ciclo while.

    Se pasan las listas a array(elección personal) para posteriormente graficar a traves del tiempo los vectores posicion, velocidad, fuerzas magneticas y sol tanto normalizado como su valor real en el marco de referencia inercial, utilizando la libreria de matplotlib.

### Guardar datos en un archivo .csv
    Mediante la libreria de 'pandas' se pasan los arrays obtenidos en la simulacion para guardarlos en un .csv cuyo nombres es dado en la variable 'archivo'. Este archivo servirá para obtener en el modulo de python 'B_ss_ECI_orbit' los vectores de fuerzas magnetico y sol en el marco de referencia orbital (cuyo eje yaw siempre apunta hacia el centro de la tierra o nadir).

# B_ss_ECI_orbit.py
Es el modulo que rota los vectores de fuerza magnetica y vector sol obtenidos en 'Datos_ECI_SGP4.py' desde el marco de referencia inercial (ECI) al marco de referencia orbital o Roll-Pitch-Yaw (RPY), el cual su eje Yaw apunta siempre hacia el centro de la Tierra.

### Librerias necesarias para utilizar el módulo
    pip install numpy
    pip install matplotlib
    pip install pandas
    pip install scypy

### Importacion de librerias:
    import functions_01
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.spatial.transform import Rotation

## Explicación por secciones
Se explica el codigo hecho en base a la separacion de celdas que presenta el codigo.

### Cargar datos del .csv obtenido
    Utilizando la libreria 'pandas' se leen los archivos del .csv seleccionado, para posteriormente guardarlos como arrays. Teniendo ya los datos guardados, se obtienen las variables 'X_orbits', 'Y_orbits' y 'Z_orbits', los cuales son ecuaciones utiles a utilizar para hacer la rotacion en el sistema de referencia. Estas dependen del vector posicion y velocidad del satélite.

### Obtención de los vectores en marco de referencia orbit
    Se inicializa el código generando listas vacias donde guardar los cuaterniones que rotan desde ECI a orbit. Tambien donde guardar los vectores de fuerza magnetica y vector sol en el marco de referencia orbital.

    Luego se realiza un ciclo for para obtener en cada paso de tiempo la rotacion de las fuerzas magneticas y vector sol en el marco de referencia orbital, por toda la propagación obtenida del .csv cargado. En cada paso de tiempo, se obtiene la matriz de rotación Rs_ECI_orbit, el cual es la transposicion de los vectores X_orbits, Y_orbits y Z_orbits y permite rotar de ECI a orbit. Esta matriz se pasa a cuaternion mediante la funcion proveniente de la libreria 'scipy' llamada 'Rotation, teniendo en cuenta que entregara un cuaternion pasivo (con el escalar en el cuarto termino). 

    Estos cuaterniones obtenidos sirven para rotar los vectores de fuerza magnetica y sol. Para ello, se transforma el vector a rotar en un cuaternion puro (con escalar igual a 0), ademas de obtener la inversa del cuaternion utilizando la funcion 'inv_q'. Luego, se aplica multiplicacion de cuaterniones entre el cuaternion de rotacion, el vector a rotar hecho cuaternion y la inversa del cuaternion, en el orden señalado, lo que representa la rotacion del vector.

    Finalmente, se obtienen las normas, para comparar su cambio en el tiempo.

### Gráficas de los vectores rotados a orbit
    Se grafican los vectores fuerza magnetica y vector sol en todo el periodo de tiempo simulado.

### Guardar vectores orbit en un .csv
    Mediante la libreria de 'pandas' se pasan los arrays obtenidos en la simulacion para guardarlos en un .csv cuyo nombres es dado en la variable 'archivo_c'. Este archivo servirá para otorgar los vectores basado en la dinamica orbital en el amrco de referencia orbital en el periodo de tiempo propagado. En los sigueintes modulos de python debe seguir restrictivamente un tiempo de propagacion menor al .csv 'Vectores_orbit_ECI.csv' con el mismo delta_t para todos los procedimientos de determinacion y control de actitud.

# functions_01.py
Este modulo de python guarda las funciones creadas para el desarrollo de esta sección. Gracias a esto, se logro tener modulos cortos y simples de comprender en la Sección 01.

### Librerias necesarias para utilizar el módulo
    pip install skyfield
    pip install numpy

### Importacion de librerias:
    from skyfield.positionlib import Geocentric
    import numpy as np
    from datetime import datetime

## Explicación por funciones
Se explican de manera general el uso y creacion de las funciones locales hechas por el desarrollador.

### inv_q(q):
    La inversa de un cuaternion es equivalente a tener la parte vectorial negativa, manteniendo el escalar igual.
### datetime_to_jd2000(fecha):
    Aqui se debe entregar la fecha en formato 'datetime' en el orden YYYY/MM/DD/HH/MM/SS, la cual se restará con el inicio de los Julian Date, equivalente al 01 de enero del 2000 a las 12 del mediodia, para conocer la cantidad de dias ocurridos desde esa referencia. Posteriormente se divide en la cantidad de segundos en un dia para obtener el resultado en segundos.
### sun_vector(jd2000):
    Esta es la funcion para obtener las componentes del vector sol leidas en el marco de referencia inercial (ECI) desde el satélite. La variable a entregar debe ser la fecha en Julian Date, para posteriormente aplicar las ecuaciones de meridiano y lambda del sol.
### quaternion_to_euler(q):
    Al entregar un cuaternion, mediante ecuaciones de transformacion, se pasan a angulos de Euler, teniendo los cuidados necesarios en angulos criticos, como lo son el 90° y el 270°.
### quat_mult(dqk,qk_priori):
    Permite la multiplicacion de dos cuaterniones ingresados como entrada.


# pyIGRF
## What is pyIGRF?  
    This is a package of IGRF-13 (International Geomagnetic Reference Field) about python version. 
    We can calculate magnetic field intensity and transform coordinate between GeoGraphical and GeoMagnetic.
    It don't need any Fortran compiler. But it needs NumPy package.  

## How to Install?
    Download this package and run install.
    ```python setup.py install```

## How to Use it?
    First import this package.  
    ```import pyIGRF```

    You can calculate magnetic field intensity.   
    ```pyIGRF.igrf_value(lat, lon, alt, date)```

    or calculate the annual variation of magnetic filed intensity.  
    ```pyIGRF.igrf_variation(lat, lon, alt, date)```

    the response is 7 float number about magnetic filed which is:  
    - D: declination (+ve east)
    - I: inclination (+ve down)
    - H: horizontal intensity
    - X: north component
    - Y: east component
    - Z: vertical component (+ve down)
    - F: total intensity  
    *unit: degree or nT*

    If you want to use IGRF-13 more flexibly, you can use module *calculate*. 
    There is two function which is closer to Fortran. You can change it for different coordination.
    ```from pyIGRF import calculate```  

    Another module *load_coeffs* can be used to get *g[m][n]* or *h[m][n]* same as that in formula.
    ```from pyIGRF.load_coeffs import get_coeffs``` 



## Model Introduction and igrf13-coeffs File Download
    https://www.ngdc.noaa.gov/IAGA/vmod/igrf.html
