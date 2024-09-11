# Sección 02: Diferencias entre los modelos dinámicos obtenidos.


# test_mod_dinamicos.py / test_mod_dinamicos_rw.py
Es el modulo de python que prueba los modelos dinamicos lineal continuo, lineal discreto y no lineal continuo utilizando como actuador tres magnetorquers / ruedas de reacción y una acción de control constante. Para las diferencias entre los modulos, se mostrara antes de la explicacion el nombre encerrado entre los siguiente simbolos /''\
### Librerias necesarias para utilizar el módulo
    pip install numpy
    pip install matplotlib
    pip install pandas

### Importacion de librerias:
    import functions_02        /'test_mod_dinamicos.py'\
    import functions_02_rw     /'test_mod_dinamicos_rw.py'\
    import numpy as np   
    import pandas as pd
    import matplotlib.pyplot as plt

## Explicación por secciones
Se explica el codigo hecho en base a la separacion de celdas que presenta el codigo.
### Cargar datos del .csv obtenido
    Utilizando la libreria 'pandas' se leen los archivos del .csv seleccionado, para posteriormente guardar como arrays los vectores sol y fuerza magnetico en el marco de referencia orbital. Además, se crea una nueva variable de tiempo llamada 't' menor a 't_aux' para un menor tiempo de simulación.


### Condiciones iniciales y parámetros geométricos dados
    / 'test_mod_dinamicos.py' \
    Se entregan las condiciones iniciales para los sistemas. En este caso un cuaternion y una velocidad angular en sus tres componentes. Además, se entrega valores de los momentos principales de inercia y un tamaño de paso para la resolucion de las ecuaciones diferenciales de la siguiente sección.

    / 'test_mod_dinamicos_rw.py' \
    Se entregan las condiciones iniciales para los sistemas. En este caso un cuaternion, una velocidad angular del satélite en sus tres componentes y la velocidad angular individual de cada rueda de reaccion. Además, se entrega valores de los momentos principales de inercia tanto del satélite como de las ruedas de reacción, la distancia entre el origen del satelite y el origen de las ruedas de reaccion alineadas en cada eje del cuerpo (variable 'b'), las masas de las ruedas de reaccion, el calculo del segundo momento de inercia total y finalmente un tamaño de paso para la resolucion de las ecuaciones diferenciales de la siguiente sección.

### Simulación de los modelos dinámico lineales y no lineales
    
    / 'test_mod_dinamicos.py' \
    Se crean listas en las cuales se guardara en cada paso de tiempo los cuaterniones y las velocidades angulares. En detalle, son 7 listas para el modelo dinamico lineal continuo, 7 listas para el modelo dinamico discreto y 7 listas para el modelo dinámico no lineal continuo.

    / 'test_mod_dinamicos_rw.py' \
    Se crean listas en las cuales se guardara en cada paso de tiempo los cuaterniones y las velocidades angulares del satélite y de cada rueda de reacción. En detalle, son 10 listas para el modelo dinamico lineal continuo, 10 listas para el modelo dinamico discreto y 10 listas para el modelo dinámico no lineal continuo.
    
    Luego, se crea un ciclo for para analizar en cada paso de tiempo como afecta la accion de control u constante en cada modelo. Para diferenciarlos entre si, se muestra el ejemplo de los cuaterniones, donde 'qq' es lineal continuo, 'qq_disc' es lineal discreto y 'qq_nl' es no lineal. Por cada modelo se genera una rotacion propia desde el marco de referencia orbital al del cuerpo en el vector sol y fuerzas magneticas utilizando la funcion 'rotacion_v'.

    Posteriormente. se entregan todos los parámetros necesarios de entrada en la funcion A_B, para obtener las matrices de estado (A) y de control (B), las cuales son utiles en los modelos lineales continuos y discretos.

    Finalmente, se resuelven los tres sistemas dinamicos. Cada uno de los casos analizados presenta su propia funcion que resuelve su respectivo sistema de ecuaciones diferenciales. Para el caso continuo lineal esta 'mod_lineal_cont', para el caso discreto lineal esta 'mod_lineal_disc' y finalmente para el no lineal esta 'mod_nolineal', la cual devuelve el vector estado x y el termino escalar del cuaternion (no incluido dentro del vector estado). 
    
    Es importante mencionar que no se utiliza el termino escalar para general el modelo dinamico debido a que si se incluye, el sistema no seria observable al tener siete estados, el cual es mayor al rango de la combinacion lineal entre A y C, que es la matriz de medición, el cual se calculará en los analisis de determinacion de actitud.

### Gráficas de los cambios en los cuaterniones con accion de control constante

    Se observan mediante gráficas de la libreria 'matplotlib' la similitud entre los modelos, para comprbar si presentan comportamientos similares en cortos periodos de tiempo.

### Calculo del Mean Square Error

    Se analiza de manera cuantitativa la diferencia entre los modelos mediante el calculo del Mean Square Error (MSE).

# functions_02 / functions_02_rw
Estos modulos de python guarda las funciones creadas para el desarrollo de esta sección para magnetorquers y rueda de reaccion por separado. Gracias a esto, se logro tener modulos cortos y simples de comprender en la Sección 02. Para las diferencias entre los modulos, se mostrara antes de la explicacion el nombre encerrado entre los siguiente simbolos /''\

### Librerias necesarias para utilizar el módulo
    pip install control
    pip install numpy

### Importacion de librerias:
    import numpy as np
    import control as ctrl
    import math

## Explicación por funciones
Se explican de manera general el uso y creacion de las funciones locales hechas por el desarrollador.

### inv_q(q):
    La inversa de un cuaternion es equivalente a tener la parte vectorial negativa, manteniendo el escalar igual.

### quat_mult(dqk,qk_priori):
    Permite la multiplicacion de dos cuaterniones ingresados como entrada.

### simulate_magnetometer_reading(B_body, ruido):
    Genera ruido blanco y se le agrega al vector 'B_body'. Esto basado en la desviacion estandar del sensor.

### rotacion_v(q, b_i, sigma):
    Genera una rotacion en base al cuaternion 'q' al vector 'b_i' para obtenerla en el sistema de referencia del cuerpo. Se utiliza en conjunto con 'simulate_magnetometer_reading' para aplicarle el ruido.

### A_PD(I_x,I_y,I_z,w0_O, w0,w1,w2):
                                / 'functions_02.py' \
    Con esta funcion se puede obtener A, que es la matriz linealizada de la funcion dinamica no lineal derivada respecto al vector estado y evaluada en el punto de equilibrio x = [0,0,0,0,0,0] (tres primeros cuaterniones y tres componentes de velocidad angular). Solo se evaluan las entradas, que son parametros orbitales y geometricos.

### A_PD(I_x,I_y,I_z,w0_O, w0,w1,w2, I_s0_x, I_s1_y, I_s2_z, w_s0,w_s1,w_s2, J_x, J_y, J_z):
                                / 'functions_02_rw.py' \
    Con esta funcion se puede obtener A, que es la matriz linealizada de la funcion dinamica no lineal derivada respecto al vector estado y evaluada en el punto de equilibrio x = [0,0,0,0,0,0,0,0,0] (tres primeros cuaterniones, tres componentes de velocidad angular y las tres velocidades angulares de cada rueda de reaccion). Solo se evaluan las entradas, que son parametros orbitales y geometricos.

### B_PD(I_x,I_y,I_z,B_magnet):
                                / 'functions_02.py' \
    Con esta funcion se puede obtener B, que es la matriz linealizada de la funcion dinamica nolineal derivada respecto a la accion de control y evaluada en el punto de equilibrio u(momentos dipolares) = [0,0,0]. En esta matriz, existe un cambio en los valores a traves del tiempo, ya que la matriz depende de las fuerzas magneticas (B_magnet) que cambian en el tiempo y no desaparecen con la evaluacion del vector estado en el equilibrio

### B_PD(w0_O,I_s0_x,I_s1_y,I_s2_z,J_x,J_y,J_z):
                                / 'functions_02_rw.py' \
    Con esta funcion se puede obtener B, que es la matriz linealizada de la funcion dinamica nolineal derivada respecto a la accion de control y evaluada en el punto de equilibrio u(torques rueda de reaccion) = [0,0,0].

### H_k_bar(b0,b1,b2,s0,s1,s2):
                                / 'functions_02.py' \
    Con esta funcion se puede obtener H o C, que es la matriz linealizada de la función no lineal que relaciona las mediciones con el vector estado. Esta se deriva respecto al punto de equilibrio x = [0,0,0,0,0,0] (tres primeros cuaterniones y tres componentes de velocidad angular). Se evaluan en la matriz las mediciones del magnetometro (b0,b1 y b2) y del sensor de sol (s0,s1,s2).

### H_k_bar(b0,b1,b2,s0,s1,s2,I_s0_x,I_s1_y,I_s2_z):
                                / 'functions_02_rw.py' \
    Con esta funcion se puede obtener H o C, que es la matriz linealizada de la función no lineal que relaciona las mediciones con el vector estado. Esta se deriva respecto al punto de equilibrio x = [0,0,0,0,0,0,0,0,0]  (tres primeros cuaterniones, tres componentes de velocidad angular y las tres velocidades angulares de cada rueda de reaccion). Se evaluan en la matriz las mediciones del magnetometro (b0,b1 y b2) y del sensor de sol (s0,s1,s2) y el segundo momento de inercia de cada rueda de reacción (I_s0_x,I_s1_y,I_s2_z).

###  A_B(I_x,I_y,I_z,w0_O,w0,w1,w2,deltat,h,b_orbit,b_body, s_body):
                                / 'functions_02.py' \
    Se obtiene los valores de las matrices A,B y C al evaluar las entradas de la funcion. Además, gracias a la libreria 'control' se pasan las matrices a su version discreta para su uso en el modelo lineal discreto. 

### A_B(I_x,I_y,I_z,w0_O, w0,w1,w2, I_s0_x, I_s1_y, I_s2_z, w_s0,w_s1,w_s2, J_x, J_y, J_z,deltat,h,b_orbit,b_body, s_body):
                                / 'functions_02_rw.py' \
    Se obtiene los valores de las matrices A,B y C al evaluar las entradas de la funcion. Además, gracias a la libreria 'control' se pasan las matrices a su version discreta para su uso en el modelo lineal discreto. 

### dynamics(A, x, B, u):
    Funcion que representa la ecuacion espacio estado x_dot = Ax + Bu

### rk4_step_PD(dynamics, x, A, B, u, h):
    Utilizacion del solver Runge-kutta 4 para la solucion continua del espacio estado. Este se programo para la solucion especifica de la ecuacion.

### mod_lineal_cont(x,u,deltat,h,A,B):
    Genera como salida el vector estado en el siguiente paso de tiempo (en base al deltat) al solucionar las ecuaciones diferenciales del espacio estado. Para ello se entregar el vector estado x, la accion de control u, y las matrices A y B, con un paso h. Ademas, se sabe que la parte escalar del cuaternion se obtiene mediante la relacion de 1 menos la norma de la parte vectorial en raiz. Esta relacion esta restringida, de tal manera de evitar que exista una raiz negativa.


### mod_lineal_disc(x,u,deltat, h,A_discrete,B_discrete):
    Genera como salida el vector estado en el siguiente paso de tiempo (en base al deltat) al solucionar las ecuaciones discretas del espacio estado. Para ello se entregar el vector estado x, la accion de control u, y las matrices discretas A y B, con un paso h. Ademas, se sabe que la parte escalar del cuaternion se obtiene mediante la relacion de 1 menos la norma de la parte vectorial en raiz. Esta relacion esta restringida, de tal manera de evitar que exista una raiz negativa.


### f1_K(t, q0, q1, q2,q3, w0, w1, w2):
    Ecuacion diferencial no lineal del q0_dot. Relaciona los marcos de referencia orbit y body
### f2_K(t, q0, q1, q2,q3, w0, w1, w2):
    Ecuacion diferencial no lineal del q1_dot. Relaciona los marcos de referencia orbit y body
### f3_K(t, q0, q1, q2,q3, w0, w1, w2):
    Ecuacion diferencial no lineal del q2_dot. Relaciona los marcos de referencia orbit y body
### f4_K(t, q0, q1, q2,q3, w0, w1, w2):
    Ecuacion diferencial no lineal del q3_dot. Relaciona los marcos de referencia orbit y body
### f5_K(t, q0, q1, q2,q3, w0, w1, w2,w0_o,tau_x_ctrl,tau_x_per,I_x,I_y,I_z):
    Ecuacion diferencial no lineal del w0_dot. Relaciona los marcos de referencia orbit y body
### f6_K(t, q0, q1, q2,q3, w0, w1, w2,w0_o,tau_x_ctrl,tau_x_per,I_x,I_y,I_z):
    Ecuacion diferencial no lineal del w1_dot. Relaciona los marcos de referencia orbit y body
### f7_K(t, q0, q1, q2,q3, w0, w1, w2,w0_o,tau_x_ctrl,tau_x_per,I_x,I_y,I_z):
    Ecuacion diferencial no lineal del w2_dot. Relaciona los marcos de referencia orbit y body
### rk4_EKF_step(t, q0, q1, q2,q3, w0, w1, w2, h, w0_o,tau_x_ctrl,tau_x_per,tau_y_ctrl,tau_y_per,tau_z_ctrl,tau_z_per,I_x,I_y,I_z):
    Utilizacion del solver Runge-kutta 4 para la solucion continua de las 7 ecuaciones diferenciales no lineales simultaneamente. Este se programo para la solucion especifica de la ecuacion en un paso.

### mod_nolineal(x,u,deltat, b,h,w0_O,I_x,I_y,I_z):
    Genera como salida el vector estado en el siguiente paso de tiempo (en base al deltat) al solucionar las ecuaciones diferenciales no lineales. Para ello se entrega el vector estado x, la accion de control u, las fuerzas magneticas en el marco de referencia del cuerpo y un paso h para su uso en el propagador. Ademas, se sabe que la parte escalar del cuaternion se obtiene mediante la relacion de 1 menos la norma de la parte vectorial en raiz. Esta relacion esta restringida, de tal manera de evitar que exista una raiz negativa.

### cuat_MSE_NL(q0,q1,q2,q3,w0,w1,w2,q0_nl,q1_nl,q2_nl,q3_nl,w0_nl,w1_nl,w2_nl):
    Calcula el mean square error entre dos modelos. Esto en base a los diferentes cuaterniones calculados en la entrada.