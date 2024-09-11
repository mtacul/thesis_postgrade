# Sección 03: Suite de simulación con modelos lineales discretos


# suite_MT_LQR.py / suite_RW_LQR
Es el modulo de python que realiza la dinámica rotacional implementando como sensores los magnetometros, sensores de sol y giroscopios. El algoritmo de determinación de actitud es un filtro de kalman semiextendido. Las opciones de actuador son la implementacion de tres magnetorquers o 3 ruedas de reacción. El controlador utilizado es un LQR. Todo se realiza con los modelos dinámicos continuos discretos. Para las diferencias entre los modulos (que se diferencian en el uso de uno u otro actuador, siendo MT magnetorquer y RW rueda de reacción), se mostrara antes de la explicacion el nombre del modulo encerrado entre los siguiente simbolos /''\

### Librerias necesarias para utilizar el módulo
    pip install numpy
    pip install matplotlib
    pip install pandas
    pip install scipy

### Importacion de librerias:
    import functions_03        /'suite_MT_LQR.py'\
    import functions_03_rw     /'suite_RW_LQR.py'\
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.linalg import solve_discrete_are

## Explicación por secciones
Se explica el codigo hecho en base a la separacion de celdas que presenta el codigo.
### Cargar datos del .csv obtenido
    Utilizando la libreria 'pandas' se leen los archivos del .csv seleccionado, para posteriormente guardar como arrays los vectores sol y fuerza magnetico en el marco de referencia orbital. Además, se crea una nueva variable de tiempo llamada 't' menor a 't_aux' para un menor tiempo de simulación.


### Parámetros geométricos y orbitales dados
    / 'suite_MT_LQR.py' \
    Se entregan valores de los momentos principales de inercia, la velocidad angular orbital del satélite y un tamaño de paso para la resolucion de las ecuaciones diferenciales de la siguiente sección.

    / 'suite_RW_LQR.py' \
    Se entregan valores de los momentos principales de inercia tanto del satélite como de las ruedas de reacción, la distancia entre el origen del satelite y el origen de las ruedas de reaccion alineadas en cada eje del cuerpo (variable 'b'), las masas de las ruedas de reaccion, el calculo del segundo momento de inercia total y finalmente un tamaño de paso para la resolucion de las ecuaciones diferenciales de la siguiente sección.

### Seleccion del nivel del sensor
    Esta parte del codigo esta hecha para hacer elegir al usuario que nivel de sensores quiere probar en la suite. Para ello debe colocar un numero entre el 1, 2 o 3, siendo el mas bajo un sensor de peor calidad (mayor desviacion estandar). La eleccion es general, osea que si se selecciona un nivel de sensor, este nivel sera para el magnetometro, sensor de sol y giroscopio.

### Seleccion del nivel de actuador
    Esta parte del codigo esta hecha para hacer elegir al usuario que nivel de actuadores quiere probar en la suite. Para ello debe colocar un numero entre el 1, 2 o 3, siendo el mas bajo un actuador de peor calidad (menor limite maximo y minimo en la accion de control).
### Condiciones iniciales reales y estimadas
    / 'suite_MT_LQR.py' \
    Se entregan las condiciones iniciales para el sistema. En este caso un cuaternion y una velocidad angular en sus tres componentes 'real' y 'estimada'. Se hace la diferenciacion debido a que las condiciones iniciales reales inicializan para la obtencion del vector estado proveniente del modelo, en donde la velocidad angular se le aplica ruido blanco para simular el realismo del giroscopio. Por otro lado, el 'estimado' es el cuaternion y velocidad angular obtenido por el filtro bayesiano de kalman semiextendido en base a las mediciones de los sensores. Para obtener los valores de la simulación, se crean listas a rellenar para cada uno de los componentes mencionados.Además, se obtiene la primera medición del magnetometro y sensor de sol en base a la condición inicial del cuaternion.

    / 'suite_RW_LQR.py' \
    Se entregan las condiciones iniciales para los sistemas. En este caso un cuaternion, una velocidad angular del satélite en sus tres componentes y la velocidad angular individual de cada rueda de reaccion 'real' y 'estimada'. Se hace la diferenciacion debido a que las condiciones iniciales reales inicializan para la obtencion del vector estado proveniente del modelo, en donde la velocidad angular se le aplica ruido blanco para simular el realismo del giroscopio. Por otro lado, el 'estimado' es el cuaternion, velocidad angular del satélite y velocidad angular de cada rueda de reacción obtenido por el filtro bayesiano de kalman semiextendido en base a las mediciones de los sensores. Para obtener los valores de la simulación, se crean listas a rellenar para cada uno de los componentes mencionados. Además, se obtiene la primera medición del magnetometro y sensor de sol en base a la condición inicial del cuaternion.

### Obtencion de un B_prom representativo (solo suite_MT_LQR.py)
    En primera instancia, se obtiene las matrices A y B del espacio estado mediante la funcion 'A_B', guardando en una lista el primer B discretizado, en base a la condicion inicial. Posteriormente, utilizando un ciclo for, se guardan las matrices B discretizadas obtenidas en el paso de una orbita del satélite (en el caso del ejemplo son 5764 segundos). Esto se realiza ya que despues de una orbita, se repite el mismo ciclo de matrices B, al ser dependientes de las fuerzas amgneticas, que a su vez varian en base a la posicion del satélite respecto a la Tierra. Finalmente, se obtiene el B_prom, que seria la matriz discretizada representativa para ejercer el control, el cual se calcula como un promedio de cada componente de todas las matrices B guardadas (Esto no se realiza en la amtriz A al no depender de vectores externo al vector estado, el cual esta evaluado en el equilibrio).

    Para el caso del modulo 'suite_RW_LQR' las matrices A y B son constantes en el tiempo, por lo que solo se obtienen con la funcion 'A_B'.

### Control LQR
    El control a realizar en estos modulos es el Linear Quadratic Regulator (LQR). Esta es un método óptimo de control utilizado para sistemas lineales. Su objetivo es encontrar una ley de control que minimice una función de costo cuadrática, que equilibra el desempeño del sistema con el esfuerzo de control. Dentro de esta funcion de costo se encuentran las matrices Q y R, en donde la matriz Q pondera el costo del estado del sistema y la matriz R Pondera el costo del esfuerzo de control . Con esto, se puede obtener la matriz P mediante la ecuacion de Riccati y genera una ley de control en forma de retroalimentación de estados 𝑢 =−𝐾𝑥, donde 𝐾 es la matriz de ganancia calculada para minimizar el costo. 
    
    Para realizarlo en Python, se debe entregar las matrices Q y R en base a los requerimientos del diseñador, para posteriormente calcular la matriz P utilizando la funcion 'solve_discrete_are' entregando como entrada A,B,Q y R. Finalmente se obtiene la matriz de ganancia K multiplicando como aparece en el codigo.

### Simulación dinámica de actitud
    
    Para inicializar antes de empezar el ciclo for con la obtencion del vector estado en cada paso de tiempo, se entrega una condicion inicial para la matriz de covarianza P, la cual es utilizada para el filtro de kalman semiextendido. Tambien se considera el uso de una semilla 'np.random.seed(42)' para tener un tipo de numero aleatorios en el ruido del sensor.

    Una vez inicializado el ciclo for, se observa la creacion de los vectores estado estimados, en conjunto con la accion de control estimada al multiplicarla con la matriz de ganancia K. Esta accion de control se limita segun el nivel de actuador seleccionado por el usuario al aplicarle la funcion 'torquer'. 
    
    Posteriormente se observa la obtencion de los vectores de estado 'reales' en el siguiente paso de tiempo al utilizar la funcion 'mod_lineal_disc'. De aqui se observa la entrega de 'u_est' como entrada a la funcion, la cual se debe a que las acciones de control "realistas" se obtienen del vector estimado por las mediciones de los sensores y el algoritmo de estimacion. Finalmente para la parte 'real', se reemplazan los nuevos valores en 'x_real' para seguir la iteracion en cada paso de tiempo, guardandolos tambien en las listas correspndientes con el .append.

    En la siguiente parte del codigo dentro del ciclo for, se observa la rotacion de los vector sol y fuerza magnetico desde orbit a body utilizando un cuaternion 'real' y agregandole el ruido blanco basado en la seleccion de la desviacion estandar del usuario para simular el sensor sol y magnetometro. Con esto se obtienen las matrices A y B discretas utilizando la funcion A_B.

    Estos valores, en conjunto con las estimaciones guardadas al inicio del ciclo for, se ponen como entrada en la funcion 'kalman_lineal' para obtener el vector estado estimado en el siguiente paso de tiempo. Se dividio en un if/else donde la opcion 4 del usuario es probar sin ruido los sensores y ver si es igual e modelo al fitro de kalman. Cualquier otra opcion genera los distintos niveles de ruido basados en la eleccion. Finalmente se guardan los valores en sus respectivas listas y se reinicia el valor de la matriz de covarianza para volver a empezar el ciclo.

    Este proceso es el mismo para ambos modulos, cambiando solamente los vectores estado y las funciones de modelo dinamico explicadas en la seccion 'functions_03' y 'functions_03_rw'

    Al final de la seccion se obtienen los mean square error entre los estimados y reales tanto de los cuaterniones como de los angulos de Euler con las funciones 'cuat_MSE_NL' y 'RPY_MSE'

### Gráficas de los resultados obtenidos en la suite de simulacion

    Se observan mediante gráficas de la libreria 'matplotlib' las gráficas obtenidas por la suite de simulación

### Guardar resultados en un .csv
    Mediante la libreria de 'pandas' se pasan los arrays obtenidos en la simulacion para guardarlos en un .csv cuyo nombres es dado en la variable 'nombre_archivo' y depende de la eleccion del usuario en nivel de sensor y actuador. Este archivo servirá para otorgar los resultados a la seccion de tratamientos de datos y obtener los parametros de rendimiento.

# suite_MT_LQR.py
Es el modulo de python que realiza la dinámica rotacional implementando como sensores los magnetometros, sensores de sol y giroscopios. El algoritmo de determinación de actitud es un filtro de kalman semiextendido. Las opciones de actuador son la implementacion de tres magnetorquers. El controlador utilizado es un PD. Todo se realiza con los modelos dinámicos continuos discretos.

## Explicación por secciones
Este modulo presenta las mismas secciones aplicada en 'suite_MT_LQR' a diferencia del control, la cual sera explicada a continuación:

### Control PD
    En el caso expuesto en el codigo se presenta una matriz K con valores de nxm, donde n es la cantidad de acciones de control y m es la cantidad de vectores de estado. Esta matriz presenta solo valores en las diagonales de las submatrices cuadradas 3x3 de K. Estos valores fueron calculados utilizando la funcion opt_K, ingresando una condicion inicial.

# functions_03 / functions_03_rw
Estos modulos de python guarda las funciones creadas para el desarrollo de esta sección para magnetorquers y rueda de reaccion por separado. Gracias a esto, se logro tener modulos cortos y simples de comprender en la Sección 03. Para las diferencias entre los modulos, se mostrara antes de la explicacion el nombre encerrado entre los siguiente simbolos /''\

### Librerias necesarias para utilizar el módulo
    pip install control
    pip install numpy
    pip install sigfig
    pip install scipy
    
### Importacion de librerias:
    import numpy as np
    import control as ctrl
    import math
    from scipy.optimize import minimize
    from sigfig import round

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

### simulate_gyros_reading(w,ruido,bias):
    Genera ruido blanco y el bias, agregandose al vector 'B_body'. Esto basado en la desviacion estandar del sensor.

### A_PD(I_x,I_y,I_z,w0_O, w0,w1,w2):
                                / 'functions_03.py' \
    Con esta funcion se puede obtener A, que es la matriz linealizada de la funcion dinamica no lineal derivada respecto al vector estado y evaluada en el punto de equilibrio x = [0,0,0,0,0,0] (tres primeros cuaterniones y tres componentes de velocidad angular). Solo se evaluan las entradas, que son parametros orbitales y geometricos.

### A_PD(I_x,I_y,I_z,w0_O, w0,w1,w2, I_s0_x, I_s1_y, I_s2_z, w_s0,w_s1,w_s2, J_x, J_y, J_z):
                                / 'functions_03_rw.py' \
    Con esta funcion se puede obtener A, que es la matriz linealizada de la funcion dinamica no lineal derivada respecto al vector estado y evaluada en el punto de equilibrio x = [0,0,0,0,0,0,0,0,0] (tres primeros cuaterniones, tres componentes de velocidad angular y las tres velocidades angulares de cada rueda de reaccion). Solo se evaluan las entradas, que son parametros orbitales y geometricos.

### B_PD(I_x,I_y,I_z,B_magnet):
                                / 'functions_03.py' \
    Con esta funcion se puede obtener B, que es la matriz linealizada de la funcion dinamica nolineal derivada respecto a la accion de control y evaluada en el punto de equilibrio u(momentos dipolares) = [0,0,0]. En esta matriz, existe un cambio en los valores a traves del tiempo, ya que la matriz depende de las fuerzas magneticas (B_magnet) que cambian en el tiempo y no desaparecen con la evaluacion del vector estado en el equilibrio

### B_PD(w0_O,I_s0_x,I_s1_y,I_s2_z,J_x,J_y,J_z):
                                / 'functions_03_rw.py' \
    Con esta funcion se puede obtener B, que es la matriz linealizada de la funcion dinamica nolineal derivada respecto a la accion de control y evaluada en el punto de equilibrio u(torques rueda de reaccion) = [0,0,0].

### H_k_bar(b0,b1,b2,s0,s1,s2):
                                / 'functions_03.py' \
    Con esta funcion se puede obtener H o C, que es la matriz linealizada de la función no lineal que relaciona las mediciones con el vector estado. Esta se deriva respecto al punto de equilibrio x = [0,0,0,0,0,0] (tres primeros cuaterniones y tres componentes de velocidad angular). Se evaluan en la matriz las mediciones del magnetometro (b0,b1 y b2) y del sensor de sol (s0,s1,s2).

### H_k_bar(b0,b1,b2,s0,s1,s2,I_s0_x,I_s1_y,I_s2_z):
                                / 'functions_03_rw.py' \
    Con esta funcion se puede obtener H o C, que es la matriz linealizada de la función no lineal que relaciona las mediciones con el vector estado. Esta se deriva respecto al punto de equilibrio x = [0,0,0,0,0,0,0,0,0]  (tres primeros cuaterniones, tres componentes de velocidad angular y las tres velocidades angulares de cada rueda de reaccion). Se evaluan en la matriz las mediciones del magnetometro (b0,b1 y b2) y del sensor de sol (s0,s1,s2) y el segundo momento de inercia de cada rueda de reacción (I_s0_x,I_s1_y,I_s2_z).

###  A_B(I_x,I_y,I_z,w0_O,w0,w1,w2,deltat,h,b_orbit,b_body, s_body):
                                / 'functions_03.py' \
    Se obtiene los valores de las matrices A,B y C al evaluar las entradas de la funcion. Además, gracias a la libreria 'control' se pasan las matrices a su version discreta para su uso en el modelo lineal discreto. 

### A_B(I_x,I_y,I_z,w0_O, w0,w1,w2, I_s0_x, I_s1_y, I_s2_z, w_s0,w_s1,w_s2, J_x, J_y, J_z,deltat,h,b_orbit,b_body, s_body):
                                / 'functions_03_rw.py' \
    Se obtiene los valores de las matrices A,B y C al evaluar las entradas de la funcion. Además, gracias a la libreria 'control' se pasan las matrices a su version discreta para su uso en el modelo lineal discreto. 

### dynamics(A, x, B, u):
    Funcion que representa la ecuacion espacio estado x_dot = Ax + Bu

### rk4_step_PD(dynamics, x, A, B, u, h):
    Utilizacion del solver Runge-kutta 4 para la solucion continua del espacio estado. Este se programo para la solucion especifica de la ecuacion.

### mod_lineal_cont(x,u,deltat,h,A,B):
    Genera como salida el vector estado en el siguiente paso de tiempo (en base al deltat) al solucionar las ecuaciones diferenciales del espacio estado. Para ello se entregar el vector estado x, la accion de control u, y las matrices A y B, con un paso h. Ademas, se sabe que la parte escalar del cuaternion se obtiene mediante la relacion de 1 menos la norma de la parte vectorial en raiz. Esta relacion esta restringida, de tal manera de evitar que exista una raiz negativa.


### mod_lineal_disc(x,u,deltat, h,A_discrete,B_discrete):
    Genera como salida el vector estado en el siguiente paso de tiempo (en base al deltat) al solucionar las ecuaciones discretas del espacio estado. Para ello se entregar el vector estado x, la accion de control u, y las matrices discretas A y B, con un paso h. Ademas, se sabe que la parte escalar del cuaternion se obtiene mediante la relacion de 1 menos la norma de la parte vectorial en raiz. Esta relacion esta restringida, de tal manera de evitar que exista una raiz negativa.


### R_k(sigma_m, sigma_s):
    Covarianza del ruido de medición (en base a las desviaciones estandar del sensor sol y magnetometro). Modela la incertidumbre en las observaciones. Valores altos en 𝑅𝑘 indican que las mediciones son menos confiables.
### Q_k(sigma_w,sigma_bias,deltat):
    Se obtiene la Covarianza del ruido del proceso. Mide la incertidumbre en el modelo dinámico. Mientras mayor sea 𝑄𝑘, más ruido se supone que hay en la evolución del estado.
### P_k_prior(F_k, P_ki, Q_k):
    Covarianza del error de estado a priori. Mide la incertidumbre en la estimación del estado en el instante 𝑘 en la prediccion
### k_kalman(R_k, P_k_priori, H_mat):
    Se obtiene la ganancia de Kalman 𝐾𝑘, que es el factor que determina cómo se ajustará la predicción basada en la medida.
### P_posteriori(K_k,H_k,P_k_priori,R_k):
    Covarianza del error de estado a posteriori. Con esta ecuacion se actualiza para el siguiente paso de tiempo.
### kalman_lineal(A, B, C, x, u, b_orbit,b_real, s_orbit,s_real, P_ki, sigma_b, sigma_s,deltat,hh,sigma_bb,sigma_ss):
    EN la primera parte de la funcion se obtiene la matriz H o de medicion, para posteriormente calcular la prediccion del estado, donde se utiliza la funcion 'mod_lineal_disc' para la obtencion del estado a priori. Luego, con las funciones establecidas anteriormente, se obtiene la matriz Q, P_priori, R y K_k. 
    
    Luego, se obtienen con los cuaterniones a priori obtenidos las mediciones estimadas del vector sol y fuerza magnetica. Esto con el objetivo de obtener el residuo denotado con la variable y. Esta resta la medicion real 'z_real' de la modelada 'z_modelo'.

    Despues se obtiene la actualizacion, al obtener en primera instancia el delta_x mediante la multiplicacion de la ganancia de kalman con el residuo. En el caso de los cuaterniones, para actualizarlos se requiere utilizar la multiplicacion de cuaterniones entre delta_q y q_priori con la funcion 'quat_mult'. Para la velocidad angular simplemente se suman.

    Finalmente se actualiza la matriz del error covarianza de estados y se entrega como salida esta, en conjunto con la actualizacion de los estados.

### quaternion_to_euler(q):
    Al entregar un cuaternion, mediante ecuaciones de transformacion, se pasan a angulos de Euler, teniendo los cuidados necesarios en angulos criticos, como lo son el 90° y el 270°.

### torquer(u_PD_NL,lim):
    Limita la accion de control 'u_PD_NL' entregada con el limite de torque 'lim'.

### cuat_MSE_NL(q0,q1,q2,q3,w0,w1,w2,q0_nl,q1_nl,q2_nl,q3_nl,w0_nl,w1_nl,w2_nl):
    Calcula el mean square error entre dos vectores estados diferentes. Esto en base a los diferentes cuaterniones de la entrada.

### def RPY_MSE(t, q0_disc, q1_disc, q2_disc, q3_disc,q0_control, q1_control, q2_control, q3_control):
    Calcula el mean square error entre dos modelos. Esto en base a los diferentes cuaterniones de la entrada. El MSE obtenido esta en angulos de Euler y la funcion es capaz de otorgar los angulos de Euler.

### Optimizacion del K (solo para functions_03)
    eigenvalue_constraint(x, A, B): Esta representa la restriccion para la optimizacion de K, que se basa en que en modelos discretos, para que K sea estable, el valor absoluto de los valores propios de A + BK deben ser siempre menores a 1.

    objective_function(x): La funcion objetivo es la norma del valor inicial a implementar en la matriz K.

    opt_K(A_discrete,B_discrete,deltat,h,x0): Aplica la optimizacion, utilizando la funcion minimize del modulo scipy.

