# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 20:58:18 2024

@author: nachi
"""

import pyomo.environ as pyo

model = pyo.ConcreteModel()

model.x = pyo.Var(initialize=5.0, bounds=(1.001,None))
model.y = pyo.Var(initialize=5.0)

def obj_rule(m):
    return (m.x-1.01)**2 + m.y**2
model.obj = pyo.Objective(rule=obj_rule)

def con_rule(m):
    return m.y == pyo.sqrt(m.x - 1.0)
model.con = pyo.Constraint(rule=con_rule)

solver = pyo.SolverFactory('baron', executable='C:\\baron\\baron.exe')
# solver.options['NLPSubSolver'] = 'IPOPT'
# solver = pyo.SolverFactory('ipopt')
# solver = pyo.SolverFactory('baron', executable='C:\\baron\\baron.exe')
# solver.options['halt_on_ampl_error'] = 'yes'
solver.solve(model, tee=True)

print(pyo.value(model.x))
print(pyo.value(model.y))

#%%
import pyomo.environ as pyo

# Crear un modelo
model = pyo.ConcreteModel()

# Definir una variable
model.std_sensor_sol = pyo.Var(initialize=30.0)  # Inicializado en grados, por ejemplo

# Definir una expresión que calcula el seno
model.seno_std_sensor_sol = pyo.Expression(expr=pyo.sin(model.std_sensor_sol * 3.1415 / 180))

# Imprimir el valor del seno
print("Seno de std_sensor_sol:", pyo.value(model.seno_std_sensor_sol))
