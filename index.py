#importamos librerias
import pandas as pd
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
#leemos los datos
jobs = pd.read_csv('jobskills.csv',encoding='utf-8')

#imprimimos un pequeño resumen
#print(jobs.head())

#separamos en una lista de valores los skills
jobs['skills'] = jobs['skills'].str.split(',')
#delete if skills is 'see job description'
jobs = jobs[jobs.skills != 'See job description']
#delete empty skills
jobs = jobs[jobs.skills != '']


#Copiando el marco de datos de la pelicula en uno nuevo ya que no necesitamos la información del género por ahora.
skillsdf = jobs.copy()

#Para cada fila del marco de datos, iterar la lista de géneros y colocar un 1 en la columna que corresponda
#ONE HOT ENCODING
for index, row in jobs.iterrows():
    # if not empty
    if type(row['skills']) == list:
        for skill in row['skills']:
            # add column named genre and set value to 1
            skillsdf.at[index, skill] = 1

#Completar los valores NaN con 0 para mostrar que una película no tiene el género de la columna
skillsdf = skillsdf.fillna(0)
skillsdf.head()
print(skillsdf.head())
