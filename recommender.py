# %% [markdown]
# <a href="https://colab.research.google.com/github/Jesusmaing/jobsRecommender/blob/master/recommender.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# ##Importamos las librerias necesarias

# %%
import pandas as pd #cargar los daots
import numpy as np #operaciones matriciales
#import matplotlib.pyplot as plt #graficar
import nltk  as nltk#libreria de procesamiento de lenguaje natural
import requests
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# %% [markdown]
# ## Leemos el csv que está localizado en el repositorio de GitHub y lo cargamos en un pandas dataframe para tratarlo como una datatable

# %%

jobs = pd.read_csv('https://raw.githubusercontent.com/Jesusmaing/jobsRecommender/master/jobskills.csv',encoding='utf-8')
jobs

# %%
#Separamos en una lista de valores los skills
jobs['skills'] = jobs['skills'].str.split(',')
jobs['job'] = jobs['job'].str.split(',')

# %%
jobs

# %% [markdown]
# ### 1. Quitamos espacios del inicio y del final 
# ### 2. Quitamos todos aquellos trabajos que no tiene skills con dropna y si la lista está vacía se va a borrar
# ### 3. Al tener todos los datos normalizados, en minusculas y sin espacios, se procede a borrar todos los duplicados

# %%
#Limpiamos los datos, pasamos a minúsculas, quitamos espacios al inicio y final y quitamos duplicados if skill is list and not float
#DELETE EMPTY SKILLS
jobs['skills'] = jobs['skills'].apply(lambda x: [i.lower().strip() for i in x] if type(x) is list else x)
jobs.dropna()
#delete if job skills list and non float is empty or nan 
jobs = jobs[jobs['skills'].map(lambda d: len(d) > 0 if type(d) is list else False)]
#remove duplicates
jobs['skills'] = jobs['skills'].apply(lambda x: list(dict.fromkeys(x) ) if type(x) is list else x)


# %%
jobs

# %% [markdown]
# #Creamos una lista de frecuencias.
# ## Esto se hará para gráficar  el número de veces que se repiten ciertos skills para ver cuales son los más demandados en el dataset que seleccionamos.

# %%
vocabulary1 = nltk.FreqDist()
for skill in jobs['skills']:
    if type(skill) is list:
        # delete empty skills
        if len(skill) > 0:
            vocabulary1.update(skill)
    


# %% [markdown]
# Obtenemos el número de palabra más repetidas y lo graficamos

# %%
#plot all skills as histogram
# fig , ax = plt.subplots(figsize=(40,32))
# plt.barh([x[0] for x in vocabulary1.most_common(40)],[x[1] for x in vocabulary1.most_common(40)], label='skills')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.set_title('# Skills')
# plt.title('Skills más comunes')
# plt.show()


# %%
#Creamos una instancia para vectorizar los skills
vectorizer = TfidfVectorizer()
# Pasamos la lista de Strings a un solo string con espacios para transformar los datos
tfidf_skills = vectorizer.fit_transform(jobs['skills'].apply(lambda x: ' '.join(x) if type(x) is list else x))




# %%
def recommender(listOfSkills):
    #create a list of skills
    skills = listOfSkills
    #to lower case and remove spaces
    skills = [i.lower().strip() for i in skills]
    #remove duplicates
    skills = list(dict.fromkeys(skills))
    #all list of single string
    skills = ' '.join(skills)
    #vectorize the skills
    skills = vectorizer.transform([skills])
    
    #se calcula la similitud del coseno de la lista de skills dadas con el resto de las listas de skills
    #con eso se va a obtener un vector de similitud con cada uno de los trabajos de la lista de trabajos    
    similarity_list = cosine_similarity(skills, tfidf_skills)
    
    #sort the list of similarity in order desc and get the index
    #es una lista sorteada de distancias de menor a mayor, nosotros necesitamos la mayor similitud, por eso se hace sort descendentemente 
    #para obtener los indices de la mayor similitud
    sorted_indexes = np.argsort(similarity_list[0])[::-1]
    
    #get 10 recommendations jobs
    return jobs['job'].iloc[sorted_indexes].values[0:20]

# %%
def getRecommendations(skills):
    print('Skills: ',skills)
    recommendations = recommender(skills)
    print('recommendations: ',recommendations)
    #format tuple to list of strings
    recommendations = [list(x) for x in recommendations]
    # list of list to list of strings
    return recommendations



