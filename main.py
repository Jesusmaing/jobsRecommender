import pandas as pd #cargar los daots
import numpy as np #operaciones matriciales
import matplotlib.pyplot as plt #graficar
import nltk  as nltk#libreria de procesamiento de lenguaje natural
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def recommenderSystemAPI(request):
    """Responds to any HTTP request.
    Args:
        request (flask.Request): HTTP request object.
    Returns:
        The response text or any set of values that can be turned into a
        Response object using
        `make_response <http://flask.pocoo.org/docs/1.0/api/#flask.Flask.make_response>`.
    """
    if request.args and 'skills' in request.args:
        request_json = request.get_json()
        vectorizer = TfidfVectorizer()
        jobs = pd.read_csv('jobsCleaned.csv',encoding='utf-8')
        tfidf_skills = vectorizer.fit_transform(jobs['skills'].apply(lambda x: ' '.join(x) if type(x) is list else x))
        #create a list of skills
        skills = request_json['skills']
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
    else:
        return f'Error: No skills provided. Please specify a skills.'

