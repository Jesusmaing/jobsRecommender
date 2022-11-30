from flask import Flask, jsonify
from flask_restful import Resource, Api
from flask_cors import CORS

import json
import pandas as pd #cargar los daots
import numpy as np #operaciones matriciales
import nltk  as nltk#libreria de procesamiento de lenguaje natural
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify, make_response
import joblib
app = Flask(__name__)
api = Api(app)
CORS(app)

def recommender(skills): 
    jobs = pd.read_csv('https://raw.githubusercontent.com/Jesusmaing/jobsRecommender/master/jobsCleaned.csv',encoding='utf-8')
    vectorizer = joblib.load("./vectorizer.pkl")
    tfidf_skills = joblib.load("./tfidf_skills.pkl")
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
    return json.dumps(jobs['job'].iloc[sorted_indexes].values[0:20].tolist())

class status (Resource):
    def get(self):
        try:
            return {'data': 'Api is Running, please type /recommender?skills=skill1,skill2,skill3'}
        except:
            return {'data': 'An Error Occurred during fetching Api'}


class Sum(Resource):
    def get(self, a, b):
        return jsonify({'data': a+b})

class Recommender(Resource):
    def get(self):
        skills = request.args.get('skills')
        return jsonify({'data': recommender(skills.split(','))})

api.add_resource(status, '/')
api.add_resource(Sum, '/add/<int:a>/<int:b>')
api.add_resource(Recommender, '/recommender')

if __name__ == '__main__':
    app.run()