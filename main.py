import json
import pandas as pd #cargar los daots
import numpy as np #operaciones matriciales
import nltk  as nltk#libreria de procesamiento de lenguaje natural
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import flask
from flask_cors import cross_origin
import joblib
app = flask.Flask(__name__)

@app.route('/recommender', methods=['GET'])
def testFlask():
    skills = flask.request.args.get('skills')



@app.route('/', methods=['GET'])
@cross_origin(allowed_methods=['GET'])
def recommenderAPI(request):
    try:
        skills = request.args.get('skills')
        if skills:
            # Set CORS headers for the preflight request
            if request.method == 'OPTIONS':
                # Allows GET requests from any origin with the Content-Type
                # header and caches preflight response for an 3600s
                headers = {
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'GET',
                    'Access-Control-Allow-Headers': 'Content-Type',
                    'Access-Control-Max-Age': '3600'
                }
                return ('', 204, headers)
            # Set CORS headers for the main request
            headers = {
                'Access-Control-Allow-Origin': '*'
            }
            return flask.make_response(recommender(skills.split(',')), 200, headers)
        # return flask.jsonify(recommender(skills.split(',')))
        else:
            return (f'Error: No skills provided. Please specify a skills.', 400)
    except Exception as e:
        return (f'Error: {e}', 400)



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
# app.run(host='0.0.0.0', debug=True)

