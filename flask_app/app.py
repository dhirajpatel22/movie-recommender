from flask import Flask, render_template, request
from recommend import recommend_movies

import pandas as pd
from scipy.sparse import csr_matrix
import pickle

app = Flask(__name__)

# Load recommendation data
with open('recommendation_data.pkl', 'rb') as file:
    data = pickle.load(file)
# Load SVD model
with open ('svd_model.pkl', 'rb') as file:
    svd = pickle.load(file)

@app.route('/')
@app.route('/index')
def index():
        return render_template('index.html')

@app.route('/movie')
def get_movie():
        movie_user_sparse = data['movie_user_sparse']
        movie_index = data['movie_index']
        movie_lookup = data['movie_lookup']
        movie_to_idx = data['movie_to_idx']
        idx_to_movie = data['idx_to_movie']

        movie_factors = svd
            
        movie = request.args.get('movie')
        recommendations = recommend_movies(movie, movie_user_sparse, svd, movie_lookup, 'svd', n_recs=5)
        recommendations = recommendations.reset_index(drop=True)
        recommendations.index = recommendations.index + 1
        
        return render_template('movie.html', title = movie, 
                               recommendations=recommendations.to_html(
                                   classes='recommendations-table',
                                   index=True,
                                   index_names=False,
                                   border=0,
                                   justify='center'
                               ))

app.run(debug=True)