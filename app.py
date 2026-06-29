from flask import Flask, render_template, request
from recommend import get_hybrid_recommendations

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
# Load TF-IDF matrix    
with open('tfidf_matrix.pkl', 'rb') as f:
    content_matrix = pickle.load(f)


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
    content_data = data['content_data']
    movie_id_map = data['movie_id_map']

    movie_factors = svd

    content_movie_ids = content_data['tmdbId']
    content_lookup = content_data.set_index('tmdbId')




    movie = request.args.get('movie').strip()

    (recommendations, title) = get_hybrid_recommendations(movie, 
                                                            movie_user_sparse, 
                                                            svd, 
                                                            content_matrix, 
                                                            movie_lookup, 
                                                            content_lookup, 
                                                            n_recs=5, 
                                                            collab_weight=0.65, 
                                                            content_weight=0.35)
    if recommendations is None:
        return render_template('try-again.html')

    recommendations = recommendations.reset_index(drop=True)
    recommendations.index = recommendations.index + 1
    
    return render_template('movie.html', title = title, 
                            recommendations=recommendations.to_html(
                                classes='recommendations-table',
                                index=True,
                                index_names=False,
                                border=0,
                                justify='center'
                            ))

app.run(debug=True)