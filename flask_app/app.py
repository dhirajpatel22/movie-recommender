from flask import Flask, render_template, request
from recommend import recommend_movies

import pandas as pd
from scipy.sparse import csr_matrix
import pickle

app = Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
        return render_template('index.html')

@app.route('/movie')
def get_movie():
        movie_metadata = pd.read_csv('Data/movies_metadata.csv', low_memory=False)
        links = pd.read_csv('Data/links.csv')
        ratings = pd.read_csv('Data/ratings.csv')

        # Collaborative Filtering Approach
        # Correct merge: ratings (MovieLens movieId) -> links (TMDB tmdbId) -> movies_metadata (TMDB id)
        movie_metadata['id'] = pd.to_numeric(movie_metadata['id'], errors='coerce')
        movie_metadata = movie_metadata.dropna(subset=['id']).copy()
        movie_metadata['id'] = movie_metadata['id'].astype(int)

        links['tmdbId'] = pd.to_numeric(links['tmdbId'], errors='coerce')
        links = links.dropna(subset=['tmdbId']).copy()
        links['tmdbId'] = links['tmdbId'].astype(int)

        # Merge ratings with links on MovieLens movieId to get TMDB tmdbId
        movie_data = ratings.merge(links[['movieId', 'tmdbId']], on='movieId', how='inner')
        # Merge with metadata on TMDB id to get titles and metadata
        movie_data = movie_data.merge(movie_metadata[['id', 'title']], left_on='tmdbId', right_on='id', how='inner')
        # Keep only necessary columns for recommendations
        movie_data = movie_data[['userId', 'movieId', 'rating', 'title']]


        user_codes = movie_data['userId'].astype('category').cat.codes
        movie_codes = movie_data['movieId'].astype('category').cat.codes

        user_item_sparse = csr_matrix(
            (movie_data['rating'], (user_codes, movie_codes))
        )

        movie_user_sparse = user_item_sparse.T
        movie_index = movie_data['movieId'].astype('category').cat.categories

        movie_to_idx = {movie_id: idx for idx, movie_id in enumerate(movie_index)}
        idx_to_movie = {idx: movie_id for movie_id, idx in movie_to_idx.items()}

        # Load SVD
        with open ('svd_model.pkl', 'rb') as file:
                svd = pickle.load(file)

        movie_factors = svd.fit_transform(movie_user_sparse)

        movie_lookup = (
            movie_data[['movieId', 'title']]
            .dropna(subset=['title'])
            .drop_duplicates(subset=['movieId'])
            .set_index('movieId')
        )
            
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