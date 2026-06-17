import pandas as pd
from difflib import get_close_matches
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle

# Load recommendation data
with open('recommendation_data.pkl', 'rb') as file:
    data = pickle.load(file)
# Load SVD model
with open ('svd_model.pkl', 'rb') as file:
    svd = pickle.load(file)

movie_user_sparse = data['movie_user_sparse']
movie_index = data['movie_index']
movie_lookup = data['movie_lookup']
movie_to_idx = data['movie_to_idx']
idx_to_movie = data['idx_to_movie']

movie_factors = svd

def find_movie_id(movie_name, lookup_df):
    """Find the movieId for a given movie title, using exact and fuzzy matching. Returns a tuple of the recomendations df, 
    and the title of the movie in the csv. 
    Returns (None, None) if no match is found."""
    titles = lookup_df['title']

    exact_match = titles[titles.str.lower() == movie_name.lower()]
    if not exact_match.empty:
        return exact_match.index[0]

    close = get_close_matches(movie_name, titles.tolist(), n=1, cutoff=0.5)
    if close:
        matched_title = close[0]
        return titles[titles == matched_title].index[0]

    return None

def recommend_movies(movie_name, matrix, cf_model, lookup_df, model_type='svd', n_recs=10):
    """Return a dataframe of top-N similar movies for a given movie title."""
    
    movie_id = find_movie_id(movie_name, lookup_df)
    if movie_id is None:
        return None, None
    movie_idx = movie_to_idx[movie_id]

    title = movie_lookup.loc[movie_id, 'title']

    if model_type.lower() == 'svd':
        #print("Using SVD model for recommendations.")
        movie_vector = matrix[movie_idx].reshape(1, -1)  # Get the latent factors for the selected movie
        similarities = cosine_similarity(movie_vector, matrix).flatten()  # Compare against all movie vectors
        similar_indices = similarities.argsort()[::-1][1:n_recs+1]  # Exclude the movie itself
        
        recs = []
        for idx in similar_indices:
            rec_movie_id = idx_to_movie[idx]
            recs.append({
                'movieId': rec_movie_id,
                'Title': lookup_df.loc[rec_movie_id, 'title'],
                'Similarity': similarities[idx],
            })

        return pd.DataFrame(recs), title
    
if __name__ == "__main__":
    movie = input("Please enter a movie: ")

    (rec, title) = recommend_movies(movie, movie_user_sparse, svd, movie_lookup, 'svd', n_recs=5)
    print(rec)
    print("--------------------------")
    print(f"Movie name in csv: {title}")