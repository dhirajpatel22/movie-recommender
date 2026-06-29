import pandas as pd
from difflib import get_close_matches
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle

from sklearn.preprocessing import MinMaxScaler

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
content_data = data['content_data']
movie_id_map = data['movie_id_map']

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

# Collaborative Filtering Recommendation Function

def get_collaborative_recommendations(movie_name, matrix, cf_model, lookup_df, model_type='svd', n_recs=10):
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
                'Similarity': round(similarities[idx], 3)
            })
        
        df = pd.DataFrame(recs)
        return df, title
    

# Content-Based Filtering Approach

vectorizer = TfidfVectorizer(
    stop_words='english',
    min_df=2,
    max_df=0.7,
    ngram_range=(1, 2)
    )
content_matrix = vectorizer.fit_transform(content_data['content'].fillna(''))
content_movie_ids = content_data['tmdbId']
content_lookup = content_data.set_index('tmdbId')

def get_content_recommendations(movie_name, movie_features, lookup_df, n_recs=10):
    """Return a dataframe of top-N similar movies for a given movie title using content-based filtering."""
    
    movie_id = find_movie_id(movie_name, lookup_df)
    if movie_id is None:
        raise ValueError(f"Movie '{movie_name}' not found.")
    
    title = content_lookup.loc[movie_id, 'title']
    
    movie_positions = content_movie_ids[content_movie_ids == movie_id].index
    if len(movie_positions) == 0:
        raise ValueError(f"Movie '{movie_name}' not found in keyword features.")
    movie_idx = movie_positions[0]
    
    # Get the feature vector for the selected movie
    movie_features_vector = movie_features[movie_idx]
    
    # Calculate cosine similarity between the selected movie and all other movies
    similarities = cosine_similarity(movie_features_vector, movie_features).flatten()
    
    # Get the indices of the most similar movies
    similar_indices = similarities.argsort()[::-1][1:n_recs+1]  # Exclude the movie itself
    
    # Create a list of recommended movies
    recs = []
    for idx in similar_indices:
        rec_movie_id = int(content_movie_ids.iloc[idx])
        recs.append({
            'tmdbId': rec_movie_id,
            'Title': content_lookup.loc[rec_movie_id, 'title'],
            'Similarity': similarities[idx]
        })
    
    df = pd.DataFrame(recs)
    return df, title

# Hybrid Recommendation System

def get_hybrid_recommendations(movie_name, movie_user_sparse,model, keyword_matrix, movie_lookup, content_lookup, n_recs=10, collab_weight=0.5, content_weight=0.5):
    """Return a dataframe of top-N similar movies for a given movie title using both collaborative and content-based filtering."""

    # Detect collaborative model type
    if hasattr(model, 'kneighbors'):
        model_type = 'knn'
    elif hasattr(model, 'components_'):
        model_type = 'svd'
    else:
        raise ValueError("Unsupported collaborative model.")

    movie_id = find_movie_id(movie_name, movie_lookup)
    if movie_id is None:
        return None, None

    # Get recommendations from both systems
    (collab_recs, title) = get_collaborative_recommendations( movie_name, movie_user_sparse, model,  movie_lookup,  model_type, n_recs=500)
    (content_recs, _) = get_content_recommendations(  movie_name,  keyword_matrix, content_lookup, n_recs=500)

    collab_recs = collab_recs.merge(movie_id_map, on='movieId', how='left')

    # Rename columns for clarity
    collab_recs = collab_recs.rename(columns={'Similarity': 'Similarity_collab','Title': 'Title_collab'})
    content_recs = content_recs.rename(columns={'Similarity': 'Similarity_content','Title': 'Title_content' })

    # Keep ALL recommendations from either system
    hybrid = pd.merge(collab_recs, content_recs, on='tmdbId', how='outer')

    # Missing score = movie wasn't recommended by that model
    hybrid['Similarity_collab'] = hybrid['Similarity_collab'].fillna(0)
    hybrid['Similarity_content'] = hybrid['Similarity_content'].fillna(0)

    # Use whichever title exists
    hybrid['Title'] = hybrid['Title_collab'].fillna(hybrid['Title_content'])

    # Normalize scores to 0-1
    scaler = MinMaxScaler()
    hybrid['Similarity_collab'] = scaler.fit_transform( hybrid[['Similarity_collab']] )
    hybrid['Similarity_content'] = scaler.fit_transform(hybrid[['Similarity_content']])

    # Weighted hybrid score
    hybrid['Hybrid Score'] = (collab_weight * hybrid['Similarity_collab'] + content_weight * hybrid['Similarity_content'])

    # Remove duplicate movies
    hybrid = hybrid.drop_duplicates(subset=['tmdbId'])
    # Sort by hybrid score
    hybrid = hybrid.sort_values(by='Hybrid Score', ascending=False)

    df = hybrid[['tmdbId', 'Title', 'Hybrid Score']].head(n_recs)

    return df, title

if __name__ == "__main__":
    movie = input("Please enter a movie (content): ")

    (rec, title) = get_hybrid_recommendations(movie, movie_user_sparse, svd, content_matrix, movie_lookup, content_lookup, n_recs=10, collab_weight=0.5, content_weight=0.5)
    print(f"Recommendations for '{title}':")
    print(rec)
    print("\n Title in the dataset:", title)