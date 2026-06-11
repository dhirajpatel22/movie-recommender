import pandas as pd
from difflib import get_close_matches
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

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

#KNN Approach
knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10, n_jobs=-1)
knn.fit(movie_user_sparse)

movie_index = movie_data['movieId'].astype('category').cat.categories

movie_to_idx = {movie_id: idx for idx, movie_id in enumerate(movie_index)}
idx_to_movie = {idx: movie_id for movie_id, idx in movie_to_idx.items()}

# SVD Approach
svd_components = min(100, max(1, min(movie_user_sparse.shape) - 1))
svd = TruncatedSVD(n_components=svd_components, random_state=42)
movie_factors = svd.fit_transform(movie_user_sparse)

movie_lookup = (
    movie_data[['movieId', 'title']]
    .dropna(subset=['title'])
    .drop_duplicates(subset=['movieId'])
    .set_index('movieId')
)

def find_movie_id(movie_name, lookup_df):
    """Find the movieId for a given movie title, using exact and fuzzy matching. Returns None if no match is found."""
    titles = lookup_df['title']

    exact_match = titles[titles.str.lower() == movie_name.lower()]
    if not exact_match.empty:
        return exact_match.index[0]

    close = get_close_matches(movie_name, titles.tolist(), n=1, cutoff=0.5)
    if close:
        matched_title = close[0]
        return titles[titles == matched_title].index[0]

    return None

def recommend_movies(movie_name, matrix, cf_model, lookup_df, model_type, n_recs=10):
    """Return a dataframe of top-N similar movies for a given movie title."""
    
    movie_id = find_movie_id(movie_name, lookup_df)
    if movie_id is None:
        raise ValueError(f"Movie '{movie_name}' not found.")
    movie_idx = movie_to_idx[movie_id]

    if model_type.lower() == 'knn':
        print("Using KNN model for recommendations.")

        n_neighbors = min(n_recs + 1, matrix.shape[0]) # +1 because the closest neighbor is the movie itself
        distances, indices = cf_model.kneighbors( # KNN search
            matrix[movie_idx], # matrix[movie_idx] is the vector of the selected movie
            n_neighbors=n_neighbors) 

        neighbor_movie_ids = [idx_to_movie[i] for i in indices.flatten()] #map neighbor indices back to movieIds
        neighbor_distances = distances.flatten().tolist()

        neighbor_similarity = [] #use cosine similarity instead of distance for better interpretability
        for dist in neighbor_distances:
            similarity = 1 - dist
            neighbor_similarity.append(similarity)

        recs = []
        for rec_movie_id, similarity in zip(neighbor_movie_ids, neighbor_similarity):
            if rec_movie_id == movie_id: #skips the movie itself in the recommendations
                continue
            recs.append({
                'movieId': rec_movie_id,
                'Title': lookup_df.loc[rec_movie_id, 'title'],
                'Similarity': similarity,
            })
            if len(recs) >= n_recs:
                break
        
        print(f"Recommendations for '{movie_name}' at movieId {movie_id}:")
        return pd.DataFrame(recs)
    elif model_type.lower() == 'svd':
        print("Using SVD model for recommendations.")
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
        print(f"Recommendations for '{movie_name}' at movieId {movie_id}:")
        return pd.DataFrame(recs)
  

#KNN Test: 
sample_title1 = 'Batman'
recommendations1 = recommend_movies(sample_title1, movie_user_sparse, knn, movie_lookup, 'knn', n_recs=5)
print(recommendations1)

sample_title2 = 'Toy Story'
recommendations2 = recommend_movies(sample_title2, movie_user_sparse, knn, movie_lookup, 'knn', n_recs=5)
print(recommendations2)

sample_title3 = 'It'
recommendations3 = recommend_movies(sample_title3, movie_user_sparse, knn, movie_lookup, 'knn', n_recs=5)
print(recommendations3)

#SVD Test:
recommendations_svd1 = recommend_movies(sample_title1, movie_factors, svd,  movie_lookup, 'svd', n_recs=5)
print(recommendations_svd1)

recommendations_svd2 = recommend_movies(sample_title2, movie_factors, svd, movie_lookup, 'svd', n_recs=5)
print(recommendations_svd2)

recommendations_svd3 = recommend_movies(sample_title3, movie_factors, svd, movie_lookup, 'svd', n_recs=5)
print(recommendations_svd3)