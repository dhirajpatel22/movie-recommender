import pandas as pd
from difflib import get_close_matches
from sklearn.neighbors import NearestNeighbors

movie_metadata = pd.read_csv('Data/movies_metadata.csv', low_memory=False)
ratings = pd.read_csv('Data/ratings_small.csv') # use small ratings for now

#Collaborative Filtering Approach

movie_metadata['movieId'] = pd.to_numeric(movie_metadata['id'], errors='coerce') # Convert 'id' to numeric, make any errors to NaN
movie_metadata = movie_metadata.dropna(subset=['movieId']).copy() # Drop rows where 'movieId' is NaN (i.e., where conversion failed)
movie_metadata['movieId'] = movie_metadata['movieId'].astype(int) # Now 'movieId' is a clean integer column that can be merged with ratings (id column turned into movieId)

movie_data = pd.merge(movie_metadata, ratings, on='movieId', how = 'inner')
#print(movie_data[['movieId', 'title', 'rating']].head())

# Check for duplicates in movieId from movie_metadata
duplicates = movie_metadata[
    movie_metadata.duplicated(subset='movieId', keep=False)
]

#print(duplicates[['movieId', 'title']].sort_values('movieId'))


user_item_matrix = movie_data.pivot_table(  #use .pivot_table to handle duplicates of movieId 
    index='userId',
    columns='movieId',
    values='rating'
).fillna(0)                                 # fill missing ratings with 0

#print(user_item_matrix.head())

# For movie-to-movie recommendations, fit KNN on movie vectors.
movie_user_matrix = user_item_matrix.T

#print(movie_user_matrix.head())

movie_lookup = (
    movie_data[['movieId', 'title']]
    .dropna(subset=['title'])
    .drop_duplicates(subset=['movieId'])
    .set_index('movieId')
)

#TODO: Compress user item matrix with scipy sparse matrix to save memory, then mabye apply SVD for dimesnionality reduction and make recommendations based on that

#KNN Approach
knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10, n_jobs=-1)
knn.fit(movie_user_matrix)

def _find_movie_id(movie_name, lookup_df):
    titles = lookup_df['title']

    exact_match = titles[titles.str.lower() == movie_name.lower()]
    if not exact_match.empty:
        return exact_match.index[0]

    close = get_close_matches(movie_name, titles.tolist(), n=1, cutoff=0.5)
    if close:
        matched_title = close[0]
        return titles[titles == matched_title].index[0]

    return None

def recommend_movies(movie_name, matrix, cf_model, lookup_df, n_recs=10):
    """Return top-N similar movies for a given movie title."""
    movie_id = _find_movie_id(movie_name, lookup_df)
    if movie_id is None:
        raise ValueError(f"Movie '{movie_name}' not found.")

    # +1 because the closest neighbor is the movie itself.
    n_neighbors = min(n_recs + 1, len(matrix))
    distances, indices = cf_model.kneighbors(matrix.loc[[movie_id]], n_neighbors=n_neighbors)

    neighbor_ids = matrix.index[indices.flatten()].tolist()
    neighbor_distances = distances.flatten().tolist()

    recs = []
    for rec_movie_id, dist in zip(neighbor_ids, neighbor_distances):
        if rec_movie_id == movie_id:
            continue
        recs.append({
            'movieId': rec_movie_id,
            'Title': lookup_df.loc[rec_movie_id, 'title'],
            'Distance': dist,
        })
        if len(recs) >= n_recs:
            break

    return pd.DataFrame(recs)



sample_title = 'Batman'
recommendations = recommend_movies(sample_title, movie_user_matrix, knn, movie_lookup, n_recs=5)
print(recommendations)