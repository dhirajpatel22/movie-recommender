import pandas as pd
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


user_item_matrix = movie_data.pivot_table(index='userId', columns='movieId', values='rating').fillna(0) #use .pivot_table to handle duplicates of movieId and fill missing ratings with 0
#print(user_item_matrix.head())

#TODO: Compress user item matrix with scipy sparse matrix to save memory, then mabye apply SVD for dimesnionality reduction and make recommendations based on that

#KNN Approach
knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10, n_jobs=-1)
knn.fit(user_item_matrix)

