import pandas as pd
# This line is used to make the dataset into tfidfVectorizer(we can provide priority to every word).This will help in predicting the similar kind of movies(same genres) . For example if you want the recommendations of action content movies then more preferences is given to action genre and find and throw the top 10 movies with that genre
from sklearn.feature_extraction.text import TfidfVectorizer
# This line we have imported linear_kernel. firstly the value merged and modified dataset is converted into tfidf vectorizer form which is matrix. Directly using this matrix will not provide more efficient results in finding the predictions ,so we convert tfidf matrix to consine_sim to get better efficient results
from sklearn.metrics.pairwise import linear_kernel

# Load the MovieLens dataset
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# Merge movies and ratings data
movie_ratings = pd.merge(movies, ratings, on='movieId')

# Calculate average rating for each movie
avg_ratings = movie_ratings.groupby('title')['rating'].mean().reset_index()

# Calculate number of ratings for each movie
num_ratings = movie_ratings.groupby('title')['rating'].count().reset_index()

# Merge average ratings and number of ratings
movie_stats = pd.merge(avg_ratings, num_ratings, on='title')
movie_stats = movie_stats.rename(columns={'rating_x': 'avg_rating', 'rating_y': 'num_ratings'})

# Create a TF-IDF vectorizer to process movie genres
# This line creates a object to the tdidfVectorizer that may not contain the stopwords of english for the column names(for example : is , the , "," etc)
tfidf = TfidfVectorizer(stop_words='english')
# This line fill the empty genre with NaN for the movies
movies['genres'] = movies['genres'].fillna('')
# This line creates a tfidf_matrix for the genres of movie dataset that contain the columns names as all the genre and each row is the movie name they are orderly arranged . The values in the matrix shows the relation between that movie and genre with a value (that can be 0 or 1 or floating point between 0 or 1). As the value increases the relation will be more. 
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Compute similarity scores between movies based on genres
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to get recommendations based on movie title
def get_recommendations(title, cosine_sim=cosine_sim):
    idx = movies.loc[movies['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Get top 10 similar movies
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices]

# Example: Get recommendations for a movie
recommendations = get_recommendations('Toy Story (1995)')
print(recommendations)
