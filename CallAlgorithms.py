import pandas as pd
from math import sqrt
import numpy as np
from sklearn.preprocessing import StandardScaler
from collections import Counter
from itertools import groupby
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import requests
import Alogithms

Movies_df = pd.read_csv(r'movies_metadata.csv')
inputMovies = pd.read_csv(r'inputMovies.csv')
movies_df = pd.read_csv(r'Movies_df.csv')

# Removing watched movies
cond = Movies_df['title'].isin(inputMovies['title'])
Movies_df.drop(Movies_df[cond].index, inplace=True)


def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US".format(movie_id)
    data = requests.get(url)
    data = data.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w342/" + poster_path
    return full_path

#CONTENT BASED WITH RATINGS
def CONTENT_BASED_recommend():
    inputMovies = pd.read_csv(r'inputMovies.csv')
    movies_df = pd.read_csv(r'Movies_df.csv')

    # Removing watched movies
    cond = Movies_df['title'].isin(inputMovies['title'])
    Movies_df.drop(Movies_df[cond].index, inplace=True)
   # index = movies[movies['title'] == movie].index[0]
    Most_Popular_Movies = Alogithms.UserInput(inputMovies,movies_df)
    Most_Popular_Movies['movieId'] = Most_Popular_Movies['movieId'].astype(int)
    recommended_movie_names = []
    recommended_movie_posters = []
    movie_track = 0
    movie_serial_no = 0
    for i in range(5):
        # fetch the movie poster
        movie_id = Most_Popular_Movies.iloc[i].movieId
        R_movie_id = fetch_poster(movie_id)
        recommended_movie_posters.append(R_movie_id)
        recommended_movie_names.append(Most_Popular_Movies.iloc[i].title)
    return recommended_movie_posters,recommended_movie_names


#CONTENT BASED WITHOUT RATINGS
def CONTENT_BASED_WITHOUT_RATINGS_recommend():
   # index = movies[movies['title'] == movie].index[0]
    movies_df = pd.read_csv(r'Movies_df.csv')
    Most_Popular_Movies= Alogithms.contentBasedWithoutRatings(movies_df, inputMovies)
    Most_Popular_Movies['movieId'] = Most_Popular_Movies['movieId'].astype(int)
    recommended_movie_names = []
    recommended_movie_posters = []
    movie_track = 0
    movie_serial_no = 0
    for i in range(5):
        # fetch the movie poster
        movie_id = Most_Popular_Movies.iloc[i].movieId
        R_movie_id = fetch_poster(movie_id)
        recommended_movie_posters.append(R_movie_id)
        recommended_movie_names.append(Most_Popular_Movies.iloc[i].title)
    return recommended_movie_posters,recommended_movie_names



#CLUSTERING

def Cluster_recommend():
    moviesMetaData = pd.read_csv(r'movies_ratings_data.csv')
    movies_df = pd.read_csv(r'Movies_df.csv')
    inputMovies = pd.read_csv(r'inputMovies.csv')
    Most_Popular_Movies= Alogithms.recomendation_cluster(movies_df, inputMovies,moviesMetaData)
    Most_Popular_Movies['movieId'] = Most_Popular_Movies['movieId'].astype(int)
    recommended_movie_names = []
    recommended_movie_posters = []
    movie_track = 0
    movie_serial_no = 0
    for i in range(5):
        # fetch the movie poster
        movie_id = Most_Popular_Movies.iloc[i].movieId
        R_movie_id = fetch_poster(movie_id)
        recommended_movie_posters.append(R_movie_id)
        recommended_movie_names.append(Most_Popular_Movies.iloc[i].title)
    return recommended_movie_posters,recommended_movie_names


#COLABORATIVE

def Collaborative_recommend():
    ratings_df = pd.read_csv(r'ratings.csv')
    movies_df = pd.read_csv(r'movies.csv')
    inputMovies = pd.read_csv(r'inputMovies.csv')
    HMovies_df = pd.read_csv(r'Movies_df.csv')
    CMovies_df = pd.read_csv(r'Movies_df.csv')
    recomendation_df = Alogithms.collaborative(ratings_df, movies_df, inputMovies,CMovies_df)
    Colrecomendation = recomendation_df['title'].head(10)
    Colrecomendationlist = [movies for movies in Colrecomendation]
    collaborativeFilter = HMovies_df[HMovies_df['original_title'].isin(Colrecomendationlist)]
    Most_Popular_Movies = collaborativeFilter[['id', 'original_title']]

    #Most_Popular_Movies['movieId'] = Most_Popular_Movies['id'].astype(int)
    recommended_movie_names = []
    recommended_movie_posters = []
    movie_track = 0
    movie_serial_no = 0
    for i in range(5):
        # fetch the movie poster
        movie_id = Most_Popular_Movies.iloc[i].id
        R_movie_id = fetch_poster(movie_id)
        recommended_movie_posters.append(R_movie_id)
        recommended_movie_names.append(Most_Popular_Movies.iloc[i].original_title)
    return recommended_movie_posters,recommended_movie_names

#HYBRID

def Hybrid_recommend():
    HMovies_df = pd.read_csv(r'Movies_df.csv')
    CMovies_df = pd.read_csv(r'Movies_df.csv')
    ratings_df = pd.read_csv(r'ratings.csv')
    movies_df = pd.read_csv(r'movies.csv')
    inputMovies = pd.read_csv(r'inputMovies.csv')
    SelectedMovies = pd.read_csv(r'Movies_df.csv')
    Hybrid_Model_recomednation = Alogithms.hybrid_Model(inputMovies, movies_df, ratings_df, HMovies_df,CMovies_df)

    Hybrid_movies = SelectedMovies[SelectedMovies['original_title'].isin(Hybrid_Model_recomednation)]
    Most_Popular_Movies = Hybrid_movies[['id', 'original_title']]

   # Most_Popular_Movies['movieId'] = Most_Popular_Movies['id'].astype(int)
    recommended_movie_names = []
    recommended_movie_posters = []
    movie_track = 0
    movie_serial_no = 0
    for i in range(5):
        # fetch the movie poster
        movie_id = Most_Popular_Movies.iloc[i].id
        R_movie_id = fetch_poster(movie_id)
        recommended_movie_posters.append(R_movie_id)
        recommended_movie_names.append(Most_Popular_Movies.iloc[i].original_title)
    return recommended_movie_posters,recommended_movie_names











