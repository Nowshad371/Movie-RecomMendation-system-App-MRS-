from math import sqrt
import numpy as np
from itertools import groupby
import pandas as pd
import requests
import GENRE_TYPE
#IMPORTING CSV FILE
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


#ANIMATION
def Animation_recommend():
   # index = movies[movies['title'] == movie].index[0]
    Most_Popular_Movies = GENRE_TYPE.Animation_Movies(Movies_df)
    Most_Popular_Movies['id'] = Most_Popular_Movies['id'].astype(int)
    recommended_movie_names = []
    recommended_movie_posters = []
    movie_track = 0
    movie_serial_no = 0
    for i in range(5):
        # fetch the movie poster
        movie_id = Most_Popular_Movies.iloc[i].id
        R_movie_id = fetch_poster(movie_id)
        recommended_movie_posters.append(R_movie_id)
        recommended_movie_names.append(Most_Popular_Movies.iloc[i].title)
    return recommended_movie_posters,recommended_movie_names



#Adventure_Movies
def Adventure_recommend():
   # index = movies[movies['title'] == movie].index[0]
    Most_Popular_Movies = GENRE_TYPE.Adventure_Movies(Movies_df)
    Most_Popular_Movies['id'] = Most_Popular_Movies['id'].astype(int)
    recommended_movie_names = []
    recommended_movie_posters = []
    movie_serial_no = 0
    for i in range(5):
        # fetch the movie poster
        movie_id = Most_Popular_Movies.iloc[i].id
        R_movie_id = fetch_poster(movie_id)
        recommended_movie_posters.append(R_movie_id)
        recommended_movie_names.append(Most_Popular_Movies.iloc[i].title)
    return recommended_movie_posters,recommended_movie_names



#Comedy_Movies
def Comedy_recommend():
   # index = movies[movies['title'] == movie].index[0]
    Most_Popular_Movies = GENRE_TYPE.Comedy_Movies(Movies_df)
    Most_Popular_Movies['id'] = Most_Popular_Movies['id'].astype(int)
    recommended_movie_names = []
    recommended_movie_posters = []
    movie_serial_no = 0
    for i in range(5):
        # fetch the movie poster
        movie_id = Most_Popular_Movies.iloc[i].id
        R_movie_id = fetch_poster(movie_id)
        recommended_movie_posters.append(R_movie_id)
        recommended_movie_names.append(Most_Popular_Movies.iloc[i].title)
    return recommended_movie_posters,recommended_movie_names



#Fantasy_Movies
def Fantasy_recommend():
   # index = movies[movies['title'] == movie].index[0]
    Most_Popular_Movies = GENRE_TYPE.Fantasy_Movies(Movies_df)
    Most_Popular_Movies['id'] = Most_Popular_Movies['id'].astype(int)
    recommended_movie_names = []
    recommended_movie_posters = []
    movie_serial_no = 0
    for i in range(5):
        # fetch the movie poster
        movie_id = Most_Popular_Movies.iloc[i].id
        R_movie_id = fetch_poster(movie_id)
        recommended_movie_posters.append(R_movie_id)
        recommended_movie_names.append(Most_Popular_Movies.iloc[i].title)
    return recommended_movie_posters,recommended_movie_names


#Action_Movies
def Action_recommend():
   # index = movies[movies['title'] == movie].index[0]
    Most_Popular_Movies = GENRE_TYPE.Action_Movies(Movies_df)
    Most_Popular_Movies['id'] = Most_Popular_Movies['id'].astype(int)
    recommended_movie_names = []
    recommended_movie_posters = []
    movie_serial_no = 0
    for i in range(5):
        # fetch the movie poster
        movie_id = Most_Popular_Movies.iloc[i].id
        R_movie_id = fetch_poster(movie_id)
        recommended_movie_posters.append(R_movie_id)
        recommended_movie_names.append(Most_Popular_Movies.iloc[i].title)
    return recommended_movie_posters,recommended_movie_names


#Romance_Movies
def Romance_recommend():
   # index = movies[movies['title'] == movie].index[0]
    Most_Popular_Movies = GENRE_TYPE.Romance_Movies(Movies_df)
    Most_Popular_Movies['id'] = Most_Popular_Movies['id'].astype(int)
    recommended_movie_names = []
    recommended_movie_posters = []
    movie_serial_no = 0
    for i in range(5):
        # fetch the movie poster
        movie_id = Most_Popular_Movies.iloc[i].id
        R_movie_id = fetch_poster(movie_id)
        recommended_movie_posters.append(R_movie_id)
        recommended_movie_names.append(Most_Popular_Movies.iloc[i].title)
    return recommended_movie_posters,recommended_movie_names


#Horror_Movies
def Horror_recommend():
   # index = movies[movies['title'] == movie].index[0]
    Most_Popular_Movies = GENRE_TYPE.Horror_Movies(Movies_df)
    Most_Popular_Movies['id'] = Most_Popular_Movies['id'].astype(int)
    recommended_movie_names = []
    recommended_movie_posters = []
    movie_serial_no = 0
    for i in range(5):
        # fetch the movie poster
        movie_id = Most_Popular_Movies.iloc[i].id
        R_movie_id = fetch_poster(movie_id)
        recommended_movie_posters.append(R_movie_id)
        recommended_movie_names.append(Most_Popular_Movies.iloc[i].title)
    return recommended_movie_posters,recommended_movie_names



#Crime_Movies
def Crime_recommend():
   # index = movies[movies['title'] == movie].index[0]
    Most_Popular_Movies = GENRE_TYPE.Crime_Movies(Movies_df)
    Most_Popular_Movies['id'] = Most_Popular_Movies['id'].astype(int)
    recommended_movie_names = []
    recommended_movie_posters = []
    movie_serial_no = 0
    for i in range(5):
        # fetch the movie poster
        movie_id = Most_Popular_Movies.iloc[i].id
        R_movie_id = fetch_poster(movie_id)
        recommended_movie_posters.append(R_movie_id)
        recommended_movie_names.append(Most_Popular_Movies.iloc[i].title)
    return recommended_movie_posters,recommended_movie_names


#Thriller_Movies
def Thriller_recommend():
   # index = movies[movies['title'] == movie].index[0]
    Most_Popular_Movies = GENRE_TYPE.Thriller_Movies(Movies_df)
    Most_Popular_Movies['id'] = Most_Popular_Movies['id'].astype(int)
    recommended_movie_names = []
    recommended_movie_posters = []
    movie_serial_no = 0
    for i in range(5):
        # fetch the movie poster
        movie_id = Most_Popular_Movies.iloc[i].id
        R_movie_id = fetch_poster(movie_id)
        recommended_movie_posters.append(R_movie_id)
        recommended_movie_names.append(Most_Popular_Movies.iloc[i].title)
    return recommended_movie_posters,recommended_movie_names


#Mystery_Movies
def Mystery_recommend():
   # index = movies[movies['title'] == movie].index[0]
    Most_Popular_Movies = GENRE_TYPE.Mystery_Movies(Movies_df)
    Most_Popular_Movies['id'] = Most_Popular_Movies['id'].astype(int)
    recommended_movie_names = []
    recommended_movie_posters = []
    movie_serial_no = 0
    for i in range(5):
        # fetch the movie poster
        movie_id = Most_Popular_Movies.iloc[i].id
        R_movie_id = fetch_poster(movie_id)
        recommended_movie_posters.append(R_movie_id)
        recommended_movie_names.append(Most_Popular_Movies.iloc[i].title)
    return recommended_movie_posters,recommended_movie_names


#Family_Movies
def Family_recommend():
   # index = movies[movies['title'] == movie].index[0]
    Most_Popular_Movies = GENRE_TYPE.Family_Movies(Movies_df)
    Most_Popular_Movies['id'] = Most_Popular_Movies['id'].astype(int)
    recommended_movie_names = []
    recommended_movie_posters = []
    movie_serial_no = 0
    for i in range(5):
        # fetch the movie poster
        movie_id = Most_Popular_Movies.iloc[i].id
        R_movie_id = fetch_poster(movie_id)
        recommended_movie_posters.append(R_movie_id)
        recommended_movie_names.append(Most_Popular_Movies.iloc[i].title)
    return recommended_movie_posters,recommended_movie_names

#History_Movies

def fetch_poster_History_foriegn(movie_id, movie_track):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US".format(movie_id)
    data = requests.get(url)
    data = data.json()

    if not 'poster_path' in data or len(data['poster_path']) == 0:
        Most_Popular_Movies = GENRE_TYPE.History_Movies(Movies_df)
        Most_Popular_Movies['id'] = Most_Popular_Movies['id'].astype(int)
        movie_id = Most_Popular_Movies.iloc[5].id
        url = "https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US".format(155)
        data = requests.get(url)
        data = data.json()
        movie_track = movie_track + 1
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w342/" + poster_path
    return full_path,movie_track

def History_recommend():
   # index = movies[movies['title'] == movie].index[0]
    Most_Popular_Movies = GENRE_TYPE.History_Movies(Movies_df)
    Most_Popular_Movies['id'] = Most_Popular_Movies['id'].astype(int)
    recommended_movie_names = []
    recommended_movie_posters = []
    movie_track = 0
    movie_serial_no = 0
    for i in range(5):
        # fetch the movie poster
        movie_id = Most_Popular_Movies.iloc[i].id
        R_movie_id, movie_track = fetch_poster_History_foriegn(movie_id, movie_track)
        recommended_movie_posters.append(R_movie_id)
        if (movie_track > movie_serial_no):
            recommended_movie_names.append(Most_Popular_Movies.iloc[i + 1].title)
            movie_track = 0
        else:
            recommended_movie_names.append(Most_Popular_Movies.iloc[i].title)
        return recommended_movie_posters,recommended_movie_names


#Foreign_Movies


def Foreign_recommend():
    # index = movies[movies['title'] == movie].index[0]
    Most_Popular_Movies = GENRE_TYPE.Foreign_Movies(Movies_df)
    Most_Popular_Movies['id'] = Most_Popular_Movies['id'].astype(int)
    recommended_movie_names = []
    recommended_movie_posters = []
    movie_track = 0
    movie_serial_no = 0
    for i in range(5):
        # fetch the movie poster
        movie_id = Most_Popular_Movies.iloc[i].id
        R_movie_id, movie_track = fetch_poster_History_foriegn(movie_id, movie_track)
        recommended_movie_posters.append(R_movie_id)
        if (movie_track > movie_serial_no):
            recommended_movie_names.append(Most_Popular_Movies.iloc[i + 1].title)
            movie_track = 0
        else:
            recommended_movie_names.append(Most_Popular_Movies.iloc[i].title)
        return recommended_movie_posters, recommended_movie_names