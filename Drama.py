from math import sqrt
import numpy as np
from itertools import groupby
import pandas as pd
import requests

#Drama

#IMPORTING CSV FILE
Movies_df = pd.read_csv(r'movies_metadata.csv')
inputMovies = pd.read_csv(r'inputMovies.csv')
movies_df = pd.read_csv(r'Movies_df.csv')

# Removing watched movies
cond = Movies_df['title'].isin(inputMovies['title'])
Movies_df.drop(Movies_df[cond].index, inplace=True)

def Drama(Best_Drama_df):

    Best_Drama_df =  Best_Drama_df[['id','title','genres','vote_average','vote_count','release_date','status']]

    #filling missing value with 0
    Best_Drama_df =  Best_Drama_df.fillna(0)

    #Taking the movie that are available
    Best_Drama_df =  Best_Drama_df[Best_Drama_df["status"] == 'Released']


   #Sepating relase date into year,month,day
    Best_Drama_df[["year", "month", "day"]] = Best_Drama_df['release_date'].str.split("-", expand = True)
    Best_Drama_df = Best_Drama_df[Best_Drama_df['genres'].str.contains('Drama')]

    #Creating new attribbute that need to find weighted average
    Best_Drama_df['vote_average_mean'] =  Best_Drama_df['vote_average'].mean()
    Best_Drama_df['vote_count_quantile'] =  Best_Drama_df['vote_count'].quantile(0.70)

    # weighted average
    Best_Drama_df['ratings'] = Best_Drama_df.apply(lambda row: ((row.vote_average * row.vote_count) + (row.vote_average_mean * row.vote_count_quantile))/(row.vote_count + row.vote_count_quantile), axis=1)


    # Important attributes
    Best_Drama_df = Best_Drama_df[['id','title','ratings','year','status']]

    #filling missing value with 0
    Best_Drama_df = Best_Drama_df.fillna(0)

    #Converting string dtypes into integer
    Best_Drama_df['year'] = Best_Drama_df['year'].astype(int)

    #Taking latest movies
    Best_Drama_df = Best_Drama_df.loc[Best_Drama_df['year'] >= 2010]

    #Sort recommendations in descending order
    Best_Drama_df = Best_Drama_df.sort_values(by = ['ratings'],ascending=False)

    #Taking top 10 movies
    Most_Popular_Drama = Best_Drama_df.head(15)

    return Most_Popular_Drama

#fetching_data
def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US".format(movie_id)
    data = requests.get(url)
    data = data.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w342/" + poster_path
    return full_path


#ANIMATION
def Drama_recommend():
   # index = movies[movies['title'] == movie].index[0]
    Most_Popular_Movies = Drama(Movies_df)
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