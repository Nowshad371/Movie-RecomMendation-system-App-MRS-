from math import sqrt
import numpy as np
from itertools import groupby
import pandas as pd
import requests
#IMPORTING CSV FILE
Movies_df = pd.read_csv(r'movies_metadata.csv')
inputMovies = pd.read_csv(r'inputMovies.csv')
movies_df = pd.read_csv(r'Movies_df.csv')

# Removing watched movies
cond = Movies_df['title'].isin(inputMovies['title'])
Movies_df.drop(Movies_df[cond].index, inplace=True)



def fetch_poster(movie_id, movie_track):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US".format(movie_id)
    data = requests.get(url)
    data = data.json()

    if not 'poster_path' in data or len(data['poster_path']) == 0:
        Most_Popular_Movies = most_popular_movie(Movies_df)
        Most_Popular_Movies['id'] = Most_Popular_Movies['id'].astype(int)
        movie_id = Most_Popular_Movies.iloc[5].id
        url = "https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US".format(155)
        data = requests.get(url)
        data = data.json()
        movie_track = movie_track + 1
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w342/" + poster_path
    return full_path,movie_track


#most_popular_movie
#movie
def most_popular_recommend():
   # index = movies[movies['title'] == movie].index[0]
    Most_Popular_Movies = most_popular_movie(Movies_df)
    Most_Popular_Movies['id'] = Most_Popular_Movies['id'].astype(int)
    recommended_movie_names = []
    recommended_movie_posters = []
    movie_track = 0
    movie_serial_no = 0
    for i in range(5):
        # fetch the movie poster
        movie_id = Most_Popular_Movies.iloc[i].id
        R_movie_id,movie_track = fetch_poster(movie_id,movie_track)
        recommended_movie_posters.append(R_movie_id)
        if(movie_track>movie_serial_no):
            recommended_movie_names.append(Most_Popular_Movies.iloc[i+1].title)
        else:
            recommended_movie_names.append(Most_Popular_Movies.iloc[i].title)
    return recommended_movie_posters,recommended_movie_names




#MOST POPULAR MOVIES

def most_popular_movie(ratings_df):

# Taking require attribute
      ratings_df = ratings_df[['id','title','vote_average','vote_count']]
      ratings_df = ratings_df.fillna(0)

#Creating new attribbute that need to find weighted average
      ratings_df['vote_average_mean'] = ratings_df['vote_average'].mean()
      ratings_df['vote_count_quantile'] = ratings_df['vote_count'].quantile(0.70)

# weighted average
      ratings_df['ratings'] = ratings_df.apply(lambda row: ((row.vote_average * row.vote_count) + (row.vote_average_mean * row.vote_count_quantile))/(row.vote_count + row.vote_count_quantile), axis=1)

# Important attributes
      ratings_df = ratings_df[['id','title','ratings']]

#Sort recommendations in descending order
      Most_Popular_Movies = ratings_df.sort_values(by = 'ratings',ascending=False)

#Taking top 10 movies
      Most_Popular_Movies = Most_Popular_Movies.head(15)

      return Most_Popular_Movies



#TRENDING MOVIES

def Trending_Movies(ratings_df):

#filling missing value with 0
  ratings_df = ratings_df.fillna(0)

#Taking the movie that are available
  ratings_df = ratings_df[ratings_df["status"] == 'Released']

#Sepating relase date into year,month,day
  ratings_df[["year", "month", "day"]] = ratings_df['release_date'].str.split("-", expand = True)

#dropping unnecessary columns
  ratings_df = ratings_df.drop(ratings_df.columns[[0,4,7,8]],axis = 1)


#Creating new attribbute that need to find weighted average
  ratings_df['vote_average_mean'] = ratings_df['vote_average'].mean()
  ratings_df['vote_count_quantile'] = ratings_df['vote_count'].quantile(0.70)

# weighted average
  ratings_df['ratings'] = ratings_df.apply(lambda row: ((row.vote_average * row.vote_count) + (row.vote_average_mean * row.vote_count_quantile))/(row.vote_count + row.vote_count_quantile), axis=1)

# Important attributes
  ratings_df = ratings_df[['id','title','ratings','year','status']]

#filling missing value with 0
  ratings_df = ratings_df.fillna(0)

#Converting string dtypes into integer
  ratings_df['year'] = ratings_df['year'].astype(int)

#Taking latest movies
  ratings_df = ratings_df.loc[ratings_df['year'] >= 2017]

#Sort recommendations in descending order
  Trending_Movies = ratings_df.sort_values(by = ['ratings'],ascending=False)

#Taking top 10 movies
  Trending_Movies = Trending_Movies.head(15)

  return Trending_Movies


#Calling TRENDING MOVIES
def Trending_recommend():
   # index = movies[movies['title'] == movie].index[0]
    Most_Popular_Movies = Trending_Movies(Movies_df)
    Most_Popular_Movies['id'] = Most_Popular_Movies['id'].astype(int)
    recommended_movie_names = []
    recommended_movie_posters = []
    movie_track = 0
    movie_serial_no = 0
    for i in range(5):
        # fetch the movie poster
        movie_id = Most_Popular_Movies.iloc[i].id
        R_movie_id,movie_track = fetch_poster(movie_id,movie_track)
        recommended_movie_posters.append(R_movie_id)
        if(movie_track>movie_serial_no):
            recommended_movie_names.append(Most_Popular_Movies.iloc[i+1].title)
        else:
            recommended_movie_names.append(Most_Popular_Movies.iloc[i].title)
    return recommended_movie_posters,recommended_movie_names


#Upcoming movies

def Upcoming_Movies(ratings_df):

  ratings_df = ratings_df[['id','title','release_date','status']]

  #filling missing value with 0
  ratings_df = ratings_df.fillna(0)

  #Taking the movie that are available
  ratings_df = ratings_df[ratings_df["status"] == 'Post Production']

  #Sepating relase date into year,month,day
  ratings_df[["year", "month", "day"]] = ratings_df['release_date'].str.split("-", expand = True)



# Important attributes
  ratings_df = ratings_df[['id','title','year','status']]

#filling missing value with 0
  ratings_df = ratings_df.fillna(0)

#Converting string dtypes into integer
  ratings_df['year'] = ratings_df['year'].astype(int)

#Taking latest movies
  ratings_df = ratings_df.loc[ratings_df['year'] >= 2017]


#Taking top 15 movies
  Upcoming_Movies = ratings_df.head(15)

  return Upcoming_Movies

#Calling Upcoming movies
def Upcoming_recommend():
   # index = movies[movies['title'] == movie].index[0]
    Most_Popular_Movies = Upcoming_Movies(Movies_df)
    Most_Popular_Movies['id'] = Most_Popular_Movies['id'].astype(int)
    recommended_movie_names = []
    recommended_movie_posters = []
    movie_track = 0
    movie_serial_no = 0
    for i in range(5):
        # fetch the movie poster
        movie_id = Most_Popular_Movies.iloc[i].id
        R_movie_id,movie_track = fetch_poster(movie_id,movie_track)
        recommended_movie_posters.append(R_movie_id)
        if(movie_track>movie_serial_no):
            recommended_movie_names.append(Most_Popular_Movies.iloc[i+1].title)
        else:
            recommended_movie_names.append(Most_Popular_Movies.iloc[i].title)
    return recommended_movie_posters,recommended_movie_names