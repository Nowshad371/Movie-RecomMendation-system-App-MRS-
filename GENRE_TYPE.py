from math import sqrt
import numpy as np
from itertools import groupby
import pandas as pd
import requests


def Animation_Movies(Best_Animation_df):

    Best_Animation_df = Best_Animation_df[['id','title','genres','vote_average','vote_count','release_date','status']]

    #filling missing value with 0
    Best_Animation_df = Best_Animation_df.fillna(0)

    #Taking the movie that are available
    Best_Animation_df = Best_Animation_df[Best_Animation_df["status"] == 'Released']

    #Sepating relase date into year,month,day
    Best_Animation_df[["year", "month", "day"]] = Best_Animation_df['release_date'].str.split("-", expand = True)
    Best_Animation_df = Best_Animation_df[Best_Animation_df['genres'].str.contains('Animation')]

    #Creating new attribbute that need to find weighted average
    Best_Animation_df['vote_average_mean'] = Best_Animation_df['vote_average'].mean()
    Best_Animation_df['vote_count_quantile'] = Best_Animation_df['vote_count'].quantile(0.70)

    # weighted average
    Best_Animation_df['ratings'] = Best_Animation_df.apply(lambda row: ((row.vote_average * row.vote_count) + (row.vote_average_mean * row.vote_count_quantile))/(row.vote_count + row.vote_count_quantile), axis=1)


    # Important attributes
    Best_Animation_df = Best_Animation_df[['id','title','ratings','year','status']]

    #filling missing value with 0
    Best_Animation_df = Best_Animation_df.fillna(0)

    #Converting string dtypes into integer
    Best_Animation_df['year'] = Best_Animation_df['year'].astype(int)

    #Taking latest movies
    Best_Animation_df = Best_Animation_df.loc[Best_Animation_df['year'] >= 2017]

    #Sort recommendations in descending order
    Most_Popular_Movies = Best_Animation_df.sort_values(by = ['ratings'],ascending=False)

    #Taking top 10 movies
    Most_Popular_Animation_Movies = Most_Popular_Movies.head(15)

    return Most_Popular_Animation_Movies

#Adventure movies

def Adventure_Movies(Best_Adventure_df):

    Best_Adventure_df = Best_Adventure_df[['id','title','genres','vote_average','vote_count','release_date','status']]

    #filling missing value with 0
    Best_Adventure_df = Best_Adventure_df.fillna(0)

    #Taking the movie that are available
    Best_Adventure_df = Best_Adventure_df[Best_Adventure_df["status"] == 'Released']

    #Sepating relase date into year,month,day
    Best_Adventure_df[["year", "month", "day"]] = Best_Adventure_df['release_date'].str.split("-", expand = True)
    Best_Adventure_df = Best_Adventure_df[Best_Adventure_df['genres'].str.contains('Adventure')]

    #Creating new attribbute that need to find weighted average
    Best_Adventure_df['vote_average_mean'] = Best_Adventure_df['vote_average'].mean()
    Best_Adventure_df['vote_count_quantile'] = Best_Adventure_df['vote_count'].quantile(0.70)

    # weighted average
    Best_Adventure_df['ratings'] = Best_Adventure_df.apply(lambda row: ((row.vote_average * row.vote_count) + (row.vote_average_mean * row.vote_count_quantile))/(row.vote_count + row.vote_count_quantile), axis=1)


    # Important attributes
    Best_Adventure_df = Best_Adventure_df[['id','title','ratings','year','status']]

    #filling missing value with 0
    Best_Adventure_df = Best_Adventure_df.fillna(0)

    #Converting string dtypes into integer
    Best_Adventure_df['year'] = Best_Adventure_df['year'].astype(int)

    #Taking latest movies
    Best_Adventure_df = Best_Adventure_df.loc[Best_Adventure_df['year'] >= 2017]

    #Sort recommendations in descending order
    Most_Popular_Movies = Best_Adventure_df.sort_values(by = ['ratings'],ascending=False)

    #Taking top 10 movies
    Most_Popular_Adventure_Movies = Most_Popular_Movies.head(15)

    return Most_Popular_Adventure_Movies

#Comedy movies

def Comedy_Movies(Best_Comedy_df):

    Best_Comedy_df = Best_Comedy_df[['id','title','genres','vote_average','vote_count','release_date','status']]

    #filling missing value with 0
    Best_Comedy_df= Best_Comedy_df.fillna(0)

    #Taking the movie that are available
    Best_Comedy_df = Best_Comedy_df[Best_Comedy_df["status"] == 'Released']

    #Sepating relase date into year,month,day
    Best_Comedy_df[["year", "month", "day"]] = Best_Comedy_df['release_date'].str.split("-", expand = True)
    Best_Comedy_df = Best_Comedy_df[Best_Comedy_df['genres'].str.contains('Comedy')]

    #Creating new attribbute that need to find weighted average
    Best_Comedy_df['vote_average_mean'] = Best_Comedy_df['vote_average'].mean()
    Best_Comedy_df['vote_count_quantile'] = Best_Comedy_df['vote_count'].quantile(0.70)

    # weighted average
    Best_Comedy_df['ratings'] = Best_Comedy_df.apply(lambda row: ((row.vote_average * row.vote_count) + (row.vote_average_mean * row.vote_count_quantile))/(row.vote_count + row.vote_count_quantile), axis=1)


    # Important attributes
    Best_Comedy_df = Best_Comedy_df[['id','title','ratings','year','status']]

    #filling missing value with 0
    Best_Comedy_df = Best_Comedy_df.fillna(0)

    #Converting string dtypes into integer
    Best_Comedy_df['year'] = Best_Comedy_df['year'].astype(int)

    #Taking latest movies
    Best_Comedy_df = Best_Comedy_df.loc[Best_Comedy_df['year'] >= 2017]

    #Sort recommendations in descending order
    Best_Comedy_df = Best_Comedy_df.sort_values(by = ['ratings'],ascending=False)

    #Taking top 10 movies
    Most_Popular_Comedy_Movies = Best_Comedy_df.head(15)

    return Most_Popular_Comedy_Movies

#Fantasy movies

def Fantasy_Movies(Best_Fantasy_df):

    Best_Fantasy_df = Best_Fantasy_df[['id','title','genres','vote_average','vote_count','release_date','status']]

    #filling missing value with 0
    Best_Fantasy_df= Best_Fantasy_df.fillna(0)

    #Taking the movie that are available
    Best_Fantasy_df = Best_Fantasy_df[Best_Fantasy_df["status"] == 'Released']

    #Sepating relase date into year,month,day
    Best_Fantasy_df[["year", "month", "day"]] = Best_Fantasy_df['release_date'].str.split("-", expand = True)
    Best_Fantasy_df = Best_Fantasy_df[Best_Fantasy_df['genres'].str.contains('Fantasy')]

    #Creating new attribbute that need to find weighted average
    Best_Fantasy_df['vote_average_mean'] = Best_Fantasy_df['vote_average'].mean()
    Best_Fantasy_df['vote_count_quantile'] = Best_Fantasy_df['vote_count'].quantile(0.70)

    # weighted average
    Best_Fantasy_df['ratings'] = Best_Fantasy_df.apply(lambda row: ((row.vote_average * row.vote_count) + (row.vote_average_mean * row.vote_count_quantile))/(row.vote_count + row.vote_count_quantile), axis=1)


    # Important attributes
    Best_Fantasy_df = Best_Fantasy_df[['id','title','ratings','year','status']]

    #filling missing value with 0
    Best_Fantasy_df = Best_Fantasy_df.fillna(0)

    #Converting string dtypes into integer
    Best_Fantasy_df['year'] = Best_Fantasy_df['year'].astype(int)

    #Taking latest movies
    Best_Fantasy_df = Best_Fantasy_df.loc[Best_Fantasy_df['year'] >= 2017]

    #Sort recommendations in descending order
    Best_Fantasy_df = Best_Fantasy_df.sort_values(by = ['ratings'],ascending=False)

    #Taking top 10 movies
    Most_Popular_Fantasy_Movies = Best_Fantasy_df.head(15)

    return Most_Popular_Fantasy_Movies

#Action movies

def Action_Movies(Best_Action_df):

    Best_Action_df = Best_Action_df[['id','title','genres','vote_average','vote_count','release_date','status']]

    #filling missing value with 0
    Best_Action_df= Best_Action_df.fillna(0)

    #Taking the movie that are available
    Best_Action_df = Best_Action_df[Best_Action_df["status"] == 'Released']

    #Sepating relase date into year,month,day
    Best_Action_df[["year", "month", "day"]] = Best_Action_df['release_date'].str.split("-", expand = True)
    Best_Action_df = Best_Action_df[Best_Action_df['genres'].str.contains('Action')]

    #Creating new attribbute that need to find weighted average
    Best_Action_df['vote_average_mean'] = Best_Action_df['vote_average'].mean()
    Best_Action_df['vote_count_quantile'] = Best_Action_df['vote_count'].quantile(0.70)

    # weighted average
    Best_Action_df['ratings'] = Best_Action_df.apply(lambda row: ((row.vote_average * row.vote_count) + (row.vote_average_mean * row.vote_count_quantile))/(row.vote_count + row.vote_count_quantile), axis=1)


    # Important attributes
    Best_Action_df = Best_Action_df[['id','title','ratings','year','status']]

    #filling missing value with 0
    Best_Action_df = Best_Action_df.fillna(0)

    #Converting string dtypes into integer
    Best_Action_df['year'] = Best_Action_df['year'].astype(int)

    #Taking latest movies
    Best_Action_df = Best_Action_df.loc[Best_Action_df['year'] >= 2017]

    #Sort recommendations in descending order
    Best_Action_df = Best_Action_df.sort_values(by = ['ratings'],ascending=False)

    #Taking top 10 movies
    Most_Popular_Action_Movies = Best_Action_df.head(15)

    return Most_Popular_Action_Movies

#Romance movies

def Romance_Movies(Best_Romance_df):

    Best_Romance_df = Best_Romance_df[['id','title','genres','vote_average','vote_count','release_date','status']]

    #filling missing value with 0
    Best_Romance_df = Best_Romance_df.fillna(0)

    #Taking the movie that are available
    Best_Romance_df = Best_Romance_df[Best_Romance_df["status"] == 'Released']

    #Sepating relase date into year,month,day
    Best_Romance_df[["year", "month", "day"]] = Best_Romance_df['release_date'].str.split("-", expand = True)
    Best_Romance_df = Best_Romance_df[Best_Romance_df['genres'].str.contains('Romance')]

    #Creating new attribbute that need to find weighted average
    Best_Romance_df['vote_average_mean'] = Best_Romance_df['vote_average'].mean()
    Best_Romance_df['vote_count_quantile'] = Best_Romance_df['vote_count'].quantile(0.70)

    # weighted average
    Best_Romance_df['ratings'] = Best_Romance_df.apply(lambda row: ((row.vote_average * row.vote_count) + (row.vote_average_mean * row.vote_count_quantile))/(row.vote_count + row.vote_count_quantile), axis=1)


    # Important attributes
    Best_Romance_df = Best_Romance_df[['id','title','ratings','year','status']]

    #filling missing value with 0
    Best_Romance_df = Best_Romance_df.fillna(0)

    #Converting string dtypes into integer
    Best_Romance_df['year'] = Best_Romance_df['year'].astype(int)

    #Taking latest movies
    Best_Romance_df = Best_Romance_df.loc[Best_Romance_df['year'] >= 2017]

    #Sort recommendations in descending order
    Best_Romance_df = Best_Romance_df.sort_values(by = ['ratings'],ascending=False)

    #Taking top 10 movies
    Most_Popular_Romance_Movies = Best_Romance_df.head(15)

    return Most_Popular_Romance_Movies

#Horror movies

def Horror_Movies(Best_Horror_df):

    Best_Horror_df = Best_Horror_df[['id','title','genres','vote_average','vote_count','release_date','status']]

    #filling missing value with 0
    Best_Horror_df = Best_Horror_df.fillna(0)

    #Taking the movie that are available
    Best_Horror_df = Best_Horror_df[Best_Horror_df["status"] == 'Released']

    #Sepating relase date into year,month,day
    Best_Horror_df[["year", "month", "day"]] = Best_Horror_df['release_date'].str.split("-", expand = True)
    Best_Horror_df = Best_Horror_df[Best_Horror_df['genres'].str.contains('Horror')]

    #Creating new attribbute that need to find weighted average
    Best_Horror_df['vote_average_mean'] = Best_Horror_df['vote_average'].mean()
    Best_Horror_df['vote_count_quantile'] = Best_Horror_df['vote_count'].quantile(0.70)

    # weighted average
    Best_Horror_df['ratings'] = Best_Horror_df.apply(lambda row: ((row.vote_average * row.vote_count) + (row.vote_average_mean * row.vote_count_quantile))/(row.vote_count + row.vote_count_quantile), axis=1)


    # Important attributes
    Best_Horror_df = Best_Horror_df[['id','title','ratings','year','status']]

    #filling missing value with 0
    Best_Horror_df = Best_Horror_df.fillna(0)

    #Converting string dtypes into integer
    Best_Horror_df['year'] = Best_Horror_df['year'].astype(int)

    #Taking latest movies
    Best_Horror_df = Best_Horror_df.loc[Best_Horror_df['year'] >= 2017]

    #Sort recommendations in descending order
    Best_Horror_df= Best_Horror_df.sort_values(by = ['ratings'],ascending=False)

    #Taking top 10 movies
    Most_Popular_Horror_Movies = Best_Horror_df.head(15)

    return Most_Popular_Horror_Movies

#Crime movies

def Crime_Movies(Best_Crime_df):

    Best_Crime_df = Best_Crime_df[['id','title','genres','vote_average','vote_count','release_date','status']]

    #filling missing value with 0
    Best_Crime_df = Best_Crime_df.fillna(0)

    #Taking the movie that are available
    Best_Crime_df = Best_Crime_df[Best_Crime_df["status"] == 'Released']

    #Sepating relase date into year,month,day
    Best_Crime_df[["year", "month", "day"]] = Best_Crime_df['release_date'].str.split("-", expand = True)
    Best_Crime_df = Best_Crime_df[Best_Crime_df['genres'].str.contains('Crime')]

    #Creating new attribbute that need to find weighted average
    Best_Crime_df['vote_average_mean'] = Best_Crime_df['vote_average'].mean()
    Best_Crime_df['vote_count_quantile'] = Best_Crime_df['vote_count'].quantile(0.70)

    # weighted average
    Best_Crime_df['ratings'] = Best_Crime_df.apply(lambda row: ((row.vote_average * row.vote_count) + (row.vote_average_mean * row.vote_count_quantile))/(row.vote_count + row.vote_count_quantile), axis=1)


    # Important attributes
    Best_Crime_df= Best_Crime_df[['id','title','ratings','year','status']]

    #filling missing value with 0
    Best_Crime_df = Best_Crime_df.fillna(0)

    #Converting string dtypes into integer
    Best_Crime_df['year'] = Best_Crime_df['year'].astype(int)

    #Taking latest movies
    Best_Crime_df = Best_Crime_df.loc[Best_Crime_df['year'] >= 2017]

    #Sort recommendations in descending order
    Best_Crime_df = Best_Crime_df.sort_values(by = ['ratings'],ascending=False)

    #Taking top 10 movies
    Most_Popular_Crime_Movies = Best_Crime_df.head(15)

    return Most_Popular_Crime_Movies

#Thriller movies

def Thriller_Movies(Best_Thriller_df):

    Best_Thriller_df = Best_Thriller_df[['id','title','genres','vote_average','vote_count','release_date','status']]

    #filling missing value with 0
    Best_Thriller_df = Best_Thriller_df.fillna(0)

    #Taking the movie that are available
    Best_Thriller_df = Best_Thriller_df[Best_Thriller_df["status"] == 'Released']

    #Sepating relase date into year,month,day
    Best_Thriller_df[["year", "month", "day"]] = Best_Thriller_df['release_date'].str.split("-", expand = True)
    Best_Thriller_df = Best_Thriller_df[Best_Thriller_df['genres'].str.contains('Thriller')]

    #Creating new attribbute that need to find weighted average
    Best_Thriller_df['vote_average_mean'] = Best_Thriller_df['vote_average'].mean()
    Best_Thriller_df['vote_count_quantile'] = Best_Thriller_df['vote_count'].quantile(0.70)

    # weighted average
    Best_Thriller_df['ratings'] = Best_Thriller_df.apply(lambda row: ((row.vote_average * row.vote_count) + (row.vote_average_mean * row.vote_count_quantile))/(row.vote_count + row.vote_count_quantile), axis=1)


    # Important attributes
    Best_Thriller_df = Best_Thriller_df[['id','title','ratings','year','status']]

    #filling missing value with 0
    Best_Thriller_df = Best_Thriller_df.fillna(0)

    #Converting string dtypes into integer
    Best_Thriller_df['year'] = Best_Thriller_df['year'].astype(int)

    #Taking latest movies
    Best_Thriller_df = Best_Thriller_df.loc[Best_Thriller_df['year'] >= 2017]

    #Sort recommendations in descending order
    Best_Thriller_df = Best_Thriller_df.sort_values(by = ['ratings'],ascending=False)

    #Taking top 10 movies
    Most_Popular_Thriller_Movies = Best_Thriller_df.head(15)

    return Most_Popular_Thriller_Movies

#Mystery movies

def Mystery_Movies(Best_Mystery_df):

    Best_Mystery_df = Best_Mystery_df[['id','title','genres','vote_average','vote_count','release_date','status']]

    #filling missing value with 0
    Best_Mystery_df = Best_Mystery_df.fillna(0)

    #Taking the movie that are available
    Best_Mystery_df = Best_Mystery_df[Best_Mystery_df["status"] == 'Released']

    #Sepating relase date into year,month,day
    Best_Mystery_df[["year", "month", "day"]] = Best_Mystery_df['release_date'].str.split("-", expand = True)
    Best_Mystery_df = Best_Mystery_df[Best_Mystery_df['genres'].str.contains('Mystery')]

    #Creating new attribbute that need to find weighted average
    Best_Mystery_df['vote_average_mean'] = Best_Mystery_df['vote_average'].mean()
    Best_Mystery_df['vote_count_quantile'] = Best_Mystery_df['vote_count'].quantile(0.70)

    # weighted average
    Best_Mystery_df['ratings'] = Best_Mystery_df.apply(lambda row: ((row.vote_average * row.vote_count) + (row.vote_average_mean * row.vote_count_quantile))/(row.vote_count + row.vote_count_quantile), axis=1)


    # Important attributes
    Best_Mystery_df = Best_Mystery_df[['id','title','ratings','year','status']]

    #filling missing value with 0
    Best_Mystery_df = Best_Mystery_df.fillna(0)

    #Converting string dtypes into integer
    Best_Mystery_df['year'] = Best_Mystery_df['year'].astype(int)

    #Taking latest movies
    Best_Mystery_df = Best_Mystery_df.loc[Best_Mystery_df['year'] >= 2017]

    #Sort recommendations in descending order
    Best_Mystery_df = Best_Mystery_df.sort_values(by = ['ratings'],ascending=False)

    #Taking top 10 movies
    Most_Popular_Mystery_Movies = Best_Mystery_df.head(15)

    return Most_Popular_Mystery_Movies

#Family movies

def Family_Movies(Best_Family_df):

    Best_Family_df =  Best_Family_df[['id','title','genres','vote_average','vote_count','release_date','status']]

    #filling missing value with 0
    Best_Family_df =  Best_Family_df.fillna(0)

    #Taking the movie that are available
    Best_Family_df =  Best_Family_df[ Best_Family_df["status"] == 'Released']


   #Sepating relase date into year,month,day
    Best_Family_df[["year", "month", "day"]] = Best_Family_df['release_date'].str.split("-", expand = True)
    Best_Family_df = Best_Family_df[Best_Family_df['genres'].str.contains('Family')]

    #Creating new attribbute that need to find weighted average
    Best_Family_df['vote_average_mean'] =  Best_Family_df['vote_average'].mean()
    Best_Family_df['vote_count_quantile'] =  Best_Family_df['vote_count'].quantile(0.70)

    # weighted average
    Best_Family_df['ratings'] = Best_Family_df.apply(lambda row: ((row.vote_average * row.vote_count) + (row.vote_average_mean * row.vote_count_quantile))/(row.vote_count + row.vote_count_quantile), axis=1)


    # Important attributes
    Best_Family_df = Best_Family_df[['id','title','ratings','year','status']]

    #filling missing value with 0
    Best_Family_df = Best_Family_df.fillna(0)

    #Converting string dtypes into integer
    Best_Family_df['year'] = Best_Family_df['year'].astype(int)

    #Taking latest movies
    Best_Family_df = Best_Family_df.loc[Best_Family_df['year'] >= 2016]

    #Sort recommendations in descending order
    Best_Family_df = Best_Family_df.sort_values(by = ['ratings'],ascending=False)

    #Taking top 10 movies
    Most_Popular_Family_Movies = Best_Family_df.head(15)

    return Most_Popular_Family_Movies

#History movies

def History_Movies(Best_History_df):

    Best_History_df =  Best_History_df[['id','title','genres','vote_average','vote_count','release_date','status']]

    #filling missing value with 0
    Best_History_df =  Best_History_df.fillna(0)

    #Taking the movie that are available
    Best_History_df=  Best_History_df[Best_History_df["status"] == 'Released']


   #Sepating relase date into year,month,day
    Best_History_df[["year", "month", "day"]] = Best_History_df['release_date'].str.split("-", expand = True)
    Best_History_df = Best_History_df[Best_History_df['genres'].str.contains('History')]

    #Creating new attribbute that need to find weighted average
    Best_History_df['vote_average_mean'] =  Best_History_df['vote_average'].mean()
    Best_History_df['vote_count_quantile'] =  Best_History_df['vote_count'].quantile(0.70)

    # weighted average
    Best_History_df['ratings'] = Best_History_df.apply(lambda row: ((row.vote_average * row.vote_count) + (row.vote_average_mean * row.vote_count_quantile))/(row.vote_count + row.vote_count_quantile), axis=1)


    # Important attributes
    Best_History_df = Best_History_df[['id','title','ratings','year','status']]

    #filling missing value with 0
    Best_History_df = Best_History_df.fillna(0)

    #Converting string dtypes into integer
    Best_History_df['year'] = Best_History_df['year'].astype(int)

    #Taking latest movies
    Best_History_df = Best_History_df.loc[Best_History_df['year'] >= 2016]

    #Sort recommendations in descending order
    Best_History_df = Best_History_df.sort_values(by = ['ratings'],ascending=False)

    #Taking top 10 movies
    Most_Popular_History_Movies = Best_History_df.head(15)

    return Most_Popular_History_Movies


#FOREIGN MOVIES

def Foreign_Movies(Best_Foreign_df):

    Best_Foreign_df =  Best_Foreign_df[['id','title','genres','vote_average','vote_count','release_date','status']]

    #filling missing value with 0
    Best_Foreign_df =  Best_Foreign_df.fillna(0)

    #Taking the movie that are available
    Best_Foreign_df =  Best_Foreign_df[Best_Foreign_df["status"] == 'Released']


   #Sepating relase date into year,month,day
    Best_Foreign_df[["year", "month", "day"]] = Best_Foreign_df['release_date'].str.split("-", expand = True)
    Best_Foreign_df = Best_Foreign_df[Best_Foreign_df['genres'].str.contains('Foreign')]

    #Creating new attribbute that need to find weighted average
    Best_Foreign_df['vote_average_mean'] =  Best_Foreign_df['vote_average'].mean()
    Best_Foreign_df['vote_count_quantile'] =  Best_Foreign_df['vote_count'].quantile(0.70)

    # weighted average
    Best_Foreign_df['ratings'] = Best_Foreign_df.apply(lambda row: ((row.vote_average * row.vote_count) + (row.vote_average_mean * row.vote_count_quantile))/(row.vote_count + row.vote_count_quantile), axis=1)


    # Important attributes
    Best_Foreign_df = Best_Foreign_df[['id','title','ratings','year','status']]

    #filling missing value with 0
    Best_Foreign_df = Best_Foreign_df.fillna(0)

    #Converting string dtypes into integer
    Best_Foreign_df['year'] = Best_Foreign_df['year'].astype(int)

    #Taking latest movies
    Best_Foreign_df = Best_Foreign_df.loc[Best_Foreign_df['year'] >= 2010]

    #Sort recommendations in descending order
    Best_Foreign_df = Best_Foreign_df.sort_values(by = ['ratings'],ascending=False)

    #Taking top 10 movies
    Most_Popular_Foreign = Best_Foreign_df.head(15)

    return Most_Popular_Foreign