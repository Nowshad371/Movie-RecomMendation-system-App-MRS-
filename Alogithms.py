import pandas as pd
from math import sqrt
import numpy as np
from sklearn.preprocessing import StandardScaler
from collections import Counter
from itertools import groupby
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist




# CONTENT BASED WITH RATINGS
def UserInput(inputMovies, movies_df):
    # Renaming the columns in movies_df to match column name with ratings and inputUsers columns name
    movies_df.rename(columns={'id': 'movieId'}, inplace=True)
    movies_df.rename(columns={'original_title': 'title'}, inplace=True)

    # Separating the genre categories with comma and []

    movies_df.dropna(subset=['genres'], inplace=True)
    movies_df['genres'] = movies_df.genres.str.split(',')

    # Taking all columns and values from movies_df into moviesWithGenres_df
    moviesWithGenres_df = movies_df.copy()

    # For every row in the dataframe, iterate through the list of genres and place a 1 into the corresponding column
    for index, row in movies_df.iterrows():
        for genre in row['genres']:
            moviesWithGenres_df.at[index, genre] = 1

    # Filling in the NaN values with 0 to show that a movie doesn't have that column's genre
    moviesWithGenres_df = moviesWithGenres_df.fillna(0)

    # Deleting Column that bears unnecessary information
    moviesWithGenres_df = moviesWithGenres_df.drop(
        moviesWithGenres_df.columns[[43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57]], axis=1)

    # Merging two same column values into a new column
    moviesWithGenres_df['Animation'] = moviesWithGenres_df.loc[:, ['   Animation', '    Animation']].sum(axis=1)
    moviesWithGenres_df['Comedy'] = moviesWithGenres_df.loc[:, ['    Comedy', '   Comedy']].sum(axis=1)
    moviesWithGenres_df['Family'] = moviesWithGenres_df.loc[:, ['    Family', '   Family']].sum(axis=1)
    moviesWithGenres_df['Adventure'] = moviesWithGenres_df.loc[:, ['   Adventure', '    Adventure']].sum(axis=1)
    moviesWithGenres_df['Fantasy'] = moviesWithGenres_df.loc[:, ['    Fantasy', '   Fantasy']].sum(axis=1)
    moviesWithGenres_df['Romance'] = moviesWithGenres_df.loc[:, ['   Romance', '    Romance']].sum(axis=1)
    moviesWithGenres_df['Drama'] = moviesWithGenres_df.loc[:, ['    Drama', '   Drama']].sum(axis=1)
    moviesWithGenres_df['Action'] = moviesWithGenres_df.loc[:, ['   Action', '    Action']].sum(axis=1)
    moviesWithGenres_df['Crime'] = moviesWithGenres_df.loc[:, ['    Crime', '   Crime']].sum(axis=1)
    moviesWithGenres_df['Thriller'] = moviesWithGenres_df.loc[:, ['    Thriller', '   Thriller']].sum(axis=1)
    moviesWithGenres_df['Horror'] = moviesWithGenres_df.loc[:, ['    Horror', '   Horror']].sum(axis=1)
    moviesWithGenres_df['History'] = moviesWithGenres_df.loc[:, ['   History', '    History']].sum(axis=1)
    moviesWithGenres_df['Science Fiction'] = moviesWithGenres_df.loc[:,
                                             ['    Science Fiction', '   Science Fiction']].sum(axis=1)
    moviesWithGenres_df['Mystery'] = moviesWithGenres_df.loc[:, ['    Mystery', '   Mystery']].sum(axis=1)
    moviesWithGenres_df['War'] = moviesWithGenres_df.loc[:, ['    War', '   War']].sum(axis=1)
    moviesWithGenres_df['Foreign'] = moviesWithGenres_df.loc[:, ['   Foreign', '    Foreign']].sum(axis=1)
    moviesWithGenres_df['Music'] = moviesWithGenres_df.loc[:, ['    Music', '   Music']].sum(axis=1)
    moviesWithGenres_df['Documentary'] = moviesWithGenres_df.loc[:, ['   Documentary', '    Documentary']].sum(axis=1)
    moviesWithGenres_df['Western'] = moviesWithGenres_df.loc[:, ['    Western', '   Western']].sum(axis=1)
    moviesWithGenres_df['TV Movie'] = moviesWithGenres_df.loc[:, ['    TV Movie', '   TV Movie']].sum(axis=1)

    # Removing duplicate columns from moviesWithGenres_df
    moviesWithGenres_df.drop(moviesWithGenres_df.iloc[:, 3:43], inplace=True, axis=1)

    # Filtering out the movies by title
    inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]

    # Dropping information we won't use from the input dataframe
    inputId = inputId.drop(inputId.columns[2], axis=1)

    # Then merging it so we can get the movieId. It's implicitly merging it by title.
    inputMovies = pd.merge(inputId, inputMovies)

    # Filtering out the movies from the input
    userMovies = moviesWithGenres_df[moviesWithGenres_df['movieId'].isin(inputMovies['movieId'].tolist())]

    # Resetting the index to avoid future issues
    userMovies = userMovies.reset_index(drop=True)

    # Dropping unnecessary issues due to save memory and to avoid issues
    userGenreTable = userMovies.drop('movieId', 1).drop('title', 1).drop('genres', 1)

    # Dot produt to get weights
    userProfile = userGenreTable.transpose().dot(inputMovies['rating'])

    # Now let's get the genres of every movie in our original dataframe
    genreTable = moviesWithGenres_df.set_index(moviesWithGenres_df['movieId'])

    # And drop the unnecessary information
    genreTable = genreTable.drop('movieId', 1).drop('title', 1).drop('genres', 1)

    # Multiply the genres by the weights and then take the weighted average
    recommendationTable_df = ((genreTable * userProfile).sum(axis=1)) / (userProfile.sum())

    # Sort our recommendations in descending order
    recommendationTable_df = recommendationTable_df.sort_values(ascending=False)

    # The final recommendation table
    movies_df.loc[movies_df['movieId'].isin(recommendationTable_df.head(20).keys())]

    recomend = movies_df.loc[movies_df['movieId'].isin(recommendationTable_df.head(20).keys())]
    finalrecomend = recomend.reset_index(drop=True)

    #finalrecomend.drop(finalrecomend.columns[[0, 2]], axis=1, inplace=True)

    recomendation = finalrecomend[['movieId','title']].head(10)

    return recomendation


# CONTENT BASED WITHOUT RATINGS
def contentBasedWithoutRatings(movies_df, inputMovies):
    # movies_df = pd.read_csv (r'/content/Movies_df.csv')
    # inputMovies = pd.read_csv (r'/content/inputMovies.csv')
    # Renaming the columns in movies_df to match column name with ratings and inputUsers columns name
    movies_df.rename(columns={'id': 'movieId'}, inplace=True)
    movies_df.rename(columns={'original_title': 'title'}, inplace=True)

    # Separating the genre categories with comma and []

    movies_df.dropna(subset=['genres'], inplace=True)
    movies_df['genres'] = movies_df.genres.str.split(',')
    id = movies_df['movieId']

    # Taking all columns and values from movies_df into moviesWithGenres_df
    moviesWithGenres_df = movies_df.copy()

    # For every row in the dataframe, iterate through the list of genres and place a 1 into the corresponding column
    for index, row in movies_df.iterrows():
        for genre in row['genres']:
            moviesWithGenres_df.at[index, genre] = 1

    # Filling in the NaN values with 0 to show that a movie doesn't have that column's genre
    moviesWithGenres_df = moviesWithGenres_df.fillna(0)

    # Deleting Column that bears unnecessary information
    moviesWithGenres_df = moviesWithGenres_df.drop(
        moviesWithGenres_df.columns[[43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57]], axis=1)

    # Merging two same column values into a new column
    moviesWithGenres_df['Animation'] = moviesWithGenres_df.loc[:, ['   Animation', '    Animation']].sum(axis=1)
    moviesWithGenres_df['Comedy'] = moviesWithGenres_df.loc[:, ['    Comedy', '   Comedy']].sum(axis=1)
    moviesWithGenres_df['Family'] = moviesWithGenres_df.loc[:, ['    Family', '   Family']].sum(axis=1)
    moviesWithGenres_df['Adventure'] = moviesWithGenres_df.loc[:, ['   Adventure', '    Adventure']].sum(axis=1)
    moviesWithGenres_df['Fantasy'] = moviesWithGenres_df.loc[:, ['    Fantasy', '   Fantasy']].sum(axis=1)
    moviesWithGenres_df['Romance'] = moviesWithGenres_df.loc[:, ['   Romance', '    Romance']].sum(axis=1)
    moviesWithGenres_df['Drama'] = moviesWithGenres_df.loc[:, ['    Drama', '   Drama']].sum(axis=1)
    moviesWithGenres_df['Action'] = moviesWithGenres_df.loc[:, ['   Action', '    Action']].sum(axis=1)
    moviesWithGenres_df['Crime'] = moviesWithGenres_df.loc[:, ['    Crime', '   Crime']].sum(axis=1)
    moviesWithGenres_df['Thriller'] = moviesWithGenres_df.loc[:, ['    Thriller', '   Thriller']].sum(axis=1)
    moviesWithGenres_df['Horror'] = moviesWithGenres_df.loc[:, ['    Horror', '   Horror']].sum(axis=1)
    moviesWithGenres_df['History'] = moviesWithGenres_df.loc[:, ['   History', '    History']].sum(axis=1)
    moviesWithGenres_df['Science Fiction'] = moviesWithGenres_df.loc[:,
                                             ['    Science Fiction', '   Science Fiction']].sum(axis=1)
    moviesWithGenres_df['Mystery'] = moviesWithGenres_df.loc[:, ['    Mystery', '   Mystery']].sum(axis=1)
    moviesWithGenres_df['War'] = moviesWithGenres_df.loc[:, ['    War', '   War']].sum(axis=1)
    moviesWithGenres_df['Foreign'] = moviesWithGenres_df.loc[:, ['   Foreign', '    Foreign']].sum(axis=1)
    moviesWithGenres_df['Music'] = moviesWithGenres_df.loc[:, ['    Music', '   Music']].sum(axis=1)
    moviesWithGenres_df['Documentary'] = moviesWithGenres_df.loc[:, ['   Documentary', '    Documentary']].sum(axis=1)
    moviesWithGenres_df['Western'] = moviesWithGenres_df.loc[:, ['    Western', '   Western']].sum(axis=1)
    moviesWithGenres_df['TV Movie'] = moviesWithGenres_df.loc[:, ['    TV Movie', '   TV Movie']].sum(axis=1)

    # Removing duplicate columns from moviesWithGenres_df
    moviesWithGenres_df.drop(moviesWithGenres_df.iloc[:, 3:43], inplace=True, axis=1)

    # Removing Movie ID and genres columns (Unnecessary columns)
    moviesWithGenres_df = moviesWithGenres_df.drop(moviesWithGenres_df.columns[[0, 2]], axis=1)

    inputMovies = moviesWithGenres_df[moviesWithGenres_df['title'].isin(inputMovies['title'].tolist())]
    inputMoviesWithGenres_df = inputMovies.drop('title', 1)

    # MULTIPLYING EACH OF THE DATASET GENRES COLUMNS BY USERS SUM OF THE GENRE
    intersect = moviesWithGenres_df.columns.intersection(inputMoviesWithGenres_df.columns)

    colno = 0
    for col in intersect:
        sum = inputMoviesWithGenres_df[col].sum()
        moviesWithGenres_df[col] = moviesWithGenres_df[col] * sum
        colno += 1

    # taking all genres into movieRatingTable to find out the relavent movies
    movieRatingTable = moviesWithGenres_df.drop('title', 1)

    # creating ratings depending on most watched genres
    ratings = []
    ratings = (movieRatingTable.sum(axis=1) / colno) * 5

    # Adding created ratings into orginal dataset
    moviesWithGenres_df['ratings'] = ratings
    moviesWithGenres_df['movieId'] = id
    recomendation = moviesWithGenres_df[['title', 'ratings', 'movieId']]

    # Sort our recommendations in descending order
    recomendation = recomendation.sort_values(by='ratings', ascending=False)
    recomendation = recomendation[['movieId', 'title']].head(10)

    return recomendation


def collaborative(ratings_df, movies_df, inputMovies,HMovies_df):
    # Using regular expressions to find a year stored between parentheses
    # We specify the parantheses so we don't conflict with movies that have years in their titles
    movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))', expand=False)
    # Removing the parentheses
    movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)', expand=False)
    # Removing the years from the 'title' column
    movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')
    # Applying the strip function to get rid of any ending whitespace characters that may have appeared
    movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())
    # Dropping the genres column
    movies_df = movies_df.drop('genres', 1)
    #Selecting available movies
    movies_df = movies_df[movies_df['title'].isin(HMovies_df['original_title'].tolist())]
    # Drop removes a specified row or column from a dataframe
    ratings_df = ratings_df.drop('timestamp', 1)
    # Filtering out the movies by title
    inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]
    # Then merging it so we can get the movieId. It's implicitly merging it by title.
    inputMovies = pd.merge(inputId, inputMovies)
    # Dropping information we won't use from the input dataframe
    inputMovies = inputMovies.drop('year', 1)
    # Final input dataframe
    # If a movie you added in above isn't here, then it might not be in the original

    # Filtering out users that have watched movies that the input has watched and storing it
    userSubset = ratings_df[ratings_df['movieId'].isin(inputMovies['movieId'].tolist())]
    # Groupby creates several sub dataframes where they all have the same value in the column specified as the parameter
    userSubsetGroup = userSubset.groupby(['userId'])

    # Sorting it so users with movie most in common with the input will have priority
    userSubsetGroup = sorted(userSubsetGroup, key=lambda x: len(x[1]), reverse=True)

    userSubsetGroup = userSubsetGroup[0:100]

    # Store the Pearson Correlation in a dictionary, where the key is the user Id and the value is the coefficient
    pearsonCorrelationDict = {}

    # For every user group in our subset
    for name, group in userSubsetGroup:
        # Let's start by sorting the input and current user group so the values aren't mixed up later on
        group = group.sort_values(by='movieId')
        inputMovies = inputMovies.sort_values(by='movieId')
        # Get the N for the formula
        nRatings = len(group)
        # Get the review scores for the movies that they both have in common
        temp_df = inputMovies[inputMovies['movieId'].isin(group['movieId'].tolist())]

        # And then store them in a temporary buffer variable in a list format to facilitate future calculations
        tempRatingList = temp_df['rating'].tolist()

        # Let's also put the current user group reviews in a list format
        tempGroupList = group['rating'].tolist()
        # Now let's calculate the pearson correlation between two users, so called, x and y
        Sxx = sum([i ** 2 for i in tempRatingList]) - pow(sum(tempRatingList), 2) / float(nRatings)
        Syy = sum([i ** 2 for i in tempGroupList]) - pow(sum(tempGroupList), 2) / float(nRatings)
        Sxy = sum(i * j for i, j in zip(tempRatingList, tempGroupList)) - sum(tempRatingList) * sum(
            tempGroupList) / float(nRatings)

        # If the denominator is different than zero, then divide, else, 0 correlation.
        if Sxx != 0 and Syy != 0:
            pearsonCorrelationDict[name] = Sxy / sqrt(Sxx * Syy)
        else:
            pearsonCorrelationDict[name] = 0

    pearsonDF = pd.DataFrame.from_dict(pearsonCorrelationDict, orient='index')
    pearsonDF.columns = ['similarityIndex']
    pearsonDF['userId'] = pearsonDF.index
    pearsonDF.index = range(len(pearsonDF))
    #
    topUsers = pearsonDF.sort_values(by='similarityIndex', ascending=False)[0:50]

    #
    topUsersRating = topUsers.merge(ratings_df, left_on='userId', right_on='userId', how='inner')
    # Multiplies the similarity by the user's ratings
    topUsersRating['weightedRating'] = topUsersRating['similarityIndex'] * topUsersRating['rating']
    # Applies a sum to the topUsers after grouping it up by userId
    tempTopUsersRating = topUsersRating.groupby('movieId').sum()[['similarityIndex', 'weightedRating']]
    tempTopUsersRating.columns = ['sum_similarityIndex', 'sum_weightedRating']

    # Creates an empty dataframe
    recommendation_df = pd.DataFrame()
    # Now we take the weighted average
    recommendation_df['weighted average recommendation score'] = tempTopUsersRating['sum_weightedRating'] / \
                                                                 tempTopUsersRating['sum_similarityIndex']
    recommendation_df['movieId'] = tempTopUsersRating.index

    recommendation_df = recommendation_df.sort_values(by='weighted average recommendation score', ascending=False)
    recomendation_df = movies_df.loc[movies_df['movieId'].isin(recommendation_df.head(20)['movieId'].tolist())]
    #Colrecomendation = recomendation_df['title'].head(10)
    return  recomendation_df


# hybrid

def hybrid_Model(inputMovies, movies_df, ratings_df, Movies_df,CMovies_df):
    # CONTENT BASED WITH RATINGS
    ContentBasedWithRatings = UserInput(inputMovies, Movies_df)
    ContentBasedWithRatings = [movies for movies in ContentBasedWithRatings['title']]

    Colrecomendation = collaborative(ratings_df, movies_df, inputMovies,CMovies_df)
    Colrecomendation = [movies for movies in Colrecomendation['title']]

    # Combining both
    Hybrid_recomendaiton = []
    for i in range(3):
        Hybrid_recomendaiton.append(Colrecomendation[i])
        Hybrid_recomendaiton.append(ContentBasedWithRatings[i])

    return Hybrid_recomendaiton


# Clustering
# Clustering

def recomendation_cluster(movies_df, inputMovies, moviesMetaData):
    # inputMovies = inputMovies[inputMovies["rating"] > 3.5]
    # Renaming the columns in movies_df to match column name with ratings and inputUsers columns name
    movies_df.rename(columns={'id': 'movieId'}, inplace=True)
    movies_df.rename(columns={'original_title': 'title'}, inplace=True)
    id = movies_df['movieId']

    # storing rating into list and importing into movies list

    # Taking require attribute
    ratings_df = moviesMetaData[['vote_average', 'vote_count']]
    ratings_df = ratings_df.fillna(0)

    # Creating new attribbute that need to find weighted average
    ratings_df['vote_average_mean'] = ratings_df['vote_average'].mean()
    ratings_df['vote_count_quantile'] = ratings_df['vote_count'].quantile(0.70)

    # weighted average
    ratings_df['ratings'] = ratings_df.apply(
        lambda row: ((row.vote_average * row.vote_count) + (row.vote_average_mean * row.vote_count_quantile)) / (
                    row.vote_count + row.vote_count_quantile), axis=1)

    ratings = ratings_df['ratings']

    movies_df['ratings'] = ratings

    # Separating the genre categories with comma and []

    movies_df.dropna(subset=['genres'], inplace=True)
    movies_df['genres'] = movies_df.genres.str.split(',')

    # Taking all columns and values from movies_df into moviesWithGenres_df
    moviesWithGenres_df = movies_df.copy()

    # For every row in the dataframe, iterate through the list of genres and place a 1 into the corresponding column
    for index, row in movies_df.iterrows():
        for genre in row['genres']:
            moviesWithGenres_df.at[index, genre] = 1

    # Filling in the NaN values with 0 to show that a movie doesn't have that column's genre
    moviesWithGenres_df = moviesWithGenres_df.fillna(0)

    # Deleting Column that bears unnecessary information
    moviesWithGenres_df = moviesWithGenres_df.drop(
        moviesWithGenres_df.columns[[44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58]], axis=1)

    # Merging two same column values into a new column
    moviesWithGenres_df['Animation'] = moviesWithGenres_df.loc[:, ['   Animation', '    Animation']].sum(axis=1)
    moviesWithGenres_df['Comedy'] = moviesWithGenres_df.loc[:, ['    Comedy', '   Comedy']].sum(axis=1)
    moviesWithGenres_df['Family'] = moviesWithGenres_df.loc[:, ['    Family', '   Family']].sum(axis=1)
    moviesWithGenres_df['Adventure'] = moviesWithGenres_df.loc[:, ['   Adventure', '    Adventure']].sum(axis=1)
    moviesWithGenres_df['Fantasy'] = moviesWithGenres_df.loc[:, ['    Fantasy', '   Fantasy']].sum(axis=1)
    moviesWithGenres_df['Romance'] = moviesWithGenres_df.loc[:, ['   Romance', '    Romance']].sum(axis=1)
    moviesWithGenres_df['Drama'] = moviesWithGenres_df.loc[:, ['    Drama', '   Drama']].sum(axis=1)
    moviesWithGenres_df['Action'] = moviesWithGenres_df.loc[:, ['   Action', '    Action']].sum(axis=1)
    moviesWithGenres_df['Crime'] = moviesWithGenres_df.loc[:, ['    Crime', '   Crime']].sum(axis=1)
    moviesWithGenres_df['Thriller'] = moviesWithGenres_df.loc[:, ['    Thriller', '   Thriller']].sum(axis=1)
    moviesWithGenres_df['Horror'] = moviesWithGenres_df.loc[:, ['    Horror', '   Horror']].sum(axis=1)
    moviesWithGenres_df['History'] = moviesWithGenres_df.loc[:, ['   History', '    History']].sum(axis=1)
    moviesWithGenres_df['Science Fiction'] = moviesWithGenres_df.loc[:,
                                             ['    Science Fiction', '   Science Fiction']].sum(axis=1)
    moviesWithGenres_df['Mystery'] = moviesWithGenres_df.loc[:, ['    Mystery', '   Mystery']].sum(axis=1)
    moviesWithGenres_df['War'] = moviesWithGenres_df.loc[:, ['    War', '   War']].sum(axis=1)
    moviesWithGenres_df['Foreign'] = moviesWithGenres_df.loc[:, ['    Foreign', '   Foreign']].sum(axis=1)

    moviesWithGenres_df['Music'] = moviesWithGenres_df.loc[:, ['    Music', '   Music']].sum(axis=1)
    moviesWithGenres_df['Documentary'] = moviesWithGenres_df.loc[:, ['   Documentary', '    Documentary']].sum(axis=1)
    moviesWithGenres_df['Western'] = moviesWithGenres_df.loc[:, ['    Western', '   Western']].sum(axis=1)
    moviesWithGenres_df['TV Movie'] = moviesWithGenres_df.loc[:, ['    TV Movie', '   TV Movie']].sum(axis=1)

    # Removing duplicate columns from moviesWithGenres_df
    moviesWithGenres_df.drop(moviesWithGenres_df.iloc[:, 4:44], inplace=True, axis=1)
    # drop unnecessary columns
    moviesWithGenres_df = moviesWithGenres_df.drop(moviesWithGenres_df.columns[[0, 2]], axis=1)

    # Normalizing the data
    df = moviesWithGenres_df.drop('title', axis=1)
    X = df.values[:, 1:]
    X = np.nan_to_num(X)
    Clus_dataSet = StandardScaler().fit_transform(X)

    # Creating levels of the cluster
    clusterNum = 9
    k_means = KMeans(init="k-means++", n_clusters=clusterNum, n_init=12)
    k_means.fit(X)
    labels = k_means.labels_

    # placing cluster labels into dataset
    moviesWithGenres_df["Clus_km"] = labels

    moviesWithGenres_df["movieId"] = id
    # appending users watched movies into list (movie_Title)
    movie_Title = inputMovies['title']
    movieNumber = []
    for i in range(len(inputMovies)):
        # finding the location of the movies that watched by users in dataset
        filt = moviesWithGenres_df['title'] == movie_Title[i]
        v = moviesWithGenres_df.index[filt].tolist()
        if (len(v) > 0):
            v1 = v[-1]
            # taking cluster number of the movies that seen by users
            value = moviesWithGenres_df.iloc[v1, 22]
            # putting the all cluster into movieNumber List
            movieNumber.append(value)

    # group most_common output by frequency
    freqs = groupby(Counter(movieNumber).most_common(), lambda x: x[1])
    # pick off the first group (highest frequency)/ type of cluster that watched movie belongs
    val = [val for val, count in next(freqs)[1]]

    # Taking the movies cluster that watched most of the time
    moviesCluster = moviesWithGenres_df[moviesWithGenres_df["Clus_km"] == val[0]]
    # Taking highest rated movies from dataset based on ratings and popularity
    moviesWithGenres_df = moviesWithGenres_df[moviesWithGenres_df["ratings"] > 7]
    if (len(moviesCluster) >= 10):

        # Sort recommendations in descending order
        moviesCluster = moviesCluster.sort_values(by='ratings', ascending=False)
        # Removing duplicated values from moviesWithGenres_df i.e., movie that already seen by users
        cond = moviesCluster['title'].isin(inputMovies['title'])
        moviesCluster.drop(moviesCluster[cond].index, inplace=True)
        recomendation = moviesCluster[['movieId', 'title', "ratings", "Clus_km"]].head(10)

    elif (len(val) > 1 and len(moviesCluster)):
        moviesCluster2 = moviesWithGenres_df[moviesWithGenres_df["Clus_km"] == val[1]]
        frames = [moviesCluster, moviesCluster2]
        result = pd.concat(frames)
        # Sort recommendations in descending order
        moviesCluster = result.sort_values(by='ratings', ascending=False)
        # Removing duplicated values from moviesWithGenres_df i.e., movie that already seen by users
        cond = moviesCluster['title'].isin(inputMovies['title'])
        moviesCluster.drop(moviesCluster[cond].index, inplace=True)
        recomendation = moviesCluster[['movieId', 'title', "ratings", "Clus_km"]].head(10)

    else:
        # Sort recommendations in descending order
        moviesCluster = moviesCluster.sort_values(by='ratings', ascending=False)
        # Removing duplicated values from moviesWithGenres_df i.e., movie that already seen by users
        cond = moviesCluster['title'].isin(inputMovies['title'])
        moviesCluster.drop(moviesCluster[cond].index, inplace=True)
        recomendation = moviesCluster[['movieId', 'title', "ratings", "Clus_km"]]
        # recomendation = moviesCluster['title']

    return recomendation[['movieId', 'title']].head(10)


