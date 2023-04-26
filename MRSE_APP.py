import streamlit as st
import hydralit_components as hc
import streamlit.components.v1 as html
from PIL import Image
import requests
import CallingGenreType
import Homepage
import CallAlgorithms
import Drama
import base64
import pandas as pd
import pickle
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('backgroun.jpg')


menu_data = [
    {'label': "BASED ON ALGORITHMS"},
    {'label': "Genres"},
    {'label': "DRAMA"},
    {'label': "SEARCH MOVIE"},  # no tooltip message
]

over_theme = {'txc_inactive': '#FFFFFF'}
with st.sidebar:
    st.header('MRSE WEB APPLICATION (NOWSHAD)')
    menu_id = hc.nav_bar(menu_definition=menu_data,home_name='Home', key='sidetbar',
                              override_theme=over_theme, first_select=6)


if(menu_id == "BASED ON ALGORITHMS"):
    title = '<p style="font-family:Bebas Neue; color:#F8F0E3; font-size: 20px;">Content Based</p>'
    st.markdown(title, unsafe_allow_html=True)

    recommended_movie_posters, recommended_movie_names = CallAlgorithms.CONTENT_BASED_recommend()
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.image(recommended_movie_posters[0], caption=recommended_movie_names[0], width=None,
                 use_column_width=None, clamp=False, channels="BGR", output_format="auto")
    with col2:
        st.image(recommended_movie_posters[1], caption=recommended_movie_names[1], width=None,
                 use_column_width=None, clamp=False, channels="RGB", output_format="auto")

    with col3:
        st.image(recommended_movie_posters[2], caption=recommended_movie_names[2], width=None,
                 use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    with col4:
        st.image(recommended_movie_posters[3], caption=recommended_movie_names[3], width=None,
                 use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    with col5:
        st.image(recommended_movie_posters[4], caption=recommended_movie_names[4], width=None,
                 use_column_width=None, clamp=False, channels="RGB", output_format="auto")

    title = '<p style="font-family:Bebas Neue; color:#F8F0E3; font-size: 20px;">Content Based without Ratings</p>'
    st.markdown(title, unsafe_allow_html=True)

    recommended_movie_posters, recommended_movie_names = CallAlgorithms.CONTENT_BASED_WITHOUT_RATINGS_recommend()
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.image(recommended_movie_posters[0], caption=recommended_movie_names[0], width=None,
                 use_column_width=None, clamp=False, channels="BGR", output_format="auto")
    with col2:
        st.image(recommended_movie_posters[1], caption=recommended_movie_names[1], width=None,
                 use_column_width=None, clamp=False, channels="RGB", output_format="auto")

    with col3:
        st.image(recommended_movie_posters[2], caption=recommended_movie_names[2], width=None,
                 use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    with col4:
        st.image(recommended_movie_posters[3], caption=recommended_movie_names[3], width=None,
                 use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    with col5:
        st.image(recommended_movie_posters[4], caption=recommended_movie_names[4], width=None,
                 use_column_width=None, clamp=False, channels="RGB", output_format="auto")

    title = '<p style="font-family:Bebas Neue; color:#F8F0E3; font-size: 20px;">Collaborative Filtering</p>'
    st.markdown(title, unsafe_allow_html=True)

    recommended_movie_posters, recommended_movie_names = CallAlgorithms.Collaborative_recommend()
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.image(recommended_movie_posters[0], caption=recommended_movie_names[0], width=None,
                 use_column_width=None, clamp=False, channels="BGR", output_format="auto")
    with col2:
        st.image(recommended_movie_posters[1], caption=recommended_movie_names[1], width=None,
                 use_column_width=None, clamp=False, channels="RGB", output_format="auto")

    with col3:
        st.image(recommended_movie_posters[2], caption=recommended_movie_names[2], width=None,
                 use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    with col4:
        st.image(recommended_movie_posters[3], caption=recommended_movie_names[3], width=None,
                 use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    with col5:
        st.image(recommended_movie_posters[4], caption=recommended_movie_names[4], width=None,
                 use_column_width=None, clamp=False, channels="RGB", output_format="auto")


    title = '<p style="font-family:Bebas Neue; color:#F8F0E3; font-size: 20px;">K-Means Clustering</p>'
    st.markdown(title, unsafe_allow_html=True)

    recommended_movie_posters, recommended_movie_names = CallAlgorithms.Cluster_recommend()
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.image(recommended_movie_posters[0], caption=recommended_movie_names[0], width=None,
                 use_column_width=None, clamp=False, channels="BGR", output_format="auto")
    with col2:
        st.image(recommended_movie_posters[1], caption=recommended_movie_names[1], width=None,
                 use_column_width=None, clamp=False, channels="RGB", output_format="auto")

    with col3:
        st.image(recommended_movie_posters[2], caption=recommended_movie_names[2], width=None,
                 use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    with col4:
        st.image(recommended_movie_posters[3], caption=recommended_movie_names[3], width=None,
                 use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    with col5:
        st.image(recommended_movie_posters[4], caption=recommended_movie_names[4], width=None,
                 use_column_width=None, clamp=False, channels="RGB", output_format="auto")

    title = '<p style="font-family:Bebas Neue; color:#F8F0E3; font-size: 20px;">Hybrid Recommendation</p>'
    st.markdown(title, unsafe_allow_html=True)

    recommended_movie_posters, recommended_movie_names = CallAlgorithms.Hybrid_recommend()
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.image(recommended_movie_posters[0], caption=recommended_movie_names[0], width=None,
                 use_column_width=None, clamp=False, channels="BGR", output_format="auto")
    with col2:
        st.image(recommended_movie_posters[1], caption=recommended_movie_names[1], width=None,
                 use_column_width=None, clamp=False, channels="RGB", output_format="auto")

    with col3:
        st.image(recommended_movie_posters[2], caption=recommended_movie_names[2], width=None,
                 use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    with col4:
        st.image(recommended_movie_posters[3], caption=recommended_movie_names[3], width=None,
                 use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    with col5:
        st.image(recommended_movie_posters[4], caption=recommended_movie_names[4], width=None,
                 use_column_width=None, clamp=False, channels="RGB", output_format="auto")



#SEARCH FUNCTION
if(menu_id == "SEARCH MOVIE"):
    def add_bg_from_local(image_file):
        with open(image_file, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        st.markdown(
            f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
            background-size: cover
        }}
        </style>
        """,
            unsafe_allow_html=True
        )
    add_bg_from_local('background4.png')


    def fetch_poster(movie_id):
        url = "https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US".format(movie_id)
        data = requests.get(url)
        data = data.json()
        poster_path = data['poster_path']
        ratings = data['vote_average']
        overview = data['overview']
        release_date = data['release_date']
        full_path = "https://image.tmdb.org/t/p/w342/" + poster_path
        return full_path,ratings,overview,release_date


    def recommend(movie):
        index = movies[movies['title'] == movie].index[0]
        distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])

        recommended_movie_names = []
        recommended_movie_posters = []
        for i in distances[1:6]:
            # fetch the movie poster
            movie_id = movies.iloc[i[0]].id
            recommended_movie_posters.append(CallAlgorithms.fetch_poster(movie_id))
            recommended_movie_names.append(movies.iloc[i[0]].title)

        return recommended_movie_names, recommended_movie_posters


    movies = pickle.load(open('movie_list.pkl', 'rb'))
    similarity = pickle.load(open('similarity.pkl', 'rb'))

    movie_list = movies['title'].values
    selected_movie = st.selectbox(
        "Type or Select Movie",
        movie_list
    )


    Movie_id = movies.loc[movies.title == selected_movie, 'id'].values[0]
    recommended_movie_posters = []
    movie_poster,movie_ratings,movie_overview,release_date = fetch_poster(int(Movie_id))
    recommended_movie_posters.append(movie_poster)
    col1 = st.columns(1)
    st.image(recommended_movie_posters[0], caption=selected_movie, width=200,
             use_column_width=None, clamp=False, channels="BGR", output_format="auto")
    title1 = '<p style="font-family:Bebas Neue; color:#000000; font-size: 20px;">Rating</p>'
    st.markdown(title1, unsafe_allow_html=True)
    st.text(movie_ratings)
    title2 = '<p style="font-family:Bebas Neue; color:#000000; font-size: 20px;">Overview</p>'
    st.markdown(title2, unsafe_allow_html=True)
    st.text(movie_overview)
    title3 = '<p style="font-family:Bebas Neue; color:#000000; font-size: 20px;">Release_Date</p>'
    st.markdown(title3, unsafe_allow_html=True)
    st.text(release_date)

    filt = movies['title'] == selected_movie
    v = movies.index[filt].tolist()
    index = int(v[-1])
    if (index > 7660):
        title = '<p style="font-family:Bebas Neue; color:#000000; font-size: 25px;">Sorry, Currently No Recommendation to give, Visit again</p>'
        st.markdown(title, unsafe_allow_html=True)
        st.text('Thank You')
    else:
        recommended_movie_names, recommended_movie_posters = recommend(selected_movie)
        title = '<p style="font-family:Bebas Neue; color:#000000; font-size: 30px;">Movies You May Also Like</p>'
        st.markdown(title, unsafe_allow_html=True)
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.image(recommended_movie_posters[0], caption=recommended_movie_names[0], width=None,
                     use_column_width=None, clamp=False, channels="BGR", output_format="auto")
        with col2:
            st.image(recommended_movie_posters[1], caption=recommended_movie_names[1], width=None,
                     use_column_width=None, clamp=False, channels="RGB", output_format="auto")

        with col3:
            st.image(recommended_movie_posters[2], caption=recommended_movie_names[2], width=None,
                     use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        with col4:
            st.image(recommended_movie_posters[3], caption=recommended_movie_names[3], width=None,
                     use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        with col5:
            st.image(recommended_movie_posters[4], caption=recommended_movie_names[4], width=None,
                     use_column_width=None, clamp=False, channels="RGB", output_format="auto")


if (menu_id == "DRAMA"):

    title = '<p style="font-family:Bebas Neue; color:#F8F0E3; font-size: 20px;">Best Drama</p>'
    st.markdown(title, unsafe_allow_html=True)

    recommended_movie_posters, recommended_movie_names = Drama.Drama_recommend()
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.image(recommended_movie_posters[0], caption=recommended_movie_names[0], width=None,
                 use_column_width=None, clamp=False, channels="BGR", output_format="auto")
    with col2:
        st.image(recommended_movie_posters[1], caption=recommended_movie_names[1], width=None,
                 use_column_width=None, clamp=False, channels="RGB", output_format="auto")

    with col3:
        st.image(recommended_movie_posters[2], caption=recommended_movie_names[2], width=None,
                 use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    with col4:
        st.image(recommended_movie_posters[3], caption=recommended_movie_names[3], width=None,
                 use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    with col5:
        st.image(recommended_movie_posters[4], caption=recommended_movie_names[4], width=None,
                 use_column_width=None, clamp=False, channels="RGB", output_format="auto")


if (menu_id == "Home"):
    title4 = '<p style="font-family:Bebas Neue; color:#F8F0E3; font-size: 20px;">Popular Movies</p>'
    st.markdown(title4, unsafe_allow_html=True)

    recommended_movie_posters, recommended_movie_names = Homepage.most_popular_recommend()
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.image(recommended_movie_posters[0], caption=recommended_movie_names[0], width=None,
                 use_column_width=None, clamp=False, channels="BGR", output_format="auto")
    with col2:
        st.image(recommended_movie_posters[1], caption=recommended_movie_names[1], width=None,
                 use_column_width=None, clamp=False, channels="RGB", output_format="auto")

    with col3:
        st.image(recommended_movie_posters[2], caption=recommended_movie_names[2], width=None,
                 use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    with col4:
        st.image(recommended_movie_posters[3], caption=recommended_movie_names[3], width=None,
                 use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    with col5:
        st.image(recommended_movie_posters[4], caption=recommended_movie_names[4], width=None,
                 use_column_width=None, clamp=False, channels="RGB", output_format="auto")

    title5 = '<p style="font-family:Bebas Neue; color:#F8F0E3; font-size: 20px;">Trending Movies</p>'
    st.markdown(title5, unsafe_allow_html=True)

    recommended_movie_posters, recommended_movie_names = Homepage.Trending_recommend()
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.image(recommended_movie_posters[0], caption=recommended_movie_names[0], width=None,
                 use_column_width=None, clamp=False, channels="BGR", output_format="auto")
    with col2:
        st.image(recommended_movie_posters[1], caption=recommended_movie_names[1], width=None,
                 use_column_width=None, clamp=False, channels="RGB", output_format="auto")

    with col3:
        st.image(recommended_movie_posters[2], caption=recommended_movie_names[2], width=None,
                 use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    with col4:
        st.image(recommended_movie_posters[3], caption=recommended_movie_names[3], width=None,
                 use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    with col5:
        st.image(recommended_movie_posters[4], caption=recommended_movie_names[4], width=None,
                 use_column_width=None, clamp=False, channels="RGB", output_format="auto")

    title6 = '<p style="font-family:Bebas Neue; color:#F8F0E3; font-size: 20px;">Upcoming Movies</p>'
    st.markdown(title6, unsafe_allow_html=True)

    recommended_movie_posters, recommended_movie_names = Homepage.Upcoming_recommend()
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.image(recommended_movie_posters[0], caption=recommended_movie_names[0], width=None,
                 use_column_width=None, clamp=False, channels="BGR", output_format="auto")
    with col2:
        st.image(recommended_movie_posters[1], caption=recommended_movie_names[1], width=None,
                 use_column_width=None, clamp=False, channels="RGB", output_format="auto")

    with col3:
        st.image(recommended_movie_posters[2], caption=recommended_movie_names[2], width=None,
                 use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    with col4:
        st.image(recommended_movie_posters[3], caption=recommended_movie_names[3], width=None,
                 use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    with col5:
        st.image(recommended_movie_posters[4], caption=recommended_movie_names[4], width=None,
                 use_column_width=None, clamp=False, channels="RGB", output_format="auto")



if (menu_id == "Genres"):
    override = st.selectbox('TOP MOVIES BASED ON GENRE', ["Animaition","Adventure","Comedy","Action","Fantasy",
                                                          "Romance","Family","History","Mystery","Thriller",
                                                           "Horror","Foreign","Crime"])
    #menu_id = hc.nav_bar(menu_definition=menu_data, key='PrimaryNav', first_select=override)
    col1, col2, col3, col4, col5 = st.columns(5)
    if(override == "Animaition"):

            recommended_movie_posters, recommended_movie_names = CallingGenreType.Animation_recommend()
            with col1:

                st.image(recommended_movie_posters[0], caption=recommended_movie_names[0], width=None,
                         use_column_width=None, clamp=False, channels="RGB", output_format="auto")
            with col2:
                st.image(recommended_movie_posters[1], caption=recommended_movie_names[1], width=None,
                         use_column_width=None, clamp=False, channels="RGB", output_format="auto")

            with col3:
                st.image(recommended_movie_posters[2], caption=recommended_movie_names[2], width=None,
                         use_column_width=None, clamp=False, channels="RGB", output_format="auto")
            with col4:
                st.image(recommended_movie_posters[3], caption=recommended_movie_names[3], width=None,
                         use_column_width=None, clamp=False, channels="RGB", output_format="auto")
            with col5:
                st.image(recommended_movie_posters[4], caption=recommended_movie_names[4], width=None,
                         use_column_width=None, clamp=False, channels="RGB", output_format="auto")

    if (override == "Adventure"):

        recommended_movie_posters, recommended_movie_names = CallingGenreType.Adventure_recommend()
        with col1:
            st.image(recommended_movie_posters[0], caption=recommended_movie_names[0], width=None,
                     use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        with col2:
            st.image(recommended_movie_posters[1], caption=recommended_movie_names[1], width=None,
                     use_column_width=None, clamp=False, channels="RGB", output_format="auto")

        with col3:
            st.image(recommended_movie_posters[2], caption=recommended_movie_names[2], width=None,
                     use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        with col4:
            st.image(recommended_movie_posters[3], caption=recommended_movie_names[3], width=None,
                     use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        with col5:
            st.image(recommended_movie_posters[4], caption=recommended_movie_names[4], width=None,
                     use_column_width=None, clamp=False, channels="RGB", output_format="auto")


    if (override == "Comedy"):
        recommended_movie_posters, recommended_movie_names = CallingGenreType.Comedy_recommend()
        with col1:
            st.image(recommended_movie_posters[0], caption=recommended_movie_names[0], width=None,
                     use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        with col2:
            st.image(recommended_movie_posters[1], caption=recommended_movie_names[1], width=None,
                     use_column_width=None, clamp=False, channels="RGB", output_format="auto")

        with col3:
            st.image(recommended_movie_posters[2], caption=recommended_movie_names[2], width=None,
                     use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        with col4:
            st.image(recommended_movie_posters[3], caption=recommended_movie_names[3], width=None,
                     use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        with col5:
            st.image(recommended_movie_posters[4], caption=recommended_movie_names[4], width=None,
                     use_column_width=None, clamp=False, channels="RGB", output_format="auto")


    if (override == "Fantasy"):
        recommended_movie_posters, recommended_movie_names = CallingGenreType.Fantasy_recommend()
        with col1:
            st.image(recommended_movie_posters[0], caption=recommended_movie_names[0], width=None,
                     use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        with col2:
            st.image(recommended_movie_posters[1], caption=recommended_movie_names[1], width=None,
                     use_column_width=None, clamp=False, channels="RGB", output_format="auto")

        with col3:
            st.image(recommended_movie_posters[2], caption=recommended_movie_names[2], width=None,
                     use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        with col4:
            st.image(recommended_movie_posters[3], caption=recommended_movie_names[3], width=None,
                     use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        with col5:
            st.image(recommended_movie_posters[4], caption=recommended_movie_names[4], width=None,
                     use_column_width=None, clamp=False, channels="RGB", output_format="auto")

    if (override == "Action"):
        recommended_movie_posters, recommended_movie_names = CallingGenreType.Action_recommend()
        with col1:
            st.image(recommended_movie_posters[0], caption=recommended_movie_names[0], width=None,
                     use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        with col2:
            st.image(recommended_movie_posters[1], caption=recommended_movie_names[1], width=None,
                     use_column_width=None, clamp=False, channels="RGB", output_format="auto")

        with col3:
            st.image(recommended_movie_posters[2], caption=recommended_movie_names[2], width=None,
                     use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        with col4:
            st.image(recommended_movie_posters[3], caption=recommended_movie_names[3], width=None,
                     use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        with col5:
            st.image(recommended_movie_posters[4], caption=recommended_movie_names[4], width=None,
                     use_column_width=None, clamp=False, channels="RGB", output_format="auto")

    if (override == "Romance"):
        recommended_movie_posters, recommended_movie_names = CallingGenreType.Romance_recommend()
        with col1:
            st.image(recommended_movie_posters[0], caption=recommended_movie_names[0], width=None,
                     use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        with col2:
            st.image(recommended_movie_posters[1], caption=recommended_movie_names[1], width=None,
                     use_column_width=None, clamp=False, channels="RGB", output_format="auto")

        with col3:
            st.image(recommended_movie_posters[2], caption=recommended_movie_names[2], width=None,
                     use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        with col4:
            st.image(recommended_movie_posters[3], caption=recommended_movie_names[3], width=None,
                     use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        with col5:
            st.image(recommended_movie_posters[4], caption=recommended_movie_names[4], width=None,
                     use_column_width=None, clamp=False, channels="RGB", output_format="auto")

    if (override == "Horror"):
        recommended_movie_posters, recommended_movie_names = CallingGenreType.Horror_recommend()
        with col1:
            st.image(recommended_movie_posters[0], caption=recommended_movie_names[0], width=None,
                     use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        with col2:
            st.image(recommended_movie_posters[1], caption=recommended_movie_names[1], width=None,
                     use_column_width=None, clamp=False, channels="RGB", output_format="auto")

        with col3:
            st.image(recommended_movie_posters[2], caption=recommended_movie_names[2], width=None,
                     use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        with col4:
            st.image(recommended_movie_posters[3], caption=recommended_movie_names[3], width=None,
                     use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        with col5:
            st.image(recommended_movie_posters[4], caption=recommended_movie_names[4], width=None,
                     use_column_width=None, clamp=False, channels="RGB", output_format="auto")

    if (override == "Crime"):
        recommended_movie_posters, recommended_movie_names = CallingGenreType.Crime_recommend()
        with col1:
            st.image(recommended_movie_posters[0], caption=recommended_movie_names[0], width=None,
                     use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        with col2:
            st.image(recommended_movie_posters[1], caption=recommended_movie_names[1], width=None,
                     use_column_width=None, clamp=False, channels="RGB", output_format="auto")

        with col3:
            st.image(recommended_movie_posters[2], caption=recommended_movie_names[2], width=None,
                     use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        with col4:
            st.image(recommended_movie_posters[3], caption=recommended_movie_names[3], width=None,
                     use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        with col5:
            st.image(recommended_movie_posters[4], caption=recommended_movie_names[4], width=None,
                     use_column_width=None, clamp=False, channels="RGB", output_format="auto")

    if (override == "Thriller"):
        recommended_movie_posters, recommended_movie_names = CallingGenreType.Thriller_recommend()
        with col1:
            st.image(recommended_movie_posters[0], caption=recommended_movie_names[0], width=None,
                     use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        with col2:
            st.image(recommended_movie_posters[1], caption=recommended_movie_names[1], width=None,
                     use_column_width=None, clamp=False, channels="RGB", output_format="auto")

        with col3:
            st.image(recommended_movie_posters[2], caption=recommended_movie_names[2], width=None,
                     use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        with col4:
            st.image(recommended_movie_posters[3], caption=recommended_movie_names[3], width=None,
                     use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        with col5:
            st.image(recommended_movie_posters[4], caption=recommended_movie_names[4], width=None,
                     use_column_width=None, clamp=False, channels="RGB", output_format="auto")

    if (override == "Mystery"):
        recommended_movie_posters, recommended_movie_names = CallingGenreType.Mystery_recommend()
        with col1:
            st.image(recommended_movie_posters[0], caption=recommended_movie_names[0], width=None,
                     use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        with col2:
            st.image(recommended_movie_posters[1], caption=recommended_movie_names[1], width=None,
                     use_column_width=None, clamp=False, channels="RGB", output_format="auto")

        with col3:
            st.image(recommended_movie_posters[2], caption=recommended_movie_names[2], width=None,
                     use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        with col4:
            st.image(recommended_movie_posters[3], caption=recommended_movie_names[3], width=None,
                     use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        with col5:
            st.image(recommended_movie_posters[4], caption=recommended_movie_names[4], width=None,
                     use_column_width=None, clamp=False, channels="RGB", output_format="auto")

    if (override == "Family"):
        recommended_movie_posters, recommended_movie_names = CallingGenreType.Family_recommend()
        with col1:
            st.image(recommended_movie_posters[0], caption=recommended_movie_names[0], width=None,
                     use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        with col2:
            st.image(recommended_movie_posters[1], caption=recommended_movie_names[1], width=None,
                     use_column_width=None, clamp=False, channels="RGB", output_format="auto")

        with col3:
            st.image(recommended_movie_posters[2], caption=recommended_movie_names[2], width=None,
                     use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        with col4:
            st.image(recommended_movie_posters[3], caption=recommended_movie_names[3], width=None,
                     use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        with col5:
            st.image(recommended_movie_posters[4], caption=recommended_movie_names[4], width=None,
                     use_column_width=None, clamp=False, channels="RGB", output_format="auto")



    if (override == "History"):
        recommended_movie_posters, recommended_movie_names = CallingGenreType.History_recommend()
        with col1:
            st.image(recommended_movie_posters[0], caption=recommended_movie_names[0], width=None,
                     use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        if(len(recommended_movie_posters) >2):

            with col2:
                st.image(recommended_movie_posters[1], caption=recommended_movie_names[1], width=None,
                         use_column_width=None, clamp=False, channels="RGB", output_format="auto")

            with col3:
                st.image(recommended_movie_posters[2], caption=recommended_movie_names[2], width=None,
                         use_column_width=None, clamp=False, channels="RGB", output_format="auto")
            with col4:
                st.image(recommended_movie_posters[3], caption=recommended_movie_names[3], width=None,
                         use_column_width=None, clamp=False, channels="RGB", output_format="auto")
            with col5:
                st.image(recommended_movie_posters[4], caption=recommended_movie_names[4], width=None,
                         use_column_width=None, clamp=False, channels="RGB", output_format="auto")

    if (override == "Foreign"):
        recommended_movie_posters, recommended_movie_names = CallingGenreType.Foreign_recommend()
        with col1:
            st.image(recommended_movie_posters[0], caption=recommended_movie_names[0], width=None,
                     use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        if (len(recommended_movie_posters) > 2):
            with col2:
                st.image(recommended_movie_posters[1], caption=recommended_movie_names[1], width=None,
                         use_column_width=None, clamp=False, channels="RGB", output_format="auto")

            with col3:
                st.image(recommended_movie_posters[2], caption=recommended_movie_names[2], width=None,
                         use_column_width=None, clamp=False, channels="RGB", output_format="auto")
            with col4:
                st.image(recommended_movie_posters[3], caption=recommended_movie_names[3], width=None,
                         use_column_width=None, clamp=False, channels="RGB", output_format="auto")
            with col5:
                st.image(recommended_movie_posters[4], caption=recommended_movie_names[4], width=None,
                         use_column_width=None, clamp=False, channels="RGB", output_format="auto")