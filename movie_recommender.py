import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from imdb import IMDb


movie_user = joblib.load("movie_user_matrix.pkl")
movies = pd.read_csv("movie_data_collaborative/movies.csv")
movies = movies.set_index('movieId')

# ---------------------------------------------------------------------------------
# ------------------------------ Get Movie Posters --------------------------------
# ---------------------------------------------------------------------------------


@st.cache_resource
def get_movie_poster(movie_name):
    ia = IMDb()
    try:
        result = ia.search_movie(movie_name)[0]
        poster = dict(result)["full-size cover url"]
    except:
        poster = r"https://static.displate.com/857x1200/displate/2022-04-15/7422bfe15b3ea7b5933dffd896e9c7f9_46003a1b7353dc7b5a02949bd074432a.jpg"

    return poster

# ---------------------------------------------------------------------------------
# ------------------------------ Recommender function -----------------------------
# ---------------------------------------------------------------------------------


@st.cache_data
def movie_recommend_user(user_id):
    user_list = movie_user.columns
    similar_users = []
    for i in range(len(user_list)):    # For all user indexes (0 to 425)
        user_user_data = movie_user.iloc[:, [user_id, i]].dropna(
            how='any').T    # user user_id and user i common movie ratings
        if (len(user_user_data.columns) > 30):    # At least they must vote in same 30 movies
            cosine_sim = cosine_similarity(user_user_data)[0][1]
            if (cosine_sim > 0.75):    # Take only if the cosine similarity is > 0.75
                similar_users.append(i)

    # The movies which is not rated by user `user_id`, and the columns of similar users
    null_movies = movie_user[movie_user.iloc[:, user_id].isna()]

    movie_set = {}

    for user in similar_users:
        x = null_movies.iloc[:, user]     # Fetch columns of similar users
        # Only pick the movies with >= 4 ratings by the similar user
        top_rated_movies = x[x >= 4].reset_index()

        # For each rated (movies, ratings) by similar users
        for _, movie_id, rating in top_rated_movies.itertuples():
            if (movie_set.get(movie_id, -1)) == -1:   # If the movie is not present in movie_set, add it
                movie_set[movie_id] = rating
            else:
                # If present, take the mean of ratings as value
                movie_set[movie_id] = np.mean([movie_set[movie_id], rating])

    # Use the rating values to sort in descending order, and getting top 5 movie ids
    top_5_movies = [movie for movie, _ in sorted(
        movie_set.items(), key=lambda x: x[1], reverse=True)][:5]

    # Movie Ids
    top_5_movie_names = movies.loc[top_5_movies, 'title'].to_numpy()
    movie_id_name_dict = dict(zip(top_5_movies, top_5_movie_names))

    return movie_id_name_dict


# ---------------------------------------------------------------------------------
# ------------------------------ Show Recommendations -----------------------------
# ---------------------------------------------------------------------------------

user_list = [f"User {i}" for i in range(1, 426)]
user_dict = dict(zip(user_list, np.arange(426)))   # {User 1: 0, User 2: 1,...}

st.title("Movie Recommender System 📽️🍿")
selected_user = st.selectbox(label="Select an user", options=user_list)

st.markdown("### Recommended Movies")
if selected_user:
    if st.button("Recommend movies"):
        with st.spinner("Please wait"):
            top_movies = movie_recommend_user(user_dict[selected_user])
        cols = st.columns(spec=5)
        for col, movie in zip(cols, top_movies.values()):
            with col:
                st.image(get_movie_poster(movie_name=movie))
                st.write(movie)
