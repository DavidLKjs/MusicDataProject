import streamlit as st
st.set_page_config(layout="wide") 
from genre_features import genre_features_page
from ranked_features import ranked_features_page
from app_Classification import classification_page
from app_Wordcloud2 import wordcloud_page
from landing_page import landing_page


# Sidebar mit Seitenoptionen als Radio-Buttons
page = st.sidebar.radio("Choose Page:", ["Landing Page", "Wordcloud", "Classification", "Genre Features", "Genres Ranked by Feature"])

if page == "Landing Page":
    landing_page()

elif page == "Wordcloud":
    wordcloud_page()

elif page == "Classification":
    classification_page()

elif page == "Genre Features":
    genre_features_page()

elif page == "Genres Ranked by Feature":
    ranked_features_page()