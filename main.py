import streamlit as st
st.set_page_config(layout="wide") 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from textblob import TextBlob
import random
from genre_features import genre_features_page
from ranked_features import ranked_features_page


# Sidebar mit Seitenoptionen als Radio-Buttons
page = st.sidebar.radio("Wähle eine Seite:", ["Seite 1", "Genre Features", "Genres ranked by Feature"])

if page == "Seite 1":
    st.header("Seite 1")
    st.write("Platzhalter")

elif page == "Genre Features":
    genre_features_page()

elif page == "Genres ranked by Feature":
    ranked_features_page()