import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer

def wordcloud_page():
    # CSS für den Dark Mode (angepasst an warme Töne und Schwarz)
    st.markdown("""
        <style>
            .main {
                background-color: #0E1117;
                color: #FFFFFF;
            }
            .stButton>button {
                background-color: #C64E40;
                color: #F1F1F1;
            }
            h1, h2, h3, h4 {
                color: #F1F1F1;
            }
            .stTextInput>div>input {
                background-color: #1E1E1E;
                color: #FFFFFF;
            }
            .stTextArea>div>textarea {
                background-color: #1E1E1E;
                color: #FFFFFF;
            }
            .stMarkdown {
                color: #F1F1F1;
            }
            .css-1v0mbdj {
                color: #F1F1F1;
            }
            .stSidebar {
                background-color: #15171A;
            }
            .stHeader {
                background-color: #1E1E1E;
            }
            .block-container {
                background-color: #1E1E1E;
            }
        </style>
    """, unsafe_allow_html=True)

    # Funktion zum Laden der Daten
    @st.cache_data
    def load_data():
        df = pd.read_parquet('reduced_song_lyrics_with_genres.parquet')
        return df

    # Daten laden
    df = load_data()

    # Punkt: WordClouds pro Genre
    st.title("WordClouds nach Genre")
    st.subheader("Wähle ein Genre und sieh dir die häufigsten Wörter an.")

    # Auswahlmenü für Genre
    genres = df['tag'].unique()
    genre_selected = st.selectbox("Wähle ein Genre:", genres)

    # Funktion zur Erstellung und Anzeige der WordCloud
    def generate_wordcloud(genre):
        genre_df = df[df['tag'] == genre]
        
        # TF-IDF Vektorisierung
        vectorizer = TfidfVectorizer(max_features=100, stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(genre_df['cleaned_lyrics'])
        
        # Extrahiere die Wörter und ihre TF-IDF-Werte
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray().sum(axis=0)
        
        # Erstelle die Wordcloud
        word_freq = dict(zip(feature_names, tfidf_scores))
        wordcloud = WordCloud(width=800, height=400, background_color='black', colormap='inferno').generate_from_frequencies(word_freq)

        # Wordcloud anzeigen
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)

    # Wordcloud für das ausgewählte Genre anzeigen
    if genre_selected:
        st.subheader(f"WordCloud für {genre_selected.capitalize()}")
        generate_wordcloud(genre_selected)
