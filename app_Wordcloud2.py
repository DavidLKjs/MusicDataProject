import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer

def wordcloud_page():
    """
    Main function for displaying a WordCloud generation page.
    A dropdown box is placed at the top, and a WordCloud for the selected genre is shown below.
    """
    # Function to load the data
    @st.cache_data
    def load_data():
        """
        Loads the dataset from a Parquet file.

        Returns:
            df (pd.DataFrame): DataFrame containing the song lyrics data.
        """
        try:
            df = pd.read_parquet('reduced_song_lyrics_with_genres.parquet')
            return df
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None

    # Load data
    df = load_data()

    # Check if the data was loaded successfully
    if df is None:
        return

    # Center the dropdown and WordCloud
    st.markdown("<h1 style='text-align: center;'>WordClouds by Genre</h1>", unsafe_allow_html=True)
    
    # Center the dropdown
    genres = df['tag'].unique()
    genre_selected = st.selectbox(
        "Select a genre:",
        genres,
        index=0,
        key="genre_select",
        help="Choose a genre to generate the WordCloud.",
    )

    def generate_wordcloud(genre):
        """
        Generates and displays a WordCloud for the selected genre.

        Args:
            genre (str): The selected genre to filter the data and generate the word cloud.
        """
        genre_df = df[df['tag'] == genre]

        # TF-IDF vectorization
        vectorizer = TfidfVectorizer(max_features=100, stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(genre_df['cleaned_lyrics'])

        # Extract the words and their TF-IDF scores
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray().sum(axis=0)

        # Create the WordCloud
        word_freq = dict(zip(feature_names, tfidf_scores))
        wordcloud = WordCloud(width=800, height=400, background_color='black', colormap='inferno').generate_from_frequencies(word_freq)

        # Display the WordCloud
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)

    # Display the WordCloud for the selected genre
    if genre_selected:
        generate_wordcloud(genre_selected)

# Call the main page function
wordcloud_page()
