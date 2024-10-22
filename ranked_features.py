import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def ranked_features_page():
    # Create dataframe
    df = pd.read_csv("train.csv")

    # Calculating ms to minutes
    df['duration_min'] = df['duration_ms'] / (1000 * 60)

    # Deleting ms column
    df.drop(columns=['duration_ms'], inplace=True)

    # Header
    st.header("Ranked Characteristics")

    col1, col2, col3 = st.columns(3)

    with col1:
        # List of characteristics
        characteristics = ['tempo', 'duration_min', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'popularity']
        
        # Characteristic Selectbox
        characteristic_select = st.selectbox(
                'Choose a Characteristic:',
                characteristics
            )

    with col2:
        median_mean_select = st.selectbox(
                'Mean/Median:',
                ['Mean', 'Median']
            )

    with col3:
        # Ascending/Descending Selectbox
        a_d_select = st.selectbox(
                'Order:',
                ['Ascending', 'Descending']
            )

        # If query: Ascending/Descending
        if a_d_select == 'Ascending':
            a_d = True
        else:
            a_d = False

    with st.container():
        # If query: Mean/Median
        if median_mean_select == 'Mean':
            genre_mean_median = df.groupby('track_genre')[characteristic_select].mean().sort_values(ascending=a_d).reset_index()
        else:
            genre_mean_median = df.groupby('track_genre')[characteristic_select].median().sort_values(ascending=a_d).reset_index()

        # Plot: Feature per Genre ordered by Mean/Median 
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 25))
        sns.barplot(x=characteristic_select, y='track_genre', data=genre_mean_median, order=genre_mean_median['track_genre'])
        ax.set_xlabel(f'{median_mean_select} of {characteristic_select}')
        ax.set_ylabel('Genre')
        ax.set_title(f'{median_mean_select} of {characteristic_select} per Genre')
        ax_top = ax.twiny()
        ax_top.set_xlabel(f'{median_mean_select} of {characteristic_select}')
        ax_top.set_xlim(ax.get_xlim())
        ax_top.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        st.pyplot(fig)