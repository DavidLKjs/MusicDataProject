import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def genre_features_page():
    # creating dataframe
    df = pd.read_csv("train.csv")

    with st.container():
        # header
        st.header("Characteristics of Music Genres")
            
        col1, col2 = st.columns(2)

        with col1:
            # creating genre list
            genre_list = df['track_genre'].unique()
            genre_list = np.insert(genre_list, 0, 'all genres')

            # dropdown menu for choosing a genre
            selected_genre = st.selectbox(
                'Choose a Genre:',
                genre_list
            )
            
            # select between all genres/specific genre
            if selected_genre == 'all genres':
                genre_df = df
            else:
                genre_df = df[df['track_genre'] == selected_genre]
            
            # showing selected genre
            st.subheader(selected_genre)

        with col2:
            median_mean_select = st.selectbox(
                'Mean/Median:',
                ['Mean', 'Median']
                )

    with st.container():
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("---")
            # mean/median: song length
            if median_mean_select == 'Mean':
                mean_median_duration_ms = genre_df['duration_ms'].mean()
            else:
                mean_median_duration_ms = genre_df['duration_ms'].median()
            
            mean_median_duration = f"{int(mean_median_duration_ms // 60000)}:{int((mean_median_duration_ms % 60000) // 1000):02d}"
            st.write(f'⏱️Songlength {median_mean_select}: **{mean_median_duration}** Minutes')

            # mean/median: tempo
            if median_mean_select == 'Mean':
                mean_median_tempo = genre_df['tempo'].mean()
            else:
                mean_median_tempo = genre_df['tempo'].median()
            
            st.write(f'🏃‍♂️‍➡️Tempo {median_mean_select}: **{round(mean_median_tempo)}** BPM')

            # mean/median: loudness
            if median_mean_select == 'Mean':
                mean_median_loudness = genre_df['loudness'].mean()
            else:
                mean_median_loudness = genre_df['loudness'].median()
            st.write(f'🔊 Loudness {median_mean_select}: **{round(mean_median_loudness, 1)}** dB')
            st.markdown("---")
            # calculation of sums for minor/major
            minor_count = (genre_df['mode'] == 0).sum()
            major_count = (genre_df['mode'] == 1).sum()
            sizes = [minor_count, major_count]
            labels = ['Minor', 'Major']
            
            # color palette
            colors = ["#66c2a5", "#fc8d62"]

            # plot: minor/major
            plt.figure(figsize=(10, 5.62))
            plt.style.use('dark_background')
            plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            plt.title(f'Proportion of songs in minor and major for {selected_genre}', y=1.08)
            plt.axis('equal')
            plt.legend([], [], frameon=False)
            st.pyplot(plt)

            # plot: tempo
            plt.figure(figsize=(10, 6))
            sns.histplot(data=genre_df, x='tempo', color='palegreen')
            plt.title(f'Distribution of songs by tempo for {selected_genre}')
            plt.xlabel('Tempo in BPM')
            plt.ylabel('Song Count')
            plt.grid(False)
            plt.legend([], [], frameon=False)
            st.pyplot(plt)

        with col2:
            st.markdown("---")
            # mean/median: energy
            if median_mean_select == 'Mean':
                mean_median_energy = genre_df['energy'].mean()
            else:
                mean_median_energy = genre_df['energy'].median() 
            
            st.write(f'🎆Energy {median_mean_select}: **{round(mean_median_energy, 2)}**')

            # mean/median: danceability
            if median_mean_select == 'Mean':
                mean_median_danceability = genre_df['danceability'].mean()
            else:
                mean_median_danceability = genre_df['danceability'].median()
            
            st.write(f'💃Danceability {median_mean_select}: **{round(mean_median_danceability, 2)}**')

            # mean/median acousticness
            if median_mean_select == 'Mean':
                mean_median_acousticness = genre_df['acousticness'].mean()
            else:
                mean_median_acousticness = genre_df['acousticness'].median()
            
            st.write(f'🎻Acousticness {median_mean_select}: **{round(mean_median_acousticness, 2)}**')
            st.markdown("---")
            # plot: count of songs per key + minor/major
            plt.figure(figsize=(10, 6))
            sns.countplot(data=genre_df, x='key', palette=colors,  hue='mode')
            plt.title(f'Song count per key for {selected_genre}')
            plt.xlabel('Key')
            plt.ylabel('Song Count')
            plt.legend(['Minor', 'Major'])
            plt.xticks(range(12), ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'])
            plt.grid(False)
            st.pyplot(plt)

            # plot: energy
            plt.figure(figsize=(10, 1.25))
            sns.boxplot(data=genre_df, x='energy', color='darksalmon')
            plt.xlabel('Energy')
            plt.legend([], [], frameon=False)
            plt.grid(False)
            plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
            st.pyplot(plt)

            # plot: danceability
            plt.figure(figsize=(10, 1.25))
            sns.boxplot(data=genre_df, x='danceability', color='orchid')
            plt.xlabel('Danceability')
            plt.legend([], [], frameon=False)
            plt.grid(False)
            plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
            st.pyplot(plt)

            # plot: accousticness
            plt.figure(figsize=(10, 1.25))
            sns.boxplot(data=genre_df, x='acousticness', color='chocolate')
            plt.xlabel('Acousticness')
            plt.legend([], [], frameon=False)
            plt.grid(False)
            plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
            st.pyplot(plt)