import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def genre_features_page():
    # Dataframe erstellen
    df = pd.read_csv("train.csv")

    with st.container():
        # Überschrift
        st.header("Characteristics of Music Genres")
            
        col1, col2 = st.columns(2)

        with col1:
            # Liste der Genres erstellen
            genre_list = df['track_genre'].unique()
            genre_list = np.insert(genre_list, 0, 'all genres')

            # Dropdown Menü für die Auswahl des Genres
            selected_genre = st.selectbox(
                'Wähle ein Genre:',
                genre_list
            )
            
            # Unterscheidung von allen Genres oder ausgewähltem Genre
            if selected_genre == 'all genres':
                genre_df = df
            else:
                genre_df = df[df['track_genre'] == selected_genre]
            
            # Ausgewähltes Genre anzeigen
            st.subheader(selected_genre)
        
        with col2:
            median_mean_select = st.selectbox(
                'Mean/Median:',
                ['Mean', 'Median']
                )

    with st.container():
        col1, col2 = st.columns(2)

        with col1:
            # Mean/Median: Songlänge
            if median_mean_select == 'Mean':
                mean_median_duration_ms = genre_df['duration_ms'].mean()
            else:
                mean_median_duration_ms = genre_df['duration_ms'].median()
            
            mean_median_duration = f"{int(mean_median_duration_ms // 60000)}:{int((mean_median_duration_ms % 60000) // 1000):02d}"
            st.write(f'⏱️Durchschnittliche Songlänge: **{mean_median_duration}** Minuten')

            # Mean/Median: Tempo
            if median_mean_select == 'Mean':
                mean_median_tempo = genre_df['tempo'].mean()
            else:
                mean_median_tempo = genre_df['tempo'].median()
            
            st.write(f'🏃‍♂️‍➡️Durchschnittliches Tempo: **{round(mean_median_tempo)}** BPM')

            # Mean/Median: Lautstärke
            if median_mean_select == 'Mean':
                mean_median_loudness = genre_df['loudness'].mean()
            else:
                mean_median_loudness = genre_df['loudness'].median()
            st.write(f'🔊Durchschnittliche Lautstärke: **{round(mean_median_loudness, 1)}** dB')
            
            # Brechnung der Anzahl von Moll und Dur
            moll_count = (genre_df['mode'] == 0).sum()
            dur_count = (genre_df['mode'] == 1).sum()
            sizes = [moll_count, dur_count]
            labels = ['Moll', 'Dur']
            
            # Farbpalette
            colors = ["#66c2a5", "#fc8d62"]

            # Plot: Moll/Dur
            plt.figure(figsize=(10, 5.62))
            plt.style.use('dark_background')
            plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            plt.title(f'Anteil der Songs in Moll und Dur für {selected_genre}', y=1.08)
            plt.axis('equal')
            plt.legend([], [], frameon=False)
            st.pyplot(plt)

            # Plot: Tempo
            plt.figure(figsize=(10, 6))
            sns.histplot(data=genre_df, x='tempo', color='palegreen')
            plt.title(f'Verteilung der Songs nach Tempo für {selected_genre}')
            plt.xlabel('Tempo in BPM')
            plt.ylabel('Anzahl der Songs')
            plt.grid(False)
            plt.legend([], [], frameon=False)
            st.pyplot(plt)

        with col2:
            # Mean/Median: Energy
            if median_mean_select == 'Mean':
                mean_median_energy = genre_df['energy'].mean()
            else:
                mean_median_energy = genre_df['energy'].median() 
            
            st.write(f'🎆Durchschnittliche Energy: **{round(mean_median_energy, 2)}**')

            # Mean/Median: Danceability
            if median_mean_select == 'Mean':
                mean_median_danceability = genre_df['danceability'].mean()
            else:
                mean_median_danceability = genre_df['danceability'].median()
            
            st.write(f'💃Durchschnittliche Danceability: **{round(mean_median_danceability, 2)}**')

            # Mean/Median Acousticness
            if median_mean_select == 'Mean':
                mean_median_acousticness = genre_df['acousticness'].mean()
            else:
                mean_median_acousticness = genre_df['acousticness'].median()
            
            st.write(f'🎻Durchschnittliche Acousticness: **{round(mean_median_acousticness, 2)}**')
            
            # Plot: Anzahl Songs pro Tonart + Moll/Dur
            plt.figure(figsize=(10, 6))
            sns.countplot(data=genre_df, x='key', palette=colors,  hue='mode')
            plt.title(f'Anzahl der Songs pro Tonart für {selected_genre}')
            plt.xlabel('Tonart')
            plt.ylabel('Anzahl der Songs')
            plt.legend(['Moll', 'Dur'])
            plt.xticks(range(12), ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'])
            plt.grid(False)
            st.pyplot(plt)

            # Plot: Energy
            plt.figure(figsize=(10, 1.25))
            sns.boxplot(data=genre_df, x='energy', color='darksalmon')
            plt.xlabel('Energy')
            plt.legend([], [], frameon=False)
            plt.grid(False)
            plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
            st.pyplot(plt)

            # Plot: Danceability
            plt.figure(figsize=(10, 1.25))
            sns.boxplot(data=genre_df, x='danceability', color='orchid')
            plt.xlabel('Danceability')
            plt.legend([], [], frameon=False)
            plt.grid(False)
            plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
            st.pyplot(plt)

            # Plot: Accousticness
            plt.figure(figsize=(10, 1.25))
            sns.boxplot(data=genre_df, x='acousticness', color='chocolate')
            plt.xlabel('Acousticness')
            plt.legend([], [], frameon=False)
            plt.grid(False)
            plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
            st.pyplot(plt)