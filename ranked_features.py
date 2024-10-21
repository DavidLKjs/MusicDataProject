import streamlit as st
st.set_page_config(layout="wide") 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Dataframe erstellen
df = pd.read_csv("train.csv")

# Umwandlung der Dauer von Millisekunden in Minuten
df['duration_min'] = df['duration_ms'] / (1000 * 60)

# Die alte Spalte wird entfernt
df.drop(columns=['duration_ms'], inplace=True)

# Überschrift
st.header("Ranked Characteristics")

col1, col2, col3 = st.columns(3)

with col1:
    # Liste der Characteristics
    characteristics = ['tempo', 'duration_min', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'popularity']
    
    # Characteristic Selectbox
    characteristic_select = st.selectbox(
            'Wähle eine Characteristic:',
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
            'Reihenfolge:',
            ['Ascending', 'Descending']
        )

    # Abfrage: Ascending/Descending
    if a_d_select == 'Ascending':
        a_d = True
    else:
        a_d = False

with st.container():
    # Abfrage Mean/Median
    if median_mean_select == 'Mean':
        genre_mean_median = df.groupby('track_genre')[characteristic_select].mean().sort_values(ascending=a_d).reset_index()
    else:
        genre_mean_median = df.groupby('track_genre')[characteristic_select].median().sort_values(ascending=a_d).reset_index()

    # Plot: Feature pro Genre sortiert nach Mean/Median 
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 25))
    sns.barplot(x=characteristic_select, y='track_genre', data=genre_mean_median, order=genre_mean_median['track_genre'])
    ax.set_xlabel(f'{median_mean_select} von {characteristic_select}')
    ax.set_ylabel('Genre')
    ax.set_title(f'{median_mean_select} von {characteristic_select} pro Genre')
    ax_top = ax.twiny()
    ax_top.set_xlabel(f'{median_mean_select} von {characteristic_select}')
    ax_top.set_xlim(ax.get_xlim())
    ax_top.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    st.pyplot(fig)