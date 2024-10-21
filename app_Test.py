import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from textblob import TextBlob
import random

# Setze den Dark Mode für die Streamlit App
st.set_page_config(page_title="Genre Text Analysis: Rock vs. Rap", layout="centered", page_icon="🎶")

# CSS für den Dark Mode (angepasst an die Farben des Diagramms)
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
    df = pd.read_parquet('D:/DS/1_Abschlussprojekt/reduced_song_lyrics_with_genres.parquet')
    return df

# Daten laden
df = load_data()

# Punkt 3: Top-Wörter pro Genre und Wortkomplexität analysieren
st.title("Top-Wörter und Textanalyse pro Genre")
st.subheader("Analyse der häufigsten Wörter in Rock vs. Rap")

# Filtern auf die Genres Rock und Rap
df_filtered = df[df['tag'].isin(['rock', 'rap'])]

# TF-IDF Vektorisierung
vectorizer = TfidfVectorizer(max_features=50, stop_words="english")
X_tfidf = vectorizer.fit_transform(df_filtered['cleaned_lyrics'])

# Top-Wörter für jedes Genre visualisieren
genre_selected = st.selectbox("Wähle ein Genre zur Analyse:", ["rock", "rap"])

# Funktion zur Anpassung der WordCloud-Farben
def custom_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    colors = ["#F5A623", "#50E3C2", "#FF6347", "#C64E40", "#77C7FF"]  # Orange, Grün, Rot, Dunkelorange, Hellblau
    return random.choice(colors)  # Zufällige Farbauswahl ohne random_state

# Wörter und ihre TF-IDF-Gewichte extrahieren
if genre_selected:
    genre_df = df_filtered[df_filtered['tag'] == genre_selected]
    tfidf_matrix = vectorizer.fit_transform(genre_df['cleaned_lyrics'])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray().sum(axis=0)

    # Erstelle eine Wordcloud mit den häufigsten Wörtern
    word_freq = dict(zip(feature_names, tfidf_scores))
    wordcloud = WordCloud(width=800, height=400, background_color='black', colormap='warm', 
                          color_func=custom_color_func).generate_from_frequencies(word_freq)

    # Visualisiere die Wordcloud
    st.subheader(f"Wordcloud für {genre_selected.capitalize()}")
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)

# Punkt 4: Vorhersage Rock vs. Rap
st.title("Genre Vorhersage: Rock vs. Rap")
st.subheader("Gib einen Songtext ein, um zu sehen, ob es eher Rock oder Rap ist.")

# Eingabefeld für den Songtext
text_input = st.text_area("Eingabe Songtext:", "")

# Daten vorbereiten
X = df_filtered['cleaned_lyrics']
y = df_filtered['tag']

# TF-IDF Vektorisierung für Vorhersage-Modell
vectorizer_pred = TfidfVectorizer(max_features=5000)
X_tfidf_pred = vectorizer_pred.fit_transform(X)

# Trainings- und Testdaten aufteilen
X_train, X_test, y_train, y_test = train_test_split(X_tfidf_pred, y, test_size=0.2, random_state=42)

# Naive Bayes Modell trainieren
model = MultinomialNB()
model.fit(X_train, y_train)

# Modell-Genauigkeit auf den Testdaten berechnen
accuracy = accuracy_score(y_test, model.predict(X_test))

# Vorhersage durchführen
if st.button("Vorhersagen"):
    if text_input:
        input_tfidf = vectorizer_pred.transform([text_input])
        prediction = model.predict(input_tfidf)
        probability = model.predict_proba(input_tfidf)

        # Extrahiere die Reihenfolge der Genres
        genre_index = list(model.classes_)  # Reihenfolge der Genres (z.B. ['rap', 'rock'])

        # Ergebnis anzeigen
        st.markdown(f"**Vorhergesagtes Genre:** {prediction[0].capitalize()}")
        st.markdown(f"**Vorhersage-Wahrscheinlichkeit:**")
        st.markdown(f"- **{genre_index[0].capitalize()}:** {probability[0][0] * 100:.2f}%")
        st.markdown(f"- **{genre_index[1].capitalize()}:** {probability[0][1] * 100:.2f}%")
    else:
        st.error("Bitte gib einen Songtext ein.")

# Modellinformationen in der Sidebar
st.sidebar.header("Modellinformationen")
st.sidebar.write(f"Modellgenauigkeit: {accuracy * 100:.2f}%")
