import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Setze den Dark Mode für die Streamlit App
st.set_page_config(page_title="Genre Classification: Rock vs. Rap", layout="centered", page_icon="🎶")

# CSS für den Dark Mode (optional zur weiteren Anpassung)
st.markdown("""
    <style>
        .main {
            background-color: #0E1117;
            color: #FFFFFF;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
        }
        h1, h2, h3, h4 {
            color: #F1F1F1;
        }
        .stTextInput>div>input {
            background-color: #1E1E1E;
            color: white;
        }
        .stTextArea>div>textarea {
            background-color: #1E1E1E;
            color: white;
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

# Filtern auf die Genres Rock und Rap
df_filtered = df[df['tag'].isin(['rock', 'rap'])]

# Daten vorbereiten
X = df_filtered['cleaned_lyrics']
y = df_filtered['tag']

# TF-IDF Vektorisierung
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(X)

# Trainings- und Testdaten aufteilen
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Naive Bayes Modell trainieren
model = MultinomialNB()
model.fit(X_train, y_train)

# Modell-Genauigkeit auf den Testdaten berechnen
accuracy = accuracy_score(y_test, model.predict(X_test))

# Streamlit Benutzeroberfläche
st.title("Genre Classification: Rock vs. Rap")
st.subheader("Gib einen Songtext ein, um herauszufinden, ob es eher Rock oder Rap ist.")

# Eingabefeld für den Songtext
text_input = st.text_area("Eingabe Songtext:", "")

if st.button("Vorhersagen"):
    if text_input:
        # Eingabe vektorisieren und Vorhersage treffen
        input_tfidf = vectorizer.transform([text_input])
        prediction = model.predict(input_tfidf)
        probability = model.predict_proba(input_tfidf)

        # Überprüfe die Reihenfolge der Klassen im Modell
        genres = model.classes_  # Gibt ['rap', 'rock'] zurück

        # Ergebnisse anzeigen (korrekt zuordnen)
        st.markdown(f"**Vorhergesagtes Genre:** {prediction[0].capitalize()}")
        st.markdown(f"**Vorhersage-Wahrscheinlichkeit:**")
        st.markdown(f"- **{genres[1].capitalize()}:** {probability[0][1] * 100:.2f}%")
        st.markdown(f"- **{genres[0].capitalize()}:** {probability[0][0] * 100:.2f}%")
    else:
        st.error("Bitte gib einen Songtext ein.")

# Modellinformationen anzeigen
st.sidebar.header("Modellinformationen")
st.sidebar.write(f"Modellgenauigkeit: {accuracy * 100:.2f}%")
