import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import time

def classification_page():
    # Function to load the data (removing cache for now)
    def load_data():
        try:
            # Attempt to load the dataset
            df = pd.read_parquet('reduced_song_lyrics_with_genres.parquet')
        except Exception as e:
            # Handle cases where the file is not found or cant be loaded
            st.error(f"Error loading data: {e}")
            st.warning("Using dummy data for testing.")
            # Dummy dataset for testing purposes
            data = {
                'cleaned_lyrics': ['Test lyrics rock', 'Test lyrics rap'],
                'tag': ['rock', 'rap']
            }
            df = pd.DataFrame(data)
        return df

    # Load the data
    df = load_data()

    # Ensure there is data to work with
    if df.empty:
        st.error("No data available for classification.")
        return

    # Filter for Rock and Rap genres
    df_filtered = df[df['tag'].isin(['rock', 'rap'])]

    # Prepare the data
    X = df_filtered['cleaned_lyrics']
    y = df_filtered['tag']

    # TFIDF vectorization
    vectorizer = TfidfVectorizer(max_features=5000)
    X_tfidf = vectorizer.fit_transform(X)

    # Split into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

    # Train the Naive Bayes model
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Calculate model accuracy on test data
    accuracy = accuracy_score(y_test, model.predict(X_test))

    # Streamlit user interface
    st.title("Genre Classification: Rock vs. Rap")
    st.subheader("Enter song lyrics to find out if it's more Rock or Rap.")

    # Input field for song lyrics
    text_input = st.text_area("Enter Song Lyrics:", "")

    if st.button("Predict"):
        if text_input:
            # Add a progress bar
            progress_bar = st.progress(0)

            # Simulate the loading process with the progress bar
            for i in range(100):
                time.sleep(0.02)  # Simulate time delay for prediction
                progress_bar.progress(i + 1)

            # Vectorize the input and make a prediction
            input_tfidf = vectorizer.transform([text_input])
            prediction = model.predict(input_tfidf)
            probability = model.predict_proba(input_tfidf)

            # Check the order of classes in the model
            genres = model.classes_

            # Display the results (correctly assigned)
            st.markdown(f"**Predicted Genre:** {prediction[0].capitalize()}")
            st.markdown(f"**Prediction Probability:**")
            st.markdown(f"- **{genres[1].capitalize()}:** {probability[0][1] * 100:.2f}%")
            st.markdown(f"- **{genres[0].capitalize()}:** {probability[0][0] * 100:.2f}%")
        else:
            st.error("Please enter song lyrics.")

    # Display model information
    st.sidebar.header("Model Information")
    st.sidebar.write(f"Model Accuracy: {accuracy * 100:.2f}%")

# Run the classification page function
if __name__ == "__main__":
    classification_page()
