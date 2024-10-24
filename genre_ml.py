import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
import time

def genre_ml_page():
    with st.container():
            # header
            st.header("Music Genre Machine Learning with DecisionTreeClassifier")
                
            col1, col2 = st.columns(2)

            # Dataset Selectbox
            dataset = st.selectbox("Select a Data Set", ['Raw', 'Grouped', 'Filtered'])

            # Creating Dataset base on Selectbox
            if dataset == 'Raw':
                 df = pd.read_csv("train.csv")

            elif dataset == 'Filtered':
                 df = pd.read_csv("train_filtered.csv")

            elif dataset == 'Grouped':
                 df = pd.read_csv("train_grouped.csv")
                 df = df.drop('track_genre', axis= 1)
                 df = df.rename(columns={'genre_category': 'track_genre'})

            # Dropping non numeric features
            df.drop(columns=['Unnamed: 0', 'track_id', 'artists', 'album_name', 'track_name'], inplace=True)
            
            # Reset Index
            df.sample(frac=1).reset_index(drop=True)
            
            # Drop Genre column for Feature Selectbox
            df_drop_genre = df.drop('track_genre', axis= 1)
            
            # Selectbox for choosing features to use for classification
            features = st.multiselect('Wähle die Features für Machine Learning aus:', df_drop_genre.columns)

            # Splitting df into features and label
            X = df[features]
            y = df['track_genre']

            # Selectbox for choosing max tree depth
            depth = st.selectbox("Select Max Depth of Trees", range(5, 31, 5))

            # Button for running the classifier
            if st.button(label="Run Classifier"):

                # train test split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Defining the classifier
                clf = DecisionTreeClassifier(random_state=0, max_depth=depth)

                # Training the model
                clf.fit(X_train, y_train)

                # Predicting labels
                predictions = clf.predict(X_test)

                # Calculating accuracy
                accuracy = accuracy_score(predictions, y_test)
               
                # Printing accuracy
                st.subheader(f"Genauigkeit des Models: {accuracy*100:.2f}%")
                st.markdown("---")

                col1, col2 = st.columns(2)
                
                with col1:
                    # Feature importance Dataframe
                    st.write("Feature Importance")

                    feature_importance_df = pd.DataFrame({
                        'Feature': X.columns,
                        'Importance': [round(importance * 100, 1) for importance in clf.feature_importances_]
                    })

                    feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False).reset_index(drop=True)

                    st.dataframe(feature_importance_df)

                with col2:
                    # Ouput for first 20 samples of Predicted/True Genre Samples
                    st.write("Predicted/True Genre Samples")

                    comparison_df = pd.DataFrame({
                        'True Genre': y_test[:20],
                        'Predicted Genre': predictions[:20]
                    })

                    st.write(comparison_df)
                
                with st.container():
                     
                     st.write(f"{df['track_genre'].nunique()} Genres used:")
                     all_genres = ""
                     for genre in df['track_genre'].unique():
                          all_genres += str(genre) + ", "
                          
                     all_genres_neu = all_genres[:-2] 
                     st.write(all_genres_neu)
                     #st.write(df['track_genre'].unique())