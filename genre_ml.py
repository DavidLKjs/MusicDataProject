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
    df = pd.read_csv("train.csv")

    df.drop(columns=['Unnamed: 0', 'track_id', 'artists', 'album_name', 'track_name'], inplace=True)

    df.sample(frac=1).reset_index(drop=True)

    df_drop_genre = df.drop('track_genre', axis= 1)
    with st.container():
            # header
            st.header("Music Genre Machine Learning with DecisionTreeClassifier")
                
            col1, col2 = st.columns(2)


            features = st.multiselect('Wähle die Features für Machine Learning aus:', df_drop_genre.columns)

            X = df[features]
            y = df['track_genre']

            depth = st.selectbox("Select Max Depth of Trees", range(5, 31, 5))

            if st.button(label="Run Classifier"):

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                clf = DecisionTreeClassifier(random_state=0, max_depth=depth)

                clf.fit(X_train, y_train)

                predictions = clf.predict(X_test)

                accuracy = accuracy_score(predictions, y_test)
               

                st.subheader(f"Genauigkeit des Models: {accuracy*100:.2f}%")
                st.markdown("---")

                col1, col2 = st.columns(2)
                
                with col1:
                    #for i,v in enumerate(clf.feature_importances_):
                    #    st.write(f"Feature: {X.columns[i]:25} Importance: {clf.feature_importances_[i]*100:.2f}%")

                    st.write("Feature Importance")

                    feature_importance_df = pd.DataFrame({
                        'Feature': X.columns,
                        'Importance': [round(importance * 100, 1) for importance in clf.feature_importances_]
                    })

                    feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False).reset_index(drop=True)

                    st.dataframe(feature_importance_df)

                with col2:
                    st.write("Predicted/True Genre Sample")

                    comparison_df = pd.DataFrame({
                        'True Genre': y_test[:20],
                        'Predicted Genre': predictions[:20]
                    })

                    st.write(comparison_df)