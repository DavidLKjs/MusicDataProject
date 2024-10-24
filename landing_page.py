import streamlit as st
def landing_page():
    with st.container():
        st.image("header_pic.png")
        st.header("Interactive Music Genre Classification and Analysis with Machine Learning")
        st.write("Abschluss Projekt by Sebastian Sitter and David Kujus")
        # Navigation buttons for each page
        st.markdown("### Welcome to the *Music Genre Insights Web App*. Choose one of the following tools in the sidebar to explore exciting aspects of music genres:")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### :cloud: Wordcloud Generator")
            st.markdown("""
            Generate a wordcloud of the most frequent words for any selected genre,
            visualizing the lyrical characteristics of music genres.
            """)
        with col2: 
            st.markdown("#### :microphone: Lyrics Classification")
            st.markdown("""
            Classify song lyrics into their corresponding genre (Rap & Rock) using machine learning (NLP & Naive Bayes).
            Test the accuracy of the model with your own lyrics input!
            """)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### :bar_chart: Genre Features")
            st.markdown("""
            Explore specific features such as tempo, energy, and danceability across various genres.
            Compare how genres differ in their musical characteristics.
            """)
            st.markdown("#### :gear: Genre Classification")
            st.markdown("""
            Classify songs into different genres with the help of several features and tree depths, using the DecissionTreeClassifier.
            """)
        with col2:
            st.markdown("#### :chart_with_upwards_trend: Ranked Features")
            st.markdown("""
            Discover how genres rank based on various features like tempo or popularity.
            Customize the view by sorting data by mean or median values.
            """)
        # Footer
        st.markdown("---")
        st.markdown("""
        This app is designed to provide deep insights into music genres using Natural Language Processing and Machine Learning.
        """)