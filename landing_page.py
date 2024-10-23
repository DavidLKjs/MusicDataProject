import streamlit as st
def landing_page():
    with st.container():
        st.image("header_pic.png")
        st.header("Interactive Music Genre Classification and Analysis with Machine Learning")
        st.subheader("Abschluss Projekt by Sebastian Sitter and David Kujus")
        st.markdown("""
        Welcome to the *Music Genre Insights Web App*. This tool allows you to explore different aspects of music genres
        using text analysis and machine learning. You can classify song lyrics, generate wordclouds, analyze genre-specific features,
        and compare genres based on different musical characteristics.
        """)
        # Navigation buttons for each page
        st.markdown("### Choose a feature to explore:")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### :microphone: Genre Classification")
            st.markdown("""
            Classify song lyrics into their corresponding genre using machine learning.
            Test the accuracy of the model with your own lyrics input!
            """)
            if st.button("Go to Classification"):
                st.experimental_set_query_params(page="classification")
                st.experimental_rerun()
        with col2:
            st.markdown("#### :globe_with_meridians: Wordcloud Generator")
            st.markdown("""
            Generate a wordcloud of the most frequent words for any selected genre,
            visualizing the lyrical characteristics of music genres.
            """)
            if st.button("Go to Wordcloud"):
                st.experimental_set_query_params(page="wordcloud")
                st.experimental_rerun()
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### :bar_chart: Genre Features")
            st.markdown("""
            Explore specific features such as tempo, energy, and danceability across various genres.
            Compare how genres differ in their musical characteristics.
            """)
            if st.button("Go to Genre Features"):
                st.experimental_set_query_params(page="genre_features")
                st.experimental_rerun()
        with col2:
            st.markdown("#### :chart_with_upwards_trend: Ranked Features")
            st.markdown("""
            Discover how genres rank based on various features like tempo or popularity.
            Customize the view by sorting data by mean or median values.
            """)
            if st.button("Go to Ranked Features"):
                st.experimental_set_query_params(page="ranked_features")
                st.experimental_rerun()
        # Footer
        st.markdown("---")
        st.markdown("""
        *Created as part of the Music Genre Classification and Analysis project.*
        This app is designed to provide deep insights into music genres using Natural Language Processing and Machine Learning.
        """)