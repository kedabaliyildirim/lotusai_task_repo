import streamlit as st
from gensim import models, similarities
import pandas as pd
@st.cache_data
class TFIDFModel:
    @st.cache_resource
    def train_tfidf_model(words_bow, _document):
        # Train the TF-IDF model
        tfidf = models.TfidfModel(words_bow)
        words_tfidf = [tfidf[doc] for doc in words_bow]

        st.subheader('TF-IDF Model')

        st.write("Testing TF-IDF model with the first 5 documents:")
        # Get the actual term for each ID
        tfidf_terms_df = pd.DataFrame([(_document[id], score) for id, score in words_tfidf[0]], columns=['Term', 'TF-IDF Score'])
        words_tfidf_with_terms = []
        for doc in words_tfidf:
            doc_with_terms = [(_document[id], score) for id, score in doc]
            words_tfidf_with_terms.append(doc_with_terms)

        # Display the first document's terms and their TF-IDF scores
        st.write(tfidf_terms_df.head())


        return words_tfidf, words_tfidf_with_terms


    @st.cache_resource
    def similarities_model(tf_idf_model, _document):
        # Train the similarities model
        index = similarities.SparseMatrixSimilarity(tf_idf_model, num_features=len(_document))
        return index


    @st.cache_data
    def calculate_similarities(tf_idf_model, _document):
        # Train the similarities model
        index = similarities.SparseMatrixSimilarity(tf_idf_model, num_features=len(_document))

        # Calculate similarity of the first document with all other documents
        sims = index[tf_idf_model[0]]
        
        # Create a DataFrame to display similarities
        similarity_df = pd.DataFrame({
            'Document Index': list(range(len(sims))),
            'Similarity Score': sims
        }).sort_values(by='Similarity Score', ascending=False)

        return index, similarity_df

    @st.cache_data
    def info_similarity(similarity_df):
        st.subheader('Similarity Scores')
        st.write(similarity_df.head())
        