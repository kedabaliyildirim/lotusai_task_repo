import streamlit as st
from gensim import models, similarities
import pandas as pd
from gensim.matutils import sparse2full
from sklearn.metrics.pairwise import cosine_similarity

class Model:
    
    @staticmethod
    def train_model(model_name, words_bow, _document, **args):
        # Dynamically get the model class from the models module
        model_class = getattr(models, model_name, None)
        
        if model_class is None:
            st.error(f"Model '{model_name}' not found in the models module.")
            return None

        # Train the model using the dynamically retrieved class
        model_instance = model_class(words_bow, **args)
        model = [model_instance[doc] for doc in words_bow]

        st.subheader(f'{model_name} Model')
        st.write(f"{model_name} testi:")

        # Get the actual term for each ID
        model_terms_df = pd.DataFrame([(_document[id], score) for id, score in model[0]], columns=['Term', f'{model_name} Score'])
        words_model_with_terms = []
        for doc in model:
            doc_with_terms = [(_document[id], score) for id, score in doc]
            words_model_with_terms.append(doc_with_terms)

        # Display the first document's terms and their scores
        st.write(model_terms_df.head())
        # Return essential information for saving
        return  model, model_terms_df, words_model_with_terms
        

    @staticmethod
    def similarities_model(model, _document):
        # Train the similarities model
        index = similarities.SparseMatrixSimilarity(model, num_features=len(_document))
        return index

    @staticmethod
    def calculate_similarities(model, _document, index):

        # Calculate similarity of the first document with all other documents
        sims = index[model[0]]
        
        # Create a DataFrame to display similarities
        similarity_df = pd.DataFrame({
            'Document Index': list(range(len(sims))),
            'Similarity Score': sims
        }).sort_values(by='Similarity Score', ascending=False)

        return similarity_df
    
    @staticmethod
    def info_similarity(similarity_df):
        st.subheader('Similarity Scores')
        st.write(similarity_df.head())