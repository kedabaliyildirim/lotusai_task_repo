from gensim import corpora
import streamlit as st
class BaseOperation:
    
    @st.cache_data
    def tokenize_text(df):
        # Tokenize the text by splitting it into words
        tokenized = df.apply(lambda x: x.split()).tolist()
        return tokenized


    @st.cache_resource
    def create_dictionary(tokenized_text):
        # Create the dictionary from tokenized text
        dictionary = corpora.Dictionary(tokenized_text)
        return dictionary
    
    @st.cache_data
    def token_id(_dictionary):
        return _dictionary.token2id

    @st.cache_data
    def create_bow(tokenized_text, _dictionary):
        # Create the bag of words
        bow = [_dictionary.doc2bow(text) for text in tokenized_text]
        return bow