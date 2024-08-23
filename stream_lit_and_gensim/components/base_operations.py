from collections import defaultdict
from gensim import corpora
import streamlit as st

class BaseOperation:

    @st.cache_resource
    def create_dictionary(tokenized_text):
        diction = corpora.Dictionary(tokenized_text)
        diction.filter_extremes(no_below=20, no_above=0.5)  # Example filter
        return diction

    def token_id(_dictionary):
        return _dictionary.token2id

    @st.cache_data
    def create_bow(tokenized_text, _dictionary):
        bow = [_dictionary.doc2bow(text) for text in tokenized_text]
        return bow