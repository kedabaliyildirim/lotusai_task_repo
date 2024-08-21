import streamlit as st
import pandas as pd
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim import corpora
from nltk.corpus import stopwords
import os
import re
from collections import defaultdict

# Streamlit caching for loading corpus
@st.cache_resource
def load_corpus():
    current_dir = os.path.dirname(__file__)
    current_dir = os.path.dirname(current_dir)
    file_directory = f'{current_dir}/eng_news_2023_100K-sentences.txt'
    df = pd.read_csv(file_directory, sep='\t', header=None)
    return df

# Load the corpus
df = load_corpus()

st.subheader("Ön İşlemeden Önce")
st.write(df.head())

def preprocess_text(text_series):
    stop_words = set(stopwords.words('english'))
    text_series = text_series.astype(str)
    
    # Convert to lowercase
    text_series = text_series.apply(lambda x: x.lower())
    
    # Remove numbers
    text_series = text_series.apply(lambda x: re.sub(r'\d+', '', x))
    
    # Remove punctuation
    text_series = text_series.apply(lambda x: re.sub(r'[^\w\s]', '', x))
    
    # Remove stopwords
    text_series = text_series.apply(lambda x: ' '.join(word for word in x.split() if word not in stop_words))

    # Remove words that appear only once
    frequency = defaultdict(int)
    for text in text_series:
        for token in text.split():
            frequency[token] += 1

    text_series = text_series.apply(lambda x: ' '.join(token for token in x.split() if frequency[token] > 1))

    return text_series

# Preprocess the text in the dataframe
df[1] = preprocess_text(df[1])

# Display the preprocessed text
st.subheader('Ön İşlemeden Sonra')
st.write(df.head())

@st.cache_data
def tokenize_text(df):
    # Tokenize the text by splitting it into words
    tokenized = df[1].apply(lambda x: x.split()).tolist()
    return tokenized

tokenized_text = tokenize_text(df)

@st.cache_resource
def create_dictionary(tokenized_text):
    # Create the dictionary from tokenized text
    dictionary = corpora.Dictionary(tokenized_text)
    return dictionary

document = create_dictionary(tokenized_text)

def get_token_id(document):
    token = document.token2id
    return token

token_id = get_token_id(document)

