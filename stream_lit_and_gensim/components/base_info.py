import os
import pandas as pd
import streamlit as st
import nltk
from nltk.corpus import stopwords
import re
from collections import defaultdict
from matplotlib import pyplot as plt
from wordcloud import WordCloud
import matplotlib.image as mpimg
class BaseInformation:
        
    @st.cache_data
    def load_corpus():
        current_dir = os.path.dirname(__file__)
        current_dir = os.path.dirname(current_dir)
        file_directory = f'{current_dir}/eng_news_2023_100K-sentences.txt'
        df = pd.read_csv(file_directory, sep='\t', header=None)
        return df

    def show_info(df, pre_processed_df):
        st.subheader("Ön İşlemeden Önce")
        st.write(df.head())

        # Display the preprocessed text
        st.subheader('Ön İşlemeden Sonra ve tokenize edilmiş metin')
        #list to dataframe
        st.write(pre_processed_df[:5][2])   
        # Display the number of documents and words 

    def preprocess_and_tokenize(df):
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))
        df = df.apply(lambda x: x.lower())  # Lowercase
        df = df.apply(lambda x: re.sub(r'\d+', '', x))  # Remove numbers
        df = df.apply(lambda x: re.sub(r'[^\w\s]', '', x))  # Remove punctuation
        df = df.apply(lambda x: ' '.join(word for word in x.split() if word not in stop_words))  # Remove stopwords

        # Tokenize text
        tokenized = df.apply(lambda x: x.split()).tolist()
        return tokenized
