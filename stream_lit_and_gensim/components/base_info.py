import os
import pandas as pd
import streamlit as st
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
        st.subheader('Ön İşlemeden Sonra')
        st.write(pre_processed_df.head())

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
