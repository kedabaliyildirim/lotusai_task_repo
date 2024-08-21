import streamlit as st
import os
import sys
from gensim import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from components.base_info import BaseInformation
from components.base_operations import BaseOperation
from components.tf_idf_model import TFIDFModel

# Load and preprocess the corpus
df = BaseInformation.load_corpus()
pre_processed_df = BaseInformation.preprocess_text(df[1])
BaseInformation.show_info(df, pre_processed_df)

# Tokenization, dictionary creation, and bag-of-words (BoW) creation
tokenized_text = BaseOperation.tokenize_text(pre_processed_df)
document = BaseOperation.create_dictionary(tokenized_text)
words_bow = BaseOperation.create_bow(tokenized_text, document)

tokenId = BaseOperation.token_id(document)

st.write('Next step is to train the TF-IDF model and calculate the similarity scores between documents. However, due to the size of the dataset, this process may take a while. Please be patient.')
st.write('To continue, please click the button below.')

if st.button('Continue'):
    # Train the TF-IDF model and calculate similarities
    tf_idf_model, terms_model = TFIDFModel.train_tfidf_model(words_bow, document)
    index = TFIDFModel.similarities_model(tf_idf_model, document)
    index, similarity_df = TFIDFModel.calculate_similarities(tf_idf_model, document)
    TFIDFModel.info_similarity(similarity_df)

    st.write('The TF-IDF model has been trained and the similarity scores have been calculated. You can now view the results below.')

@st.cache_resource
def lsi_model(words_bow, _document):
    # LSI Model
    lsi_model = models.LsiModel(words_bow, id2word=_document, num_topics=2)
    st.write("the number of topics in the LSI model is: ")
    st.write(len(lsi_model.print_topics()))
    st.write("number of topics to be displayed: ")
    st.slider("Select the number of topics to display", 1, len(lsi_model.print_topics()))
    st.write("LSI Model Topics:")
    st.write(lsi_model.print_topics(1))

lsi_model(words_bow, document)    

