import streamlit as st
import os
import sys
from gensim import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from components.base_info import BaseInformation
from components.base_operations import BaseOperation
from components.model_operant import ModelOperant
from components.word_embeddings import WordEmbedding
from components.model_functions.model_test import ModelTests
import pandas as pd

st.title("Gensim")
st.write('\n\n\n Gensim, Python programlama dili için açık kaynaklı bir doğal dil işleme kütüphanesidir. Gensim, belge benzerliği, konu modelleme ve diğer metin işleme için kullanılır \n\n\n')


st.subheader('\n\n\n Gensim ile Model Eğitimleri')
st.write('\n\n Bu uygulamada, Gensim kütüphanesini kullanarak çeşitli modelleri eğiteceğiz. İlk olarak, veri setini yükleyeceğiz ve ön işleme adımlarını gerçekleştireceğiz. Daha sonra, belgeleri tokenize edeceğiz, bir sözlük oluşturacağız ve kelime çantası (BoW) oluşturacağız. isteğe bağlı olarak tf-idf, lsi, okapi, rp, lda ve hdp modellerini eğiteceğiz ve sonuçları göstereceğiz. \n\n\n')

# Korpus yükleme ve ön işleme
df = BaseInformation.load_corpus()
pre_processed_data = BaseInformation.preprocess_and_tokenize(df[1])
doc2vec_data = BaseInformation.doc2vec_df(df[1])
st.write(pre_processed_data[:5])
BaseInformation.show_info(df, pre_processed_data)



# Tokenizasyon, sözlük oluşturma ve kelime çantası (BoW) oluşturma
document = BaseOperation.create_dictionary(pre_processed_data)
words_bow = BaseOperation.create_bow(pre_processed_data, document)


with st.form(key='model_form'):
    st.write("Hangi modeli eğitmek istersiniz?")
    train_tfidf = st.checkbox("TF-IDF (Term Frequency - Inverse Document Frequenc) Modelini Eğit", value=True)
    train_lsi = st.checkbox("LSI (Latent Semantic Indexing) Modelini Eğit", value=False)
    train_okapi = st.checkbox("OKAPI (BM25) Modelini Eğit", value=False)
    train_rp = st.checkbox("RP (Random Projections) Modelini Eğit", value=False)
    train_lda = st.checkbox("LDA (Latent Dirichlet Allocation) Modelini Eğit", value=False)
    train_hdp = st.checkbox("HDP (Hierarchical Dirichlet Process) Modelini Eğit", value=False)
    submit = st.form_submit_button("Devam Et")

trained_models = pd.DataFrame()
if submit:
    selected_models = {'TFIDF': train_tfidf, 'LSI': train_lsi, 'OKAPI': train_okapi, 'RP': train_rp, 'LDA': train_lda, 'HDP': train_hdp}
    ModelOperant.model_selection(selected_models, words_bow, document)



st.write('\n\n Model eğitimini tamamladık ve benzerlik puanlarını inceledik, sırada modellerin birlikte kullanımı var \n\n\n')

st.subheader('Model Birleştirme')
st.write('\n\n Modelleri birleştirme, farklı model türlerini birleştirerek daha iyi sonuçlar elde etmek için kullanılır. Örneğin, bir belgeyi hem TF-IDF hem de LSI modeliyle temsil edebiliriz ve bu iki modelin benzerlik puanlarını birleştirerek daha iyi bir sonuç elde edebiliriz. Aşağıda yapacaklarımızın listesi var \n\n\n')
st.write('1. LSI, LDA ve HDP ile daha kapsamlı bir topic seçimi gerçekleştirilebilir, LSI boyut azaltma için, \
          LDA veya HDP ise topiclerı keşfetmek için kullanilabilir \n\n')
st.write('2. İlk belge getirme işlemi için BM25 kullanılır, ardından belgeleri gizli topiclera göre iyileştirmek ve sıralamak için LSI veya LDA kullanılabilir. \n\n')
st.write('3. LSI veya LDA uygulamadan önce verinin boyutunu azaltmak için ön işleme aşamasında RP kullanın. \n\n')

st.write('\n Bu adımları uygulamak için aşağıdaki seçenekleri kullanabilirsiniz. \n\n\n')

with st.form(key='model_combination_form'):
    st.write("Hangi modeli birleştirmek istersiniz?")
    lsi_with_lda = st.checkbox('LSI ve LDA (LSI boyut azalmta, LDA topic bulma)', value=False)
    lsi_with_hdp = st.checkbox('LSI ve HDP (LSI boyut azalmta, HDP topic bulma)', value=False)
    okapi_with_lsi = st.checkbox('OKAPI ve LSI (OKAPI belge getirme, LSI topic bulma)', value=False)
    okapi_with_lda = st.checkbox('OKAPI ve LDA (OKAPI belge getirme, LDA topic bulma)', value=False)
    rp_with_lsi = st.checkbox('RP ve LSI (RP boyut azaltma, LSI topic bulma)', value=False)
    rp_with_lda = st.checkbox('RP ve LDA (RP boyut azaltma, LDA topic bulma)', value=False)
    submit = st.form_submit_button("Devam Et")

model_comb_df = pd.DataFrame()
if submit:
    selected_combinations = {'lsi_lda': lsi_with_lda, 'lsi_hdp': lsi_with_hdp, 'bm25_lsi': okapi_with_lsi, 'bm25_lda': okapi_with_lda, 'rp_lsi': rp_with_lsi, 'rp_lda': rp_with_lda}
    model_comb_df = ModelOperant.model_combination(selected_combinations, words_bow, document)


st.write('\n\n Model birleştirme işlemini tamamlandi \n\n\n')

st.subheader('Gömülü Kelime Modelleri')
st.write('\n\n Bu bölümde, kelime gömme modellerini eğiteceğiz. Word2Vec, FastText ve Doc2Vec modellerini eğiteceğiz. Bu modeller, kelime vektörlerini oluşturmak ve benzerlikleri hesaplamak için kullanılır. \n\n\n')
with st.form(key='word_embedding_form'):
    st.write("Hangi kelime gömme modelini eğitmek istersiniz?")
    

    test_phrase = st.text_input("Bir kelime girin (modellerin testi için):")
    st.subheader('Doküman Seçimi')
    st.write('\n\n Lütfen test etmek istediğiniz cümleyi yazın: \n\n')
    test_doc = st.text_input("Bir cümle girin (Doc2Vec modeli için):")
    train_word2vec = st.checkbox("Word2Vec Modelini Eğit", value=True)
    train_fasttext = st.checkbox("FastText Modelini Eğit", value=False)
    train_doc2vec = st.checkbox("Doc2Vec Modelini Eğit", value=False)
    submit = st.form_submit_button("Devam Et")

if submit:
    selected_word_embedding = {'word2vec': train_word2vec, 'fasttext': train_fasttext, 'doc2vec': train_doc2vec}
    word_embedding = WordEmbedding(words_bow=words_bow, document=pre_processed_data, doc2vec_document=doc2vec_data)
    word_embedding.word_embedding_selection(selected_word_embedding, test_phrase, test_doc)
