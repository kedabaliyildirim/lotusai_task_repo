import streamlit as st
from gensim import models, similarities
import pandas as pd
from gensim.test.utils import common_texts
from gensim.models import Word2Vec, FastText, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import os
import sys
sys.path.append(os.path.abspath(os.path.join('..')))
from components.model_functions.model_test import ModelTests

class WordEmbedding:
    def __init__(self, words_bow, document, **args):
        self.words_bow = words_bow
        self.document = document
        self.args = args

    def train_word2vec(self, model_name):
        try:
            model = Word2Vec.load("word2vec.model")
            st.write(f"{model_name} Modeli başarıyla yüklendi.")
        except FileNotFoundError:
            st.write(f"{model_name} Modeli bulunamadı, yeni bir model eğitiliyor...")
            model = Word2Vec(sentences=self.document, vector_size=100, window=5, min_count=1, workers=4)
            model.save("word2vec.model")
            st.write(f"{model_name} Modeli kaydedildi.")

        if isinstance(self.words_bow, list):
            model.train(self.words_bow, total_examples=len(self.words_bow), epochs=10)
            st.write(f"{model_name} Modeli {len(self.words_bow)} örnek üzerinde eğitildi.")
        else:
            st.write(f"{model_name} Modeli için BoW verisi uygun değil, eğitim başarısız.")

        if 'test_word' in self.args:
            test_word = self.args['test_word']
            try:
                similar_words = model.wv.most_similar(test_word, topn=10)
                st.write(f"{test_word} kelimesine benzer kelimeler:")
                for word, similarity in similar_words:
                    st.write(f"{word}: {similarity:.4f}")
            except KeyError:
                st.write(f"{test_word} kelimesi modelde bulunamadı.")

        return model

    def train_fasttext(self, model_name):
        try:
            model = FastText.load("fasttext.model")
        except FileNotFoundError:
            st.write(f"{model_name} Modeli bulunamadı, yeni bir model eğitiliyor...")
            model = FastText(sentences=self.document, vector_size=100, window=5, min_count=1, workers=4, alpha=0.005)

            print(model.wv.most_similar('technology'))

            model.save("fasttext.model")
            st.write(f"{model_name} Modeli kaydedildi.")

        return model

    def train_doc2vec(self, model_name):
        try:
            model = Doc2Vec.load("doc2vec.model")
            st.write(f"{model_name} Modeli başarıyla yüklendi.")
        except FileNotFoundError:
            st.write(f"{model_name} Modeli bulunamadı, yeni bir model eğitiliyor...")
            model = doc2vec.Doc2Vec(vector_size=100, window=5, min_count=1, workers=4)
            model.build_vocab(self.args['doc2vec_document'])
            model.train(self.args['doc2vec_document'], total_examples=model.corpus_count, epochs=100)
            
            model.save("doc2vec.model")
            st.write(f"{model_name} Modeli kaydedildi.")


        return model
    
    def word_embedding_selection(self, train_list, test_phrase, test_doc):
        if train_list['word2vec']:
            w2v_model = self.train_word2vec("Word2Vec")
            ModelTests.test_word2vec(w2v_model, test_phrase)
        if train_list['fasttext']:
            ft_model = self.train_fasttext("FastText")
            ModelTests.test_FastText(ft_model, test_phrase)
        if train_list['doc2vec']:
            d2v_model = self.train_doc2vec("Doc2Vec")
            test_doc = test_doc.split(" ")
            test_doc = [word for word in test_doc if word != '']
            ModelTests.test_doc2vec(d2v_model, test_doc)
        