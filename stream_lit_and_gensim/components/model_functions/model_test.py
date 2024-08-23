import streamlit as st
from gensim.models import Word2Vec

class ModelTests:
    @staticmethod
    def test_word2vec(model, word):
        try:
            # Test if the word has similar words in the model
            similar_words = model.wv.most_similar(word, topn=10)
            if similar_words:
                st.write("Word2Vec model testi başarılı. İşte benzer kelimeler:")
                for similar_word, similarity in similar_words:
                    st.write(f"{similar_word}: {similarity:.4f}")
            else:
                st.write("Verilen kelime için benzer kelime bulunamadı.")
        except KeyError:
            raise KeyError("Verilen kelime modelde bulunamadı.")    


