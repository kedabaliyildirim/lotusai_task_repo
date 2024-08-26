import streamlit as st
from gensim.models import Word2Vec, FastText, Doc2Vec

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

    @staticmethod
    def test_FastText(model, word):
        try:
            # Test if the word has similar words in the model
            word_vectors = model.wv[word]
            st.write(f"{word} kelimesinin vektörü: {word_vectors[:5]}...")
            similar_words = model.wv.most_similar(word, topn=10)
            if similar_words:
                st.write("FastText model testi başarılı. İşte benzer kelimeler:")
                for similar_word, similarity in similar_words:
                    st.write(f"{similar_word}: {similarity:.4f}")
            else:
                st.write("Verilen kelime için benzer kelime bulunamadı.")
        except KeyError:
            raise KeyError("Verilen kelime modelde bulunamadı.")

    @staticmethod
    @staticmethod
    def test_doc2vec(model, document):
        try:
            # Infer the vector for the given document
            inferred_vector = model.infer_vector(document)
            st.write(f"Dokümanın vektörü: {inferred_vector[:5]}...")
            
            # Find similar documents
            similar_docs = model.dv.most_similar([inferred_vector], topn=10)
            if similar_docs:
                st.write("Doc2Vec model testi başarılı. İşte benzer dokümanlar:")
                for doc_id, similarity in similar_docs:
                    st.write(f"Doküman ID: {doc_id}, Benzerlik: {similarity:.4f}")
            else:
                st.write("Verilen doküman için benzer doküman bulunamadı.")
        except Exception as e:
            st.write(f"Doc2Vec testi sırasında bir hata oluştu: {e}")
