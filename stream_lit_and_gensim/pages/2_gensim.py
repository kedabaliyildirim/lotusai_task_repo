import streamlit as st

from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.models import KeyedVectors

def train_word2vec_model(corpus):
    model = Word2Vec(corpus, vector_size=100, window=5, min_count=1, workers=4)
    return model

def train_doc2vec_model(corpus):
    tagged_data = [TaggedDocument(words=words, tags=[str(i)]) for i, words in enumerate(corpus)]
    model = Doc2Vec(tagged_data, vector_size=100, window=2, min_count=1, workers=4)
    return model

@st.cache_resource
def load_word2vec_model():
    return KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

def load_doc2vec_model():
    return Doc2Vec.load('doc2vec.model')

corpus = [
    ['I', 'love', 'machine', 'learning'],
    ['I', 'love', 'deep', 'learning'],
    ['I', 'love', 'NLP'],
]

st.write('Training Word2Vec model')
word2vec_model = train_word2vec_model(corpus)
st.write('Training Doc2Vec model')
doc2vec_model = train_doc2vec_model(corpus)

st.write('Loading Word2Vec model')
word2vec_model = load_word2vec_model()
st.write('Loading Doc2Vec model')
doc2vec_model = load_doc2vec_model()

st.write('Word2Vec model similarity')
st.write(word2vec_model.wv.most_similar('machine'))

st.write('Doc2Vec model similarity')
st.write(doc2vec_model.docvecs.most_similar('0'))


