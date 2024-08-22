import streamlit as st
from gensim import models, similarities
import pandas as pd

class CombinedModel:

    @staticmethod
    def train_model(model_name, words_bow, _document, **args):
        model_class = getattr(models, model_name, None)
        
        if model_class is None:
            st.error(f"Model '{model_name}' not found in the models module.")
            return None, None

        model_instance = model_class(words_bow, **args)

        st.subheader(f'{model_name} Model')
        st.write(f"Testing {model_name} model samples:")

        if model_name == 'LdaModel':
            topics = model_instance.show_topics(formatted=False)
            model_terms_df = pd.DataFrame([(term, weight) for topic in topics for term, weight in topic[1]], columns=['Term', f'{model_name} Score'])
        else:
            model = [model_instance[doc] for doc in words_bow]
            if model:
                model_terms_df = pd.DataFrame([(_document[id], score) for id, score in model[0]], columns=['Term', f'{model_name} Score'])
                st.write(model_terms_df.head())
            else:
                st.write("No terms found, the model output is empty.")
                return None, None

        return model_instance

    @staticmethod
    def similarities_model(model, _document):
        index = similarities.Similarity(None, model, num_features=len(_document))
        return index

    @staticmethod
    def calculate_similarities(model, _document, index):
        sims = index[model[0]]
        
        similarity_df = pd.DataFrame({
            'Document Index': list(range(len(sims))),
            'Similarity Score': sims
        }).sort_values(by='Similarity Score', ascending=False)

        return similarity_df

    @staticmethod
    def info_similarity(similarity_df):
        st.subheader('Similarity Scores')
        st.write(similarity_df.head())

    @staticmethod
    def hybrid_lsi_lda(words_bow, document):
        st.write("LSI + LDA combination applied...")
        
        if 'trained_models' not in st.session_state:
            st.session_state.trained_models = {}

        # Train or retrieve the LSI model
        if 'lsi' not in st.session_state.trained_models:
            lsi_model = CombinedModel.train_model('LsiModel', words_bow, document, num_topics=1)
            st.session_state.trained_models['lsi'] = lsi_model
        else:
            lsi_model = st.session_state.trained_models['lsi']
        
        # Transform documents using the LSI model
        lsi_topics = [lsi_model[doc] for doc in words_bow]

        # Convert LSI topics back to a BoW-like format for LDA
        lsi_topics_bow = []
        for topic_dist in lsi_topics:
            topic_bow = [(int(topic[0]), topic[1]) for topic in topic_dist]
            lsi_topics_bow.append(topic_bow)

        # Train the LDA model using the LSI-transformed topics
        lda_model = CombinedModel.train_model('LdaModel', lsi_topics_bow, document)

        # Calculate similarities for LDA Model
        sim_mod = [lda_model[doc] for doc in lsi_topics_bow]

        lda_index = CombinedModel.similarities_model(lda_model, document)
        lda_similarity_df = CombinedModel.calculate_similarities(lda_model, document, lda_index)
        CombinedModel.info_similarity(lda_similarity_df)

        return lsi_model, lda_model


    @staticmethod
    def hybrid_lsi_hdp(words_bow, document):
        st.write("LSI + HDP combination applied...")

        if 'trained_models' not in st.session_state:
            st.session_state.trained_models = {}
        
        if 'lsi' not in st.session_state.trained_models:
            lsi_model, _ = CombinedModel.train_model('LsiModel', words_bow, document, num_topics=1)
            st.session_state.trained_models['lsi'] = lsi_model
        else:
            lsi_model = st.session_state.trained_models['lsi']

        lsi_topics = [lsi_model[doc] for doc in words_bow]
        hdp_model, _ = CombinedModel.train_model('HdpModel', lsi_topics, document)
        st.session_state.trained_models['hdp'] = hdp_model

        return lsi_model, hdp_model

    @staticmethod
    def hybrid_okapi_lsi(words_bow, document):
        st.write("BM25 + LSI combination applied...")

        if 'trained_models' not in st.session_state:
            st.session_state.trained_models = {}

        if 'okapi' not in st.session_state.trained_models:
            okapi_model, _ = CombinedModel.train_model('OkapiBM25Model', words_bow, document)
            st.session_state.trained_models['okapi'] = okapi_model
        else:
            okapi_model = st.session_state.trained_models['okapi']

        okapi_topics = [okapi_model[doc] for doc in words_bow]
        lsi_model, _ = CombinedModel.train_model('LsiModel', okapi_topics, document, num_topics=1)
        st.session_state.trained_models['lsi'] = lsi_model

        return okapi_model, lsi_model

    @staticmethod
    def hybrid_okapi_lda(words_bow, document):
        st.write("BM25 + LDA combination applied...")

        if 'trained_models' not in st.session_state:
            st.session_state.trained_models = {}

        if 'okapi' not in st.session_state.trained_models:
            okapi_model, _ = CombinedModel.train_model('OkapiBM25Model', words_bow, document)
            st.session_state.trained_models['okapi'] = okapi_model
        else:
            okapi_model = st.session_state.trained_models['okapi']

        # Calculate similarities using Okapi BM25 model
        okapi_index = CombinedModel.similarities_model(okapi_model, document)
        okapi_similarity_df = CombinedModel.calculate_similarities(okapi_model, document, okapi_index)
        CombinedModel.info_similarity(okapi_similarity_df)

        # Train LDA model using Okapi BM25-transformed documents
        okapi_topics = [okapi_model[doc] for doc in words_bow]
        lda_model, _ = CombinedModel.train_model('LdaModel', okapi_topics, document)
        st.session_state.trained_models['lda'] = lda_model

        # Calculate similarities using LDA model
        refined_index = CombinedModel.similarities_model(lda_model, document)
        refined_similarity_df = CombinedModel.calculate_similarities(lda_model, document, refined_index)
        CombinedModel.info_similarity(refined_similarity_df)

        return okapi_model, lda_model

    @staticmethod
    def hybrid_rp_lsi(words_bow, document):
        st.write("RP + LSI combination applied...")

        if 'trained_models' not in st.session_state:
            st.session_state.trained_models = {}

        if 'rp' not in st.session_state.trained_models:
            rp_model, _ = CombinedModel.train_model('RpModel', words_bow, document, num_topics=1)
            st.session_state.trained_models['rp'] = rp_model
        else:
            rp_model = st.session_state.trained_models['rp']

        rp_topics = [rp_model[doc] for doc in words_bow]
        lsi_model, _ = CombinedModel.train_model('LsiModel', rp_topics, document, num_topics=1)
        st.session_state.trained_models['lsi'] = lsi_model

        return rp_model, lsi_model

    @staticmethod
    def hybrid_rp_lda(words_bow, document):
        st.write("RP + LDA combination applied...")

        if 'trained_models' not in st.session_state:
            st.session_state.trained_models = {}

        if 'rp' not in st.session_state.trained_models:
            rp_model, _ = CombinedModel.train_model('RpModel', words_bow, document, num_topics=1)
            st.session_state.trained_models['rp'] = rp_model
        else:
            rp_model = st.session_state.trained_models['rp']

        rp_topics = [rp_model[doc] for doc in words_bow]
        lda_model, _ = CombinedModel.train_model('LdaModel', rp_topics, document)
        st.session_state.trained_models['lda'] = lda_model

        return rp_model, lda_model