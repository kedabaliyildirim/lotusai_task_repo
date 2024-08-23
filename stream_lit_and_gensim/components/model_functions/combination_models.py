import streamlit as st
from gensim import models, similarities
import pandas as pd

class CombinedModel:

    @staticmethod
    def train_model(model_name, words_bow, _document, **args):
        # Dynamically get the model class from the models module
        model_class = getattr(models, model_name, None)
        
        if model_class is None:
            st.error(f"Model '{model_name}' not found in the models module.")
            return None

        # Train the model using the dynamically retrieved class
        model_instance = model_class(words_bow, **args)

        st.subheader(f'{model_name} Model')
        st.write(f"Testing {model_name} model samples:")

        if model_name == 'LdaModel':
            # Extract topics and their distributions for LDA
            topics = model_instance.show_topics(formatted=False)
            model_terms_df = pd.DataFrame([(term, weight) for topic in topics for term, weight in topic[1]], columns=['Term', f'{model_name} Score'])
        else:
            # Extract the actual terms and their scores for other models like LSI and HDP
            model = [model_instance[doc] for doc in words_bow]
            if model:
                model_terms_df = pd.DataFrame([(_document[id], score) for id, score in model[0]], columns=['Term', f'{model_name} Score'])
                st.write(model_terms_df.head())
            else:
                st.write("No terms found, the model output is empty.")
                return None

        return model_instance

    @staticmethod
    def hybrid_lsi_lda(words_bow, document):
        st.write("LSI + LDA combination applied...")
        
        if 'trained_models' not in st.session_state:
            st.session_state.trained_models = {}

        if 'lsi' not in st.session_state.trained_models:
            lsi_model = CombinedModel.train_model('LsiModel', words_bow, document, id2word=document, num_topics=10)
            st.session_state.trained_models['lsi'] = lsi_model
        else:
            lsi_model = st.session_state.trained_models['lsi']
            
        lsi_topics = [lsi_model[doc] for doc in words_bow]
        
        if 'lda' not in st.session_state.trained_models:
            lda_model = CombinedModel.train_model('LdaModel', lsi_topics, document, id2word=document, num_topics=10)
            st.session_state.trained_models['lda'] = lda_model
        else:
            lda_model = st.session_state.trained_models['lda']
        
        lda_topics = lda_model.show_topics(formatted=False)
        st.write(lda_topics)

        return lda_model

    @staticmethod
    def hybrid_lsi_hdp(words_bow, document):
        st.write("LSI + HDP combination applied...")

        if 'trained_models' not in st.session_state:
            st.session_state.trained_models = {}

        if 'lsi' not in st.session_state.trained_models:
            lsi_model = CombinedModel.train_model('LsiModel', words_bow, document, id2word=document, num_topics=10)
            st.session_state.trained_models['lsi'] = lsi_model
        else:
            lsi_model = st.session_state.trained_models['lsi']

        lsi_topics = [lsi_model[doc] for doc in words_bow]
        
        if 'hdp' not in st.session_state.trained_models:
            hdp_model = CombinedModel.train_model('HdpModel', lsi_topics, document, id2word=document)
            st.session_state.trained_models['hdp'] = hdp_model
        else:
            hdp_model = st.session_state.trained_models['hdp']
        
        hdp_topics = hdp_model.show_topics(formatted=False)
        st.write(hdp_topics)

        return hdp_model

    @staticmethod
    def hybrid_okapi_lsi(words_bow, document):
        st.write("BM25 + LSI combination applied...")

        if 'OkapiBM25Model' not in st.session_state.trained_models:
            okapi_model = CombinedModel.train_model('OkapiBM25Model', words_bow, document)
            st.session_state.trained_models['OkapiBM25Model'] = okapi_model
        else:
            okapi_model = st.session_state.trained_models['OkapiBM25Model']

        if 'LsiModel' not in st.session_state.trained_models:
            lsi_model = CombinedModel.train_model('LsiModel', words_bow, document, id2word=document, num_topics=10)
            st.session_state.trained_models['LsiModel'] = lsi_model
        else:
            lsi_model = st.session_state.trained_models['LsiModel']

        okapi_results = okapi_model[words_bow]
        lsi_results = lsi_model[words_bow]

        if isinstance(okapi_results, list) and isinstance(lsi_results, list):
            if isinstance(okapi_results[0], tuple) and isinstance(lsi_results[0], tuple):
                okapi_scores = [score for _, score in okapi_results]
                lsi_scores = [score for _, score in lsi_results]
            else:
                okapi_scores = okapi_results
                lsi_scores = lsi_results

            combined_results = [(okapi_score + lsi_score) / 2 for okapi_score, lsi_score in zip(okapi_scores, lsi_scores)]
        else:
            st.write("Unexpected result format from models")
            combined_results = []

        st.write(combined_results)
        return combined_results

    @staticmethod
    def hybrid_okapi_lda(words_bow, document):
        st.write("BM25 + LDA combination applied...")

        if 'OkapiBM25Model' not in st.session_state.trained_models:
            okapi_model = CombinedModel.train_model('OkapiBM25Model', words_bow, document)
            st.session_state.trained_models['OkapiBM25Model'] = okapi_model
        else:
            okapi_model = st.session_state.trained_models['OkapiBM25Model']

        okapi_results = okapi_model[words_bow]

        if 'LdaModel' not in st.session_state.trained_models:
            lda_model = CombinedModel.train_model('LdaModel', okapi_results, document, id2word=document, num_topics=10)
            st.session_state.trained_models['LdaModel'] = lda_model
        else:
            lda_model = st.session_state.trained_models['LdaModel']

        lda_topics = lda_model.show_topics(formatted=False)
        st.write(lda_topics)

        return lda_model

    @staticmethod
    def hybrid_rp_lsi(words_bow, document):
        st.write("RP + LSI combination applied...")

        if 'RpModel' not in st.session_state.trained_models:
            rp_model = CombinedModel.train_model('RpModel', words_bow, document, num_topics=10)
            st.session_state.trained_models['RpModel'] = rp_model
        else:
            rp_model = st.session_state.trained_models['RpModel']

        rp_topics = [rp_model[doc] for doc in words_bow]
        
        if 'LsiModel' not in st.session_state.trained_models:
            lsi_model = CombinedModel.train_model('LsiModel', rp_topics, document, num_topics=10)
            st.session_state.trained_models['LsiModel'] = lsi_model
        else:
            lsi_model = st.session_state.trained_models['LsiModel']

        lsi_topics = lsi_model.show_topics(formatted=False)
        st.write(lsi_topics)

        return lsi_model

    @staticmethod
    def hybrid_rp_lda(words_bow, document):
        st.write("RP + LDA combination applied...")

        if 'RpModel' not in st.session_state.trained_models:
            rp_model = CombinedModel.train_model('RpModel', words_bow, document, num_topics=10)
            st.session_state.trained_models['RpModel'] = rp_model
        else:
            rp_model = st.session_state.trained_models['RpModel']

        rp_topics = [rp_model[doc] for doc in words_bow]
        
        if 'LdaModel' not in st.session_state.trained_models:
            lda_model = CombinedModel.train_model('LdaModel', rp_topics, document)
            st.session_state.trained_models['LdaModel'] = lda_model
        else:
            lda_model = st.session_state.trained_models['LdaModel']

        lda_topics = lda_model.show_topics(formatted=False)
        st.write(lda_topics)

        return lda_model
