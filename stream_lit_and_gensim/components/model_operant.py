import streamlit as st
import os
import sys
from gensim import models
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from components.model_functions.general_model_function import Model
from components.model_functions.combination_models import CombinedModel

class ModelOperant:
    @st.cache_resource
    def model_selection(model, _words_bow, _document):
        st.write("Model seçildi.")
        
        if not any(model.values()):
            st.write("En az bir model seçmelisiniz.")
            return None
        
        trained_model = st.session_state.get('trained_model', {})
        unchanged_models = {key: trained_model[key] for key in model if model[key] and key in trained_model}

        if unchanged_models:
            st.write("Önceden eğitilmiş modeller kullanılıyor. Değiştirmek için sayfayı yenileyin veya farklı bir model seçin.")
            return unchanged_models

        def train_and_calculate(model_name, _words_bow, **args):
            # Train the model
            model_instance, model_terms, words_model_with_terms = Model.train_model(model_name, _words_bow, _document, **args)
            # Calculate similarities
            index = Model.similarities_model(model_instance, _document)
            similarity_df = Model.calculate_similarities(model_instance, _document, index)
            # Display information about similarities
            Model.info_similarity(similarity_df)
            # Store the essential information in session state

        if model.get('TFIDF'):
            st.subheader('TF-IDF Model')
            st.write("TF-IDF modeli eğitiliyor. TF-IDF (Term Frequency-Inverse Document Frequency), kelimelerin belgeler arasındaki önemini hesaplamak için kullanılır. Özellikle bilgi erişim sistemlerinde ve metin madenciliğinde yaygın olarak kullanılır.  \n\n\n")
            st.subheader('Kullanım Alanları:')
            st.write('1. Belgelerdeki kelimelerin önem derecelerini hesaplamak')
            st.write('2. Kelime sıklığına dayalı olarak belgeler arasındaki benzerlikleri belirlemek')
            st.write('3. Belgelerin içeriklerine göre sıralanmasını ve sınıflandırılmasını sağlamak')
            trained_model['tfidf'] = train_and_calculate('TfidfModel', _words_bow, id2word=_document)
            st.write('TF-IDF modeli eğitildi ve benzerlik puanları hesaplandı.')

        if model.get('LSI'):
            st.subheader('LSI Model')
            st.write("LSI modeli eğitiyor, LSI modeli boyut düşürme işlemi yaparak belgeler arasındaki benzerlikleri hesaplamak için kullanılır. gizli konseptleri keşfetmek için kullanılır. \n\n\n")
            st.subheader('Kullanim alanları:')
            st.write('1. Belgeler arasındaki benzerlikleri hesaplamak')
            st.write('2. Belgeler arasındaki gizli konseptleri keşfetmek')
            st.write('3. Belgeler arasındaki gizli konseptlere göre belgeleri gruplandırmak')
            trained_model['lsi'] = train_and_calculate('LsiModel', _words_bow, id2word=_document, num_topics=10)
            st.write('LSI modeli eğitildi ve benzerlik puanları hesaplandı.')
            
        if model.get('OKAPI'):
            st.subheader('OKAPI Model')
            st.write("OKAPI bm25 modeli eğitiliyor, OKAPI modeli belgeler arasındaki benzerlikleri hesaplamak için kullanılır. Genellikle arama motorları tarafından kullanılır.  \n\n\n")
            st.subheader('Kullanim alanları:')
            st.write('1. Belgeler arasındaki benzerlikleri hesaplamak')
            st.write('2. Belgeler arasındaki benzerlik puanlarını hesaplamak')
            st.write('3. Belgeler arasındaki benzerlik puanlarına göre belgeleri sıralamak')
            trained_model['OKAPI'] = train_and_calculate('OkapiBM25Model', _words_bow)
            st.write('OKAPI modeli eğitildi ve benzerlik puanları hesaplandı.')

        if model.get('RP'):
            st.subheader('RP Model')
            st.write("Random Projections modeli eğitiliyor, RP modeli begenin mertebesini düşürmek için kullanılır. Yüksek mertebeli veriyi düşük mertebeli uzaya yansıtarak çalışır  \n\n\n")
            st.subheader('Kullanim alanları:')
            st.write('1. Büyük veri setlerinde boyut azaltma işlemlerini verinin yapısını koruyarak etkili bir şekilde gerçekleştirmek \n\n' )
            st.write('2. Benzerlik puanlarını hesaplamak \n\n')
            st.write('3. Anlamlı analizler sağlamkla beraber verilerin rastgele yansıtılması veri madenciliği yaparken verinin anonim halinin korunmasına yardımcı olur. \n\n')
            trained_model['RP'] = train_and_calculate('RpModel', _words_bow, num_topics=10)
            st.write('RP modeli eğitildi ve benzerlik puanları hesaplandı.')

        if model.get('LDA'):
            st.subheader('LDA Model')
            st.write("LDA modeli eğitiliyor, LDA modeli belgeler koleksiyonundaki gizli konuları keşfetmek için kullanılan bir üretici olasılıksal modeldir. \n\n\n")
            st.subheader('Kullanim alanları:')
            st.write('1. Büyük metin koleksiyonlarında gizli konuları belirlemek \n\n')
            st.write('2. Belgeleri konulara göre sınıflandırmak ve konu dağılımını özellik olarak kullanmak \n\n')
            st.write('3. İçerik önerileri sunmak, konu dağılımı benzerliğine göre belgeler veya makaleler önermek \n\n')
            st.write('4. Veri keşfi yapmak, bir veri setindeki temaları veya konuları keşfetmek ve görselleştirmek \n\n')
            trained_model['LDA'] = train_and_calculate('LdaModel', _words_bow, id2word=_document, num_topics=10)
            st.write('LDA modeli eğitildi ve benzerlik puanları hesaplandı.')

        if model.get('HDP'):
            st.subheader('HDP Model')
            st.write("HDP modeli eğitiliyor, HDP modeli, konu sayısının veriden belirlenmesine izin veren, bu yüzden parametrik olmayan, LDA modelinin bir uzantısıdır. \n\n\n")
            st.subheader('Kullanim alanları:')
            st.write('1. Dinamik konu modelleme, konu sayısını önceden belirtmeye gerek kalmadan bir veri setindeki konuları modellemek \n\n')
            st.write('2. Ölçeklenebilir konu modelleme, konu sayısının bilinmediği veya zamanla değişen büyük ve karmaşık veri setlerine konu modellemesi uygulamak \n\n')
            st.write('3. Konuların hiyerarşik kümelenmesi, karmaşık veri setlerini anlamak için konuları ve alt konuları hiyerarşik bir şekilde keşfetmek \n\n')
            trained_model['HDP'] = train_and_calculate('HdpModel', _words_bow, id2word=_document)
            st.write('HDP modeli eğitildi ve benzerlik puanları hesaplandı.')

        st.session_state.trained_model = trained_model
        return trained_model

    def model_combination(selected_combinations, words_bow, document):
        st.write("Model kombinasyonu seçildi.")
        combined_results = {}
        trained_models = {}
        model_combinations_df = pd.DataFrame()
        # Hybrid Approach 1: Use LSI for dimensionality reduction, followed by LDA or HDP for topic discovery
        if selected_combinations.get('lsi_lda'):
            st.subheader('LSI + LDA Model')
            st.write("Bu model, LSI ile boyut düşürme ve LDA ile konu keşfi yapar. LSI, veriyi basitleştirir ve gürültüyü azaltır; LDA, bu basitleştirilmiş veride daha derin konular bulur.")
            st.write('Kullanım Alanları: \n1. Daha yüksek doğrulukta konu modelleme \n2. Gürültülü verileri temizleyip LDA\'ya daha net bilgi sağlama \n3. Büyük veri kümelerinde verimli konu keşfi.')
            model_combinations_df['lsi_lda'] = CombinedModel.hybrid_lsi_lda(words_bow, document)

        if selected_combinations.get('lsi_hdp'):
            st.subheader('LSI + HDP Model')
            st.write("LSI ile boyut düşürme, ardından HDP ile konu keşfi yapılır. HDP, konu sayısını otomatik belirler ve dinamik veri setlerinde kullanışlıdır.")
            st.write('Kullanım Alanları: \n1. Dinamik veri setlerinde konu modelleme \n2. Zamanla değişen temaların keşfi \n3. Belirsiz konu sayısı olan veri kümelerinde modelleme.')
            model_combinations_df['lsi_hdp'] = CombinedModel.hybrid_lsi_hdp(words_bow, document)

        if selected_combinations.get('bm25_lsi'):
            st.subheader('BM25 + LSI Model')
            st.write("BM25, ilk olarak en alakalı belgeleri getirir, ardından LSI bu belgeleri rafine ederek gizli yapıları keşfeder.")
            st.write('Kullanım Alanları: \n1. Arama motorlarında belge sıralaması \n2. İlk sıralanan belgeleri rafine etme \n3. Büyük metin koleksiyonlarında derin analiz.')
            model_combinations_df['bm25_lsi'] = CombinedModel.hybrid_okapi_lsi(words_bow, document)

        if selected_combinations.get('bm25_lda'):
            st.subheader('BM25 + LDA Model')
            st.write("BM25, ilk olarak alakalı belgeleri bulur, ardından LDA bu belgelerdeki konuları keşfeder.")
            st.write('Kullanım Alanları: \n1. Gizli konuları keşfetme \n2. Belge sınıflandırma ve öneri sistemleri \n3. İlgili belgeleri bulduktan sonra içerik analizi.')
            model_combinations_df['bm25_lda'] = CombinedModel.hybrid_okapi_lda(words_bow, document)

        if selected_combinations.get('rp_lsi'):
            st.subheader('RP + LSI Model')
            st.write("RP, veriyi düşük boyutlu bir uzaya yansıtarak yapılandırır, ardından LSI bu veriyi daha da basitleştirir.")
            st.write('Kullanım Alanları: \n1. Büyük veri kümelerinde boyut azaltma \n2. Veri yapısını koruyarak gürültüyü azaltma \n3. Yüksek boyutlu verilerde daha anlamlı analizler.')
            model_combinations_df['rp_lsi'] = CombinedModel.hybrid_rp_lsi(words_bow, document)

        if selected_combinations.get('rp_lda'):
            st.subheader('RP + LDA Model')
            st.write("RP ile boyut düşürme yapılır, ardından LDA gizli konuları keşfeder.")
            st.write('Kullanım Alanları: \n1. Yüksek boyutlu veri setlerinde verimli konu modelleme \n2. Anonimlik ve veri güvenliğini koruyarak analiz yapma \n3. Büyük veri kümelerinde daha etkin modelleme.')
            model_combinations_df['rp_lda'] = CombinedModel.hybrid_rp_lda(words_bow, document)


        return model_combinations_df
