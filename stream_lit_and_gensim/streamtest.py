import streamlit as st
import pandas as pd
import os 

current_dir = os.path.dirname(__file__)
star_df = pd.read_csv(current_dir+'/cleaned_star_data.csv')
star_df.dropna(inplace=True)


st.write(star_df.head())
st.write(star_df.columns)
st.write(star_df.style.highlight_max(axis=0))

st.line_chart(star_df['Temperature (K)'])

@st.cache_data
def map_func(df):
    df['LON'] = df['long'].astype(float)
    st.map(df)

credt_df = pd.read_csv(current_dir+'/credit_card_purchases.csv')
map_func(credt_df)


x = st.slider('Select a value')
st.write(x, 'squared is', x * x)

st.text_input('Enter some text', key='my_slider')
st.session_state.my_slider

if st.checkbox('Show dataframe'):
    chart_data = star_df[['Temperature (K)', 'Luminosity(L/Lo)', 'Radius(R/Ro)']]
    st.line_chart(chart_data)


option = st.selectbox(
    'Which column should be used as the key column',
     star_df.columns)

st.line_chart(star_df[option])
