import streamlit as st
import pandas as pd
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.svm import SVC


current_dir = os.path.dirname(__file__)
current_dir = os.path.dirname(current_dir)
st.title('Star Classification')
st.write('This is a simple star classification app that uses the Star dataset from Kaggle. \n \
         The dataset contains information about stars and their properties. \n \
         The goal is to predict the color of the star based on the other features. \n')

st.write('\n\n\n\n\n Data reading and preperation' )
@st.cache_resource
def load_data():
    return pd.read_csv(current_dir + '\\cleaned_star_data.csv')

df = load_data()

@st.cache_data
def preprocess_data(df):
    # replace empty strings with NA as whitespace is not considered as NA
    df.replace(r'^\s*$', pd.NA, regex=True, inplace=True)
    df.dropna(inplace=True, axis=0)
    df.dropna(inplace=True, axis=1)
    df.columns = df.columns.str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace('/', '_')
    le = preprocessing.LabelEncoder()
    # Encode the labels
    df['Spectral_Class'] = le.fit_transform(df['Spectral_Class'])
    
    return df

df = preprocess_data(df)
st.write(df.head())
st.write(df.shape)

@st.cache_data
def data_split(df):
    X = df.drop('Star_color', axis=1)
    y = df['Star_color']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = data_split(df)

# Add more models
model_list = {
    'SVM': SVC(),
}

# Expand parameter grid for models
param_list = {
    'SVM': {'C': [0.1, 1], 'kernel': ['linear', 'rbf']},
}

# Adjust cross-validation folds
@st.cache_resource
def select_model(X_train, y_train, _model_list, param_list):
    model_scores = {}
    best_models = {}
    
    for name, model in _model_list.items():
        grid_search = GridSearchCV(model, param_list[name], cv=5, n_jobs=-1) 
        grid_search.fit(X_train, y_train)
        st.write(f"Best parameters for {name}: {grid_search.best_params_}")
        model_scores[name] = grid_search.best_score_
        best_models[name] = grid_search.best_estimator_

    return model_scores, best_models

model_scores, best_models = select_model(X_train, y_train, model_list, param_list)

@st.cache_resource
def calculate_best_model(model_scores):
    best_model_name = max(model_scores, key=model_scores.get)
    return best_model_name, best_models[best_model_name]

best_model_name, best_model = calculate_best_model(model_scores)

@st.cache_data
def display_model_scores(model_scores):
    for model, score in model_scores.items():
        st.write(f"{model} has a score of {score}")
    st.write(f"The best model is {best_model_name} with a score of {model_scores[best_model_name]}")

display_model_scores(model_scores)

@st.cache_data
def display_confusion_matrix(best_model, X_test, y_test):
    y_pred = best_model.predict(X_test)
    st.write(confusion_matrix(y_test, y_pred))
    st.write(classification_report(y_test, y_pred))
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

display_confusion_matrix(best_model, X_test, y_test)
