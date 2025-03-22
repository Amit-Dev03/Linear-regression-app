import streamlit as st
import pandas as pd
import numpy as np
import pickle

model = pickle.load(open('linear_regression.pkl','rb'))

#let's create web app
st.title("Scikit-learn Linear Regression Model")
tv = st.text_input('Enter TV sales.....')
radio = st.text_input('Enter radio sales...')
newspaper = st.text_input('Enter newspaper sales...')

if st.button('Predict'):
    features = np.array([[tv, radio, newspaper]], dtype=np.float64) # as it gives some error hence we're using float64
    results = model.predict(features).reshape(1,-1)
    #to print use st.write
    st.write("Predicted sale -> ",results[0])
