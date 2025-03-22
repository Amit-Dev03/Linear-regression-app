import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pickle

#Loading the dataset
data = pd.read_csv('6 advertising.csv')

# Loading the trained model
model = pickle.load(open('linear_regression.pkl','rb')) #rb -> read binary

#let's create web app
st.title("Scikit-learn Linear Regression Model")

#User Input 
tv = st.text_input('Enter TV sales.....')
radio = st.text_input('Enter radio sales...')
newspaper = st.text_input('Enter newspaper sales...')

if st.button('Predict'):
     # Make sure the inputs are not empty 
     if tv and radio and newspaper:
          
        features = np.array([[tv, radio, newspaper]], dtype=np.float64) # as it gives some error hence we're using float64
        predicted_sale = model.predict(features).reshape(1,-1)
        #to print use st.write
        st.write("Predicted sale -> ",predicted_sale[0])

        #Adding the user's input as a new data point ( as seaborn need dataframe )
        new_point = pd.DataFrame({
            'TV' : [float(tv)],
            'Radio' : [float(radio)],
            'Newspaper' : [float(newspaper)],
            'Sales' : [float(predicted_sale)]
        })

        #Visualization
        #Plot 1 : Sales vs TV
        st.subheader('Sales vs Tv spend Scatter Plot')
        fig1, ax = plt.subplots()
        sns.scatterplot(data=data, x='TV', y='Sales', ax = ax, label='Dataset')
        sns.scatterplot(data=new_point, x='TV', y='Sales', ax=ax, color='red', s=100, label='Your Prediction')
        ax.legend()
        st.pyplot(fig1)

        #Plot 2: Sales vs Radio
        st.subheader('Sales vs Radio spend Scatter Plot')
        fig2, ax = plt.subplots()
        sns.scatterplot(data=data, x='Radio', y='Sales', ax = ax, label='Dataset')
        sns.scatterplot(data=new_point, x='Radio', y='Sales', ax=ax, color='red', s=100, label='Your Prediction')
        ax.legend()
        st.pyplot(fig2)
        #Plot 3 : Sales vs Newspaper
        st.subheader('Sales vs Newspaper spend Scatter Plot')
        fig3, ax = plt.subplots()
        sns.scatterplot(data=data, x='Newspaper', y='Sales', ax = ax, label='Dataset')
        sns.scatterplot(data=new_point, x='Newspaper', y='Sales', ax=ax, color='red', s=100, label='Your Prediction')
        ax.legend()
        st.pyplot(fig3)

     else:
        print('Please enter all 3 inputs (TV, Radio, Newspaper)')