import streamlit as st
from PIL import Image
from io import BytesIO
import requests

import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

import katonic
from katonic.ml.client import MLClient

import pickle

response = requests.get(url='https://katonic.ai/favicon.ico')
im = Image.open(BytesIO(response.content))
st.set_page_config(page_title='Bank Loan Default Prediction',
                   page_icon=im,
                   layout='wide',
                   initial_sidebar_state='auto')
st.set_option('deprecation.showPyplotGlobalUse', False)
st.sidebar.image('Logo.png')
st.sidebar.title('Bank Loan Default Prediction')
st.sidebar.write('---')
st.title("Bank Loan Default Prediction.")
st.write('''Bank loan default is a classic use case where ML models can be deployed to predict risky customers and hence minimize losses of the lenders. Financial industry is highly regulated, thus any model deployed or classification of customers basis their behavior, demographics etc. is highly regulated and must be explained to authorities to ensure unbiased operations.

Loans are risky but at the same time it is also a product that generates profits for the institution through differential borrowing/ lending rates.

The ML model should be explainable and be able to balance between risk and profits.''')
data = pd.read_csv("loan_default.csv")
st.write(data.head())

# showing fig1
def countplot():
    fig = plt.figure(figsize=(16, 6))
    sns.countplot(x = data['bad_loan'])
    plt.legend()
    st.pyplot(fig)
countplot()