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
from katonic.fs import FeatureStore

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
st.sidebar.subheader("Inference")
infer_type = st.sidebar.radio("Select Inference type?",("By Customer-Id","By Customer data"))

if infer_type == "By Customer-Id":
    customer_id = st.sidebar.text_input("Enter the customer id?")

    if st.sidebar.button("Predict"):
        fs = FeatureStore(
            user_name = "vinaynamani", # user name who are initializing the feature store.
            project_name = "default_loan_prediction", # name of the project
            description = "Project which will predict bad loans" # Description for the Project (Optional).
        )
        cols = ['annual_inc', 'short_emp', 'emp_length_num',
       'dti', 'last_delinq_none',
       'od_ratio', 'grade_A', 'grade_B', 'grade_C', 'grade_D',
       'grade_E', 'grade_F', 'grade_G', 'home_ownership_MORTGAGE',
       'home_ownership_OWN', 'home_ownership_RENT', 'purpose_car',
       'purpose_credit_card', 'purpose_debt_consolidation',
       'purpose_home_improvement', 'purpose_house', 'purpose_major_purchase',
       'purpose_medical', 'purpose_moving', 'purpose_other',
       'purpose_small_business', 'purpose_vacation', 'purpose_wedding',
       'term_ 36 months', 'term_ 60 months'
        ]
        # Getting the Online features by using the entity keys.

        test = fs.get_online_features(
            entity_rows=[{"id": int(customer_id)}], # Entity keys
            feature_view=['default_loan_feature-2'], # Feature View name
            features=cols,
        ).to_df()       

        model = pickle.load(open("best-model.pkl", "rb"))

        output = model.predict(test.drop("id", axis=1))

        if output[0] == 1:
            st.sidebar.write("The applicant may found out as DEFAULT.")
        else:
            st.sidebar.write("The Applicant is GENUINE, We can Approve the Loan.")
    
else:
    annual_income = st.sidebar.text_input("Enter your annual income")
    short_emp = st.sidebar.radio("Whether your under short employement(less than 1 year exp.)",("yes","no"))
    if short_emp == 'yes':
        short_emp = 1
    else:
        short_emp = 0
    
    emp_exp = st.sidebar.text_input("Years of Experience")

    dti = st.sidebar.text_input("Your Debt-to-income Ratio(0 - 50)")

    delinq = st.sidebar.radio("Your Last delinq status",("yes", "no"))

    if delinq == "yes":
        delinq = 1
    else:
        delinq = 0

    od_ratio = st.sidebar.text_input("Enter your Over-Draft ratio (0-1).")

    grade = st.sidebar.selectbox("Choose your Loan Grade",("A", "B", "C", "D", "E", "F", "G"))

    home_ownership = st.sidebar.selectbox("Choose home ownership",('RENT', 'OWN', 'MORTGAGE'))

    purpose = st.sidebar.selectbox("Choose the loan purpose",('credit_card', 'debt_consolidation', 'medical', 'other',
       'home_improvement', 'small_business', 'major_purchase', 'vacation',
       'car', 'house', 'moving', 'wedding'))
    
    term = st.sidebar.selectbox("Choose the Repay-Terms",(' 36 months', ' 60 months'))

    if st.sidebar.button('Prediction'):
        data = [grade, int(annual_income), short_emp, int(emp_exp), home_ownership, float(dti), purpose, term, delinq, float(od_ratio)]

        columns = ['grade', 'annual_inc', 'short_emp','emp_length_num', 'home_ownership', 'dti', 'purpose', 'term','last_delinq_none', 'od_ratio']
    
        final_data = pd.DataFrame(dict(zip(columns, data)), index = [0])
    
        encoder = pickle.load(open("one-hot-encoder.pkl", "rb"))

        obj_cols = ['grade', 'home_ownership', 'purpose', 'term']
    
        enc_df = pd.DataFrame(encoder.transform(final_data[obj_cols]).toarray(),columns = encoder.get_feature_names(obj_cols))
    
        enc_df = final_data.join(enc_df)

        clean_data = enc_df.drop(obj_cols, axis = 1)

        model = pickle.load(open("best-model.pkl", "rb"))

        output = model.predict(clean_data)

        st.sidebar.subheader("Predict")

        if output[0] == 1:
            st.sidebar.write("The applicant may found out as DEFAULT.")
        else:
            st.sidebar.write("The Applicant is GENUINE, We can Approve the Loan.")

st.title("Bank Loan Default Prediction.")
st.write('''Bank loan default is a classic use case where ML models can be deployed to predict risky customers and hence minimize losses of the lenders. Financial industry is highly regulated, thus any model deployed or classification of customers basis their behavior, demographics etc. is highly regulated and must be explained to authorities to ensure unbiased operations.

Loans are risky but at the same time it is also a product that generates profits for the institution through differential borrowing/ lending rates.

The ML model should be explainable and be able to balance between risk and profits.''')
st.image("loan-predict-src.png", use_column_width=True)
