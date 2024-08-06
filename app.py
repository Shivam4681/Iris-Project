# Import Packages
import pandas as pd
import numpy as np
import pickle
import streamlit as st


# Load the model and preprocessor
with open('Notebook/pipeline.pkl', 'rb') as file1:
    pre = pickle.load(file1)

# Load the model
with open('Notebook/model.pkl', 'rb') as file2:
    model = pickle.load(file2)


# Preprocess the Data
def predict_data(sep_len, sep_wid, pet_len, pet_wid):
    dct = {
        'sepal_length' : [sep_len],
        'sepal_width' : [sep_wid],
        'petal_length' : [pet_len],
        'petal_width' : [pet_wid]
    }
    xnew = pd.DataFrame(dct)
    xnew_pre = pre.transform(xnew)
    pred = model.predict(xnew_pre)
    prob = model.predict_proba(xnew_pre)
    max_prob = np.max(prob)
    return pred, max_prob


# Run Streamlit app
if __name__ == '__main__':
    st.set_page_config(page_title='Iris Project Shivam')
    st.title('Iris Project')

    # Create Button
    st.subheader('Please Provide Below Input')

    # Take input From user
    sep_len = st.number_input('sepal_Length: ', min_value=0.00, step=0.01)
    sep_wid = st.number_input('sepal_Width: ', min_value=0.00, step=0.01)
    pet_len = st.number_input('petal_Length: ', min_value=0.00, step=0.01)
    pet_wid = st.number_input('petal_width: ', min_value=0.00, step=0.01)

    # Create Button to Predict Output
    submit = st.button('predict')

    # If Submit button press
    if submit:
        pred, max_prob = predict_data(sep_len, sep_wid, pet_len, pet_wid)
        st.subheader('Model Response :')
        st.subheader(f'Prediction : {pred[0]}')
        st.subheader(f'probability : {max_prob}')
        st.progress(max_prob)