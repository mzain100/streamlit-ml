import streamlit as st
import numpy as np
import pandas as pd

# import ml package
import joblib
import os

# Paths to the model and encoders
model_file = 'RF_model.pkl'
encoder_file = 'label_encoders.pkl'
one_hot_encoder_file = 'one_hot_encoded.pkl'

# Mapping dictionaries for categorical variables
workclass_mapping = {'State-gov': 0, 'Self-emp-not-inc': 1, 'Private': 2, 'Federal-gov': 3, 
                     'Local-gov': 4, 'Self-emp-inc': 5, 'Without-pay': 6, 'Never-worked': 7}
education_mapping = {'Bachelors': 0, 'HS-grad': 1, '11th': 2, 'Masters': 3, '9th': 4, 
                     'Some-college': 5, 'Assoc-acdm': 6, 'Assoc-voc': 7, '7th-8th': 8, 
                     'Doctorate': 9, 'Prof-school': 10, '5th-6th': 11, '10th': 12, 
                     '1st-4th': 13, 'Preschool': 14, '12th': 15}
marital_status_mapping = {'Never-married': 0, 'Married-civ-spouse': 1, 'Divorced': 2, 
                          'Married-spouse-absent': 3, 'Separated': 4, 'Married-AF-spouse': 5, 
                          'Widowed': 6}
occupation_mapping = {'Adm-clerical': 0, 'Exec-managerial': 1, 'Handlers-cleaners': 2, 
                      'Prof-specialty': 3, 'Other-service': 4, 'Sales': 5, 'Craft-repair': 6, 
                      'Transport-moving': 7, 'Farming-fishing': 8, 'Machine-op-inspct': 9, 
                      'Tech-support': 10, 'Protective-serv': 11, 'Armed-Forces': 12, 
                      'Priv-house-serv': 13}
relationship_mapping = {'Not-in-family': 0, 'Husband': 1, 'Wife': 2, 'Own-child': 3, 
                        'Unmarried': 4, 'Other-relative': 5}
race_mapping = {'White': 0, 'Black': 1, 'Asian-Pac-Islander': 2, 'Amer-Indian-Eskimo': 3, 
                'Other': 4}
gender_mapping = {'Male': 0, 'Female': 1}

def load_model(model_file):
    try:
        loaded_model = joblib.load(model_file)
        return loaded_model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def load_one_hot_encoders(one_hot_encoder_file):
    try:
        one_hot_encoders = joblib.load(one_hot_encoder_file)
        return one_hot_encoders
    except Exception as e:
        st.error(f"Error loading one-hot encoders: {e}")
        return None

def get_value(val, my_dict):
    return my_dict.get(val, -1)  # Return -1 if the value is not found in the dictionary

def run_ml_app():
    st.title("Adult Income Prediction App")

    st.subheader("Input Your Data")
    age = st.number_input("age", 17, 90)
    workclass = st.selectbox('workclass', list(workclass_mapping.keys()))
    fnlwgt = st.number_input("fnlwgt")
    education = st.selectbox('education', list(education_mapping.keys()))
    marital_status = st.selectbox('marital_status', list(marital_status_mapping.keys()))
    occupation = st.selectbox('occupation', list(occupation_mapping.keys()))
    relationship = st.selectbox('relationship', list(relationship_mapping.keys()))
    race = st.selectbox('race', list(race_mapping.keys()))
    gender = st.radio('gender', list(gender_mapping.keys()))
    hours_per_week = st.number_input("hours_per_week", 1, 99)

    with st.expander("Your Selected Options"):
        result = {
            'age': age,
            'workclass': get_value(workclass, workclass_mapping),
            'fnlwgt': fnlwgt,
            'education': get_value(education, education_mapping),
            'marital_status': get_value(marital_status, marital_status_mapping),
            'occupation': get_value(occupation, occupation_mapping),
            'relationship': get_value(relationship, relationship_mapping),
            'race': get_value(race, race_mapping),
            'gender': get_value(gender, gender_mapping),
            'hours_per_week': hours_per_week,
        }
        st.write(result)

    # Load the model and encoders
    model = load_model(model_file)
    one_hot_encoders = load_one_hot_encoders(one_hot_encoder_file)

    if model is None or one_hot_encoders is None:
        st.error("Model or encoders could not be loaded. Check the logs for more details.")
        return

    # Create a DataFrame for the input
    input_df = pd.DataFrame([result])

    # Encode categorical features using one-hot encoders
    for feature, encoder in one_hot_encoders.items():
        if feature in input_df.columns:
            encoded_feature = encoder.transform(input_df[[feature]])
            encoded_feature_df = pd.DataFrame(encoded_feature, columns=encoder.get_feature_names_out([feature]))
            input_df = input_df.drop(columns=[feature])
            input_df = pd.concat([input_df, encoded_feature_df], axis=1)

    # Convert input data to array
    input_data = input_df.values

    # Prediction
    st.subheader('Prediction Result')
    try:
        prediction = model.predict(input_data)
        pred_proba = model.predict_proba(input_data)

        pred_probability_score = {'>50K': round(pred_proba[0][1] * 100, 4),
                                  '<=50K': round(pred_proba[0][0] * 100, 4)}

        if prediction == 1:
            st.success("Prediction: Income >50K")
            st.write(pred_probability_score)
        else:
            st.warning("Prediction: Income <=50K")
            st.write(pred_probability_score)
    except Exception as e:
        st.error(f"Error during prediction: {e}")
