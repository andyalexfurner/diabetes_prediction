#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 21:55:56 2023

@author: aniketmare
"""

import numpy as np 
import pickle 
import streamlit as st
import pandas as pd 
from sklearn.preprocessing import StandardScaler


# loading the saved model
loaded_model = pickle.load(open('/Users/aniketmare/Work/Diabetes_Prediction/trained_model.sav', 'rb'))

# prediction functions

#training the scaler
diabetes_dataset = pd.read_csv('/Users/aniketmare/Work/Diabetes_Prediction/diabetes.csv')
x = diabetes_dataset.drop(columns = 'Outcome', axis = 1)
scaler = StandardScaler()
scaler.fit(x)

def diabetes_prediction(input_data):

    # changing the input data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    
    # standardize the input data 
    std_data = scaler.transform(input_data_reshaped)

    prediction = loaded_model.predict(std_data)
    print(prediction)

    if (prediction[0] == 0):
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'
    
    
def main():
    
    
    
    #title
    st.title('Diabetes Prediction Web App')
    
    #taking input 
    
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure')
    SkinThickness = st.text_input('Skin Thickness')
    InsulinLevel = st.text_input('Insulin Level')
    BMI = st.text_input('Body Mass Index')
    DiabetesPedigreeFuntion = st.text_input('Diabetes Pedigree Funtion Value')
    Age = st.text_input('Age')
    
    
    #prediction
    diagnosis = ''
    
    
    #button 
    if st.button("Diabetes Test Result"):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, InsulinLevel, BMI, DiabetesPedigreeFuntion, Age])
    
    
    st.success(diagnosis)
    
    
if __name__ == '__main__':
    main()
    
    
    
    