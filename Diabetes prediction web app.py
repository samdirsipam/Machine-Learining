# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 23:32:29 2025

@author: samdi
"""

import numpy as np
import pickle
import streamlit as st

#loading the saved model
loaded_model=pickle.load(open('C:/Users/samdi/OneDrive/Desktop/DESKTOP/EDUCTION/MACHINE LEARNING/trained_model.sav','rb')) 

#creating funtion for prediction

def diabetes_prediction(input_data):
#changing the input data into numpy array
    input_data_as_numpy_array=np.asarray(input_data)

#reshape the array as we are predicting for one instance
    input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

    prediction=loaded_model.predict(input_data_reshaped)
    print(prediction)

    if(prediction[0]==0):
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'
    


def main():
    
    #giving a name
    st.title('Diabetes prediction Web App')
    
    
    #getting input data from the user
    Pregnencies= st.text_input('No. of Pregnencies')
    Glucose= st.text_input('Glucose Level')
    BloodPressure= st.text_input('Blood Pressure value')
    SkinThickness= st.text_input('SkinThickness value')
    Insulin= st.text_input('Insulin Level')
    BMI= st.text_input('BMI value')
    DiabetesPedigreeFunction= st.text_input('Diabetes Pedigree Function value')
    Age= st.text_input('Age of the Person')
    
    
    #code for prediction
    
    diagnosis=''
    
    #creating a button for prediction
    
    if st.button('Diabetes Test Result'):
            diagnosis= diabetes_prediction([Pregnencies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
    
    st.success(diagnosis)
    
    
if __name__=='__main__':
    main()
    
    