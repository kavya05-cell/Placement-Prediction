# app.py
import streamlit as st
import joblib
import pandas as pd

# Load your pre-trained model (update the path as needed)
try:
    model = joblib.load('placement-prediction-model.pkl')
except FileNotFoundError:
    st.error("Model file not found. Please place your model in the 'models' directory.")
    st.stop()

# Set the title and a brief description
st.title('Student Placement Predictor')
st.write('Enter the student details to predict their placement status.')

# Create input widgets for your model's features
# Customize these to match the features your model was trained on
st.header('Student Details')
cgpa = st.slider('CGPA', min_value=0.0, max_value=10.0, value=7.5, step=0.1)
# Add more input widgets for other features, e.g., 'skills', 'interviews', etc.
# skills = st.selectbox('Skills', ['Python', 'Java', 'Data Science', 'None'])
# communication_score = st.number_input('Communication Score', min_value=0, max_value=100)

# Create a button to trigger the prediction
if st.button('Predict Placement'):
    # Prepare the input data for the model
    # The data must be in the same format as the training data
    input_data = pd.DataFrame([[cgpa]], columns=['CGPA']) # Replace with your actual feature names

    # Make a prediction
    prediction = model.predict(input_data)[0]

    # Display the result
    st.subheader('Prediction Result')
    if prediction == 1: # Assuming 1 = Placed
        st.success('Congratulations! The student is likely to be placed.')
        
    else: # Assuming 0 = Not Placed
        st.warning('The student is likely to be not placed. Consider further training.')
