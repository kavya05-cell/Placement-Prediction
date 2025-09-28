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
# We now know the model expects two features: 'CGPA' and 'IQ'
st.header('Student Details')
cgpa = st.slider('CGPA', min_value=0.0, max_value=10.0, value=7.5, step=0.1)
iq = st.slider('IQ Score', min_value=50, max_value=150, value=100, step=1)

# Create a button to trigger the prediction
if st.button('Predict Placement'):
    # Prepare the input data for the model with BOTH features
    # Make sure the column names match the features the model was trained on
    input_data = pd.DataFrame([[cgpa, iq]], columns=['cgpa', 'iq']) 

    # Make a prediction
    prediction = model.predict(input_data)[0]

    # Display the result
    st.subheader('Prediction Result')
    if prediction == 1:
        st.success('Congratulations! The student is likely to be placed.')
    else:
        st.warning('The student is likely to be not placed. Consider further training.')


