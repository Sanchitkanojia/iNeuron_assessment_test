import streamlit as st
import pandas as pd
import pickle

# Load your trained model
model_path = 'rfc_model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

def main():
    st.title('Vehicle Insurance Interest Prediction')

    # Input fields
    gender = st.selectbox('Gender', ['Male', 'Female'])
    age = st.slider('Age', 18, 85, 30)
    region_code = st.number_input('Region Code', min_value=0.0, max_value=52.0, value=15.0, step=1.0)
    previously_insured = st.selectbox('Previously Insured', ['Yes', 'No'])
    vehicle_age = st.selectbox('Vehicle Age', ['< 1 Year', '1-2 Year', '> 2 Years'])
    vehicle_damage = st.selectbox('Vehicle Damage', ['Yes', 'No'])
    annual_premium = st.number_input('Annual Premium', min_value=1000, max_value=100000, value=30000)
    policy_sales_channel = st.number_input('Policy Sales Channel', min_value=1, max_value=163, value=26)
    vintage = st.slider('Vintage', 10, 300, 150)

    # Prepare the features DataFrame with explicit column ordering
    features = pd.DataFrame({
        'Gender': [1 if gender == 'Male' else 0],
        'Age': [age],
        'Region_Code': [region_code],
        'Previously_Insured': [1 if previously_insured == 'Yes' else 0],
        'Vehicle_Age': [2 if vehicle_age == '> 2 Years' else 1 if vehicle_age == '1-2 Year' else 0],
        'Vehicle_Damage': [1 if vehicle_damage == 'Yes' else 0],
        'Annual_Premium': [annual_premium],
        'Policy_Sales_Channel': [policy_sales_channel],
        'Vintage': [vintage]
    }, columns=['Gender', 'Age', 'Region_Code', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage'])  # Ensure this order matches exactly with the training setup

    # Predict button
    if st.button('Predict'):
        prediction = model.predict_proba(features)[:, 1]  # Assuming your model outputs probabilities for two classes
        st.write(f'Probability of being Interested: {prediction[0]:.2f}')

if __name__ == '__main__':
    main()
