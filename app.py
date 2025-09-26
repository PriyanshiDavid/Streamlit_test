# app.py
import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the trained model
with open('melbourne_rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load feature list
with open('model_features.pkl', 'rb') as f:
    model_features = pickle.load(f)

st.title("Melbourne Housing Price Predictor 🏠")

st.write("""
Enter property details below to predict the price:
""")

# User inputs
Rooms = st.number_input("Number of Rooms", min_value=1, max_value=10, value=3)
Distance = st.number_input("Distance from CBD (km)", min_value=0.0, max_value=50.0, value=10.0)
Bedroom2 = st.number_input("Number of Bedrooms", min_value=0, max_value=10, value=2)
Bathroom = st.number_input("Number of Bathrooms", min_value=0, max_value=10, value=1)
Car = st.number_input("Number of Car Parks", min_value=0, max_value=10, value=1)
Landsize = st.number_input("Land Size (sqm)", min_value=0, max_value=10000, value=200)
BuildingArea = st.number_input("Building Area (sqm)", min_value=0, max_value=2000, value=150)
YearBuilt = st.number_input("Year Built", min_value=1800, max_value=2025, value=2000)
Propertycount = st.number_input("Number of Properties in Suburb", min_value=0, max_value=10000, value=4000)
SchoolsNearby = st.number_input("Number of Schools Nearby", min_value=0, max_value=20, value=5)
AgeOfProperty = st.number_input("Age of Property", min_value=0, max_value=150, value=22)

# Create dataframe with single row for prediction
input_dict = {
    'Rooms': Rooms,
    'Distance': Distance,
    'Bedroom2': Bedroom2,
    'Bathroom': Bathroom,
    'Car': Car,
    'Landsize': Landsize,
    'BuildingArea': BuildingArea,
    'YearBuilt': YearBuilt,
    'Propertycount': Propertycount,
    'SchoolsNearby': SchoolsNearby,
    'AgeOfProperty': AgeOfProperty
}

# Ensure all model features exist (handle dummy features if any)
for feature in model_features:
    if feature not in input_dict:
        input_dict[feature] = 0

input_df = pd.DataFrame([input_dict])
input_df = input_df[model_features]  # order columns correctly

# Apply scaler
input_df_scaled = scaler.transform(input_df)

# Predict
if st.button("Predict Price"):
    prediction = model.predict(input_df_scaled)
    st.success(f"Predicted Housing Price: AUD {prediction[0]:,.0f}")