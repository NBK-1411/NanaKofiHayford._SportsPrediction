import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load your trained model
model_filename = 'Trained_model.pkl'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

# Function to make predictions
def predict_player_rating(data):
    predictions = model.predict(data)
    return predictions

# Streamlit app
st.title("Player Rating Prediction")

# Input fields for user to enter data
player_name = st.text_input("Enter player's name:")
st.write("Enter player data (comma-separated values):")
user_input = st.text_input("Example: 1.0, 2.5, 3.3, ...")

if st.button("Predict"):
    # Convert user input into a dataframe
    try:
        input_data = np.array([float(i) for i in user_input.split(',')]).reshape(1, -1)
        input_df = pd.DataFrame(input_data)

        # Make predictions
        predictions = predict_player_rating(input_df)

        # Display the player's name and the predictions
        st.write(f"Player Name: {player_name}")
        st.write(f"Predicted Player Rating: {predictions[0]}")
    except ValueError:
        st.write("Please enter valid comma-separated values.")
