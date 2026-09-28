import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("titanic_decision_tree_model.pkl")

st.title("Titanic Survival Prediction")
st.write("Enter passenger details to predict survival.")

pclass = st.selectbox(
    "Passenger Class",
    [1, 2, 3]
)

sex = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

age = st.slider(
    "Age",
    1,
    80,
    25
)

sibsp = st.slider(
    "Siblings/Spouses",
    0,
    8,
    0
)

parch = st.slider(
    "Parents/Children",
    0,
    6,
    0
)

fare = st.slider(
    "Fare",
    0.0,
    600.0,
    30.0
)

# Convert gender into the format used during training
sex_value = 0 if sex == "Male" else 1

input_data = pd.DataFrame({
    "pclass": [pclass],
    "sex": [sex_value],
    "age": [age],
    "sibsp": [sibsp],
    "parch": [parch],
    "fare": [fare]
})

if st.button("Predict Survival"):

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("Passenger is likely to SURVIVE!")
    else:
        st.error("Passenger is likely NOT to survive.")