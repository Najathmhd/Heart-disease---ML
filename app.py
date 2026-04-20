import streamlit as st
import pandas as pd
import joblib

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load('heart_disease_model.pkl')

model = load_model()

st.title("💖 Heart Disease Prediction App")
st.markdown("Enter the patient's clinical data below to predict their risk of heart disease.")

# Create columns for layout
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 20, 100, 50)
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
    cp = st.selectbox("Chest Pain Type (cp)", options=[0, 1, 2, 3])
    trestbps = st.slider("Resting Blood Pressure (trestbps)", 90, 200, 120)
    chol = st.slider("Serum Cholestoral in mg/dl (chol)", 100, 400, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    restecg = st.selectbox("Resting Electrocardiographic Results (restecg)", options=[0, 1, 2])

with col2:
    thalach = st.slider("Maximum Heart Rate Achieved (thalach)", 70, 220, 150)
    exang = st.selectbox("Exercise Induced Angina (exang)", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    oldpeak = st.slider("ST depression induced by exercise (oldpeak)", 0.0, 6.0, 1.0, step=0.1)
    slope = st.selectbox("Slope of the peak exercise ST segment (slope)", options=[0, 1, 2])
    ca = st.slider("Number of major vessels (ca)", 0, 4, 0)
    thal = st.selectbox("Thal", options=[1, 2, 3], format_func=lambda x: "Normal (1)" if x == 1 else "Fixed Defect (2)" if x == 2 else "Reversable Defect (3)")

if st.button("Predict Heart Disease Risk", type="primary"):
    # Create a dataframe with the user inputs
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'cp': [cp],
        'trestbps': [trestbps],
        'chol': [chol],
        'fbs': [fbs],
        'restecg': [restecg],
        'thalach': [thalach],
        'exang': [exang],
        'oldpeak': [oldpeak],
        'slope': [slope],
        'ca': [ca],
        'thal': [thal]
    })
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    st.markdown("### Prediction Result")
    if prediction == 1:
        st.error("⚠️ **High Risk of Heart Disease Detected.** Please consult a healthcare professional.")
    else:
        st.success("✅ **Low Risk of Heart Disease.** Keep up the healthy lifestyle!")

st.markdown("---")
st.markdown("*Note: This is a demo app relying on a Machine Learning model and should NOT be used as real medical advice.*")
