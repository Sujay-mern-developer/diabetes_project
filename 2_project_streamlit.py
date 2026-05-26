import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from models.train_model import train_models
from utils.pdf_report import create_pdf

# ✅ MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="Diabetes AI System", layout="centered")

# Load CSS
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("🩺 AI-Powered Diabetes Prediction System")

# Simple Login
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "1234":
            st.session_state.logged_in = True
        else:
            st.error("Invalid Credentials")

    st.stop()


# Simple Login
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# If NOT logged in → show login form
if not st.session_state.logged_in:
    st.subheader("🔐 Login Required")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "1234":
            st.session_state.logged_in = True
            st.rerun()   # 🔥 Refresh after login
        else:
            st.error("Invalid Credentials")

    st.stop()

# ✅ If logged in → show logout button
col1, col2 = st.columns([8, 2])
with col2:
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()   # 🔥 Refresh after logout


# Train Models
svm_model, log_model, scaler, X_train, X_test, Y_train, Y_test = train_models()

# Model Accuracy
svm_acc = accuracy_score(svm_model.predict(X_test), Y_test)
log_acc = accuracy_score(log_model.predict(X_test), Y_test)

st.subheader("📊 Model Comparison")

st.write(f"SVM Accuracy: {svm_acc:.2f}")
st.write(f"Logistic Regression Accuracy: {log_acc:.2f}")

# Accuracy Chart
fig, ax = plt.subplots()
ax.bar(["SVM", "Logistic"], [svm_acc, log_acc])
st.pyplot(fig)

st.divider()

# Input Fields
pregnancies = st.number_input("Pregnancies", 0, 20)
glucose = st.number_input("Glucose")
blood_pressure = st.number_input("Blood Pressure")
skin_thickness = st.number_input("Skin Thickness")
insulin = st.number_input("Insulin")
bmi = st.number_input("BMI")
diabetes_pedigree = st.number_input("Diabetes Pedigree Function")
age = st.number_input("Age", 1, 120)

if st.button("Predict"):
    input_data = np.asarray((pregnancies, glucose, blood_pressure,
                             skin_thickness, insulin, bmi,
                             diabetes_pedigree, age)).reshape(1, -1)

    input_data = scaler.transform(input_data)

    prediction = svm_model.predict(input_data)
    prob = svm_model.predict_proba(input_data)

    result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
    confidence = prob[0][1] * 100

    st.success(f"Prediction: {result}")
    st.write(f"Confidence: {confidence:.2f}%")

    pdf_path = create_pdf(result, confidence)

    with open(pdf_path, "rb") as f:
        st.download_button("Download Medical Report", f, file_name="report.pdf")