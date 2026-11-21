import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Configure page

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Application title

st.title("Customer Churn Prediction")

st.write(
    "Predict whether a telecom customer is likely to churn."
)

st.success("Streamlit application is working!")