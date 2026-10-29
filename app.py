import streamlit as st
import pandas as pd

st.title("Alzheimer's Disease Classification App")

test_file = st.file_uploader("Upload your test data (CSV)", type="csv")
if uploaded_file is not None:
  data = pd.read_csv(test_file)
  st.write("Data Preview:", data.head())
