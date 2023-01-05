import streamlit as st
from predict import show_predict_page

page = st.sidebar.selectbox("Predict or Explore",("Predict","Explore"))
if page == "Predict":
   show_predict_page()
