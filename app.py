import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/ask"

st.title("💰 Finance AI Assistant")

query = st.text_input("Ask a question:")

if st.button("Submit") and query:
    response = requests.get(API_URL, params={"query": query})
    data = response.json()
    st.write(data["answer"])
