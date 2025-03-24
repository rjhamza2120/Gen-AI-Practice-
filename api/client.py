import requests
import streamlit as st

def get_response(input_text):
    response = requests.post("http://localhost:8000/explainAI/invoke",
    json = {'input':{'topic':input_text}})

    return response.json()['output']['content']


st.title("Langchain with Gemini API")
input_text = st.text_input("Give the topic to describe related to AI.....")

if input_text:
    st.write(get_response(input_text))

 