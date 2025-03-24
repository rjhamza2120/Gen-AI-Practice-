import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

os.environ["GOOGLE_API_KEY"] = os.getenv("google_api_key")

# os.environ["LANGCHAIN_API_KEY"] = os.getenv("langchain_api_key")

prompt = ChatPromptTemplate.from_messages(
    [
    ("system", "You are a helpful assistant. Please response to the user queries"),
    ("user", "Question {question}")
    ]
)

st.title('Langchain Demo With GOOGLE API')
input_text=st.text_input("Ask ur query")

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
output_parser=StrOutputParser()
chain=prompt|llm|output_parser

if input_text:
    st.write(chain.invoke(input_text))
