import os
from dotenv import load_dotenv

load_dotenv()
groq_api = os.getenv("GROQ_API_KEY")
google_api = "AIzaSyAfWSNYkDmaIma83ozL4AibJD3R3bS6rP8"

import time
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

if "vector" not in st.session_state:
    st.session_state.loader = WebBaseLoader("https://uoc.edu.pk/mechatronics.html")
    st.session_state.docs = st.session_state.loader.load()
    st.session_state.splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=50)
    st.session_state.splitted_docs = st.session_state.splitter.split_documents(st.session_state.docs)
    st.session_state.embeddings_google = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api)
    st.session_state.vectors = FAISS.from_documents(st.session_state.splitted_docs, st.session_state.embeddings_google)

st.title("UOC MTE AGENT 🎓")
st.markdown("""
        <style>
        body {
            background-color: #f0f8ff; /* Alice blue background for a soft look */
            color: #333333; /* Dark grey text for readability */
        }
        .stApp {
            background-color: #f0f8ff; /* Consistent Alice blue for the main app container */
        }
        .stTextInput textarea {
            background-color: #ffffff; /* White background for the text area */
            border: 2px solid #1e90ff; /* Dodger blue border for high visibility */
            color: #1e90ff; /* Dodger blue text color inside the text area */
            box-shadow: 0 0 10px rgba(30, 144, 255, 0.5); /* Glowing blue shadow to enhance visibility */
        }
        .stTextInput textarea:focus {
            border: 2px solid #4169e1; /* Royal blue border when focused */
            box-shadow: 0 0 15px rgba(65, 105, 225, 0.8); /* Stronger blue shadow when focused */
        }
        .stButton>button {
            background-color: #1e90ff; /* Dodger blue button background color */
            color: white; /* Button text color */
            border-radius: 12px; /* Rounded corners for buttons */
            padding: 10px 20px; /* Add some padding to the buttons */
            font-size: 16px; /* Increase font size */
        }
        .stButton>button:hover {
            background-color: #4169e1; /* Royal blue button background color on hover */
        }
        h1, h2, h3 {
            color: #1e90ff; /* Dodger blue color for headers */
        }
        .stAudio {
            text-align: right; /* Align audio player to the right */
        }
        </style>
        """, unsafe_allow_html=True)

llm = ChatGroq(
    groq_api_key = groq_api,
    model="llama-3.2-90b-text-preview",
    temperature=0,
)

prompt = ChatPromptTemplate.from_template(
"""
You are given all the latest knowledge about the department of Mechatronics Engineering of University of Chakwal.
If the user asks about anything of the Mechatronics Engineering, it is your responsibility to clarify its query by giving to the point answer.
<context>
{context}
<context>
Questions:{input}

"""
)

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

input_prompt = st.text_input("Enter ur Question....")

if input_prompt:
    start=time.process_time()
    answer = retrieval_chain.invoke({"input":input_prompt})
    print("Response time :",time.process_time()-start)
    final_ans = answer['answer']
    st.write("Answer:",final_ans)

    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(answer["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")