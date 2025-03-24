import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_mistralai import MistralAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


load_dotenv()

os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")

st.title("GROQ DOCUMENT QnA")

if "vectors" not in st.session_state:
    st.session_state.loader = PyPDFLoader("data.pdf")
    st.session_state.docs = st.session_state.loader.load()
    st.session_state.splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
    st.session_state.splitted_docs = st.session_state.splitter.split_documents(st.session_state.docs)
    st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    st.session_state.vectors = FAISS.from_documents(st.session_state.splitted_docs, st.session_state.embeddings)

llm = ChatGroq(model="llama3-70b-8192",
               temperature=0)

prompt = ChatPromptTemplate.from_template("""
You are given the data of the diseases.
Now if the user asks about any query related to that please answer it in a context that it will be satisfied also tell about the disease causes,
and then give precautions and different prescriptions.
Dont use technical jargons only answer in simple words.
If the user asks about anything that is not in your content then warmly ask him to give query that is about the given context.

<context>
{context}
<context>

Question:{input}
"""
)

prompt1 = st.text_input("Ask Your Query....")

if prompt1:
    document_chain = create_stuff_documents_chain(llm,prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever,document_chain)
    answer = retrieval_chain.invoke({"input":prompt1})
    st.write(answer['answer'])
