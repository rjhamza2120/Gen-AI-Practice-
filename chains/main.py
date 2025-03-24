import streamlit as st
import os 
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

loader = PyPDFLoader("attention.pdf")

data = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 800,
    chunk_overlap = 20
)

splitted_docs = splitter.split_documents(data)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


db = FAISS.from_documents(splitted_docs, embeddings)

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")


template = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context. 
Think step by step before providing a detailed answer. 
I will tip you $1000 if the user finds the answer helpful.
<context>
{context}
</context>
Question: {input}""")


chain = create_stuff_documents_chain(llm, template)

retriever = db.as_retriever()

retrieval = create_retrieval_chain(retriever, chain)

st.title("QnA with Attention is all you Need 🤖")
input_text = st.text_input("Ask ur Question .....")

if input_text:
    ans = retrieval.invoke({"input": input_text})
    st.write(ans['answer'])