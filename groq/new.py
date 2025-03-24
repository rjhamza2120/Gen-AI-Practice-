import streamlit as st
import os
from dotenv import load_dotenv

from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

load_dotenv()
os.environ["GOOGLE_API_KEY"] = "AIzaSyBnbWikbcxHRfVXvpk9Nj6oLygZ-jYh-Mk"
groq_api = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    groq_api_key=groq_api,
    model="llama3-70b-8192",
    temperature=0.7
)

prompt = ChatPromptTemplate.from_template(
    """
    Answer the question on the context provided to you.
    Dont answer the questions if they are out of context and refuse politely.
    
    <context>
    {context}
    <context>
    Question:{input}
    """
)

def main():
    if "vectors" not in st.session_state:
        st.session_state.loader = PyPDFLoader(".\data\RAG for NLP.pdf")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        st.session_state.splitted_docs = st.session_state.splitter.split_documents(st.session_state.docs)
        st.session_state.embedding = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
        st.session_state.vectors = FAISS.from_documents(st.session_state.splitted_docs, st.session_state.embedding)

if __name__=="__main__":

    st.title("Document QnA")
    input_text = st.text_input("Enter ur query....")
    if st.button("Create Vector DB"): 
        main()  
        st.write("Vector Database Create Successfully!")

    if input_text:
        document_chain = create_stuff_documents_chain(llm,prompt)
        retrieval = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retrieval,document_chain)
        ans = retrieval_chain.invoke({"input":input_text})
        st.write("Answer:",ans["answer"])

        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(ans["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")
