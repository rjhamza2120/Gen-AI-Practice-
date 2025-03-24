import os 
import streamlit as st
from dotenv import load_dotenv

from langchain.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_objectbox import ObjectBox
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate

load_dotenv()
os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")
groq_api = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    groq_api_key=groq_api,
    model="gemma-7b-it",
    temperature=0.3
)

prompt = ChatPromptTemplate.from_template(
"""
Answer the questions on the context given to you.
Dont give answer if the question is out of context just refuse it politely.

<context>
{context}
<context>

Question:{input}

"""
)

def main():
    if "vectors" not in st.session_state:
        st.session_state.loader = PyPDFDirectoryLoader("./data")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        st.session_state.splitted_docs = st.session_state.splitter.split_documents()
        st.sessiion_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.vectors = ObjectBox.from_documents(st.session_state.splitted_docs, st.session_state.embeddings)

if __name__=="__main__":
    st.title("Gemma Groq Inferencing")
    input_text = st.text_input("Enter your query....")

    if st.button("Create Data Base"):
        main()
        st.write("Embeddings Created!")

    if input_text:
        chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(chain, retriever)
        ans = retrieval_chain.invoke("{'input':input_text}")
        st.write(ans["answer"])
        