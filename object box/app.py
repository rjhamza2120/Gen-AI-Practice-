import os
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_objectbox.vectorstores import ObjectBox
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from dotenv import load_dotenv
load_dotenv()

os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")
groq_api = os.getenv("GROQ_API_KEY")


llm = ChatGroq(
    groq_api_key=groq_api,
    model='llama3-70b-8192',
    temperature=0
    )

prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context only.
    Please provide the accurate response based on the question.
    <context>
    {context}
    <context>
    Question:{input}
    """
)

def vector_embedding():
    if 'vectors' not in st.session_state:

        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader = PyPDFDirectoryLoader('./data')
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.splitted_docs = st.session_state.splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = ObjectBox.from_documents(st.session_state.splitted_docs, st.session_state.embeddings,embedding_dimensions=768)

if __name__ == "__main__":

    st.title("Document QnA (objectbox)")

    input_prompt = st.text_input("Enter your Query......")
    if st.button("Upload Documents"):
        vector_embedding()
        st.write("Vector DB is ready!")


    if input_prompt:
        document_chain = create_stuff_documents_chain(llm,prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever,document_chain)
        ans = retrieval_chain.invoke({'input':input_prompt})
        st.write(ans["answer"])

        # with st.expander("Document Similarity Search"):
        #     for i,doc in enumerate(ans["context"])