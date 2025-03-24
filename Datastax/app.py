import os
import streamlit as st 
from dotenv import load_dotenv

load_dotenv()

ASTRA_DB_API_ENDPOINT=os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN=os.getenv("ASTRA_DB_APPLICATION_TOKEN")
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ["MISTRAL_API_KEY"]=os.getenv("MISTRAL_API_KEY")
os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")


from langchain_astradb import AstraDBVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vector_store = AstraDBVectorStore(
    collection_name="astra_vector_langchain",
    embedding=embeddings,
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    token=ASTRA_DB_APPLICATION_TOKEN,
)

from datasets import load_dataset
ds = load_dataset("datastax/philosopher-quotes")["train"]

from langchain.sc import Document

docs = []
for entry in ds:
    metadata = {"author": entry["author"]}
    if entry["tags"]:

        for tag in entry["tags"].split(";"):
            metadata[tag] = "y"

    doc = Document(page_content=entry["quote"], metadata=metadata)
    docs.append(doc)