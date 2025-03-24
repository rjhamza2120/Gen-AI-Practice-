from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langserve import add_routes
import uvicorn
import os
from dotenv import load_dotenv

load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

app = FastAPI(
    title='Langchain Server',
    version='1.0',
    description='A Simple API Server'
    )

add_routes(
    app,
    ChatGoogleGenerativeAI(model='gemini-1.5-pro'),
    path = "/gemini"
)


model = ChatGoogleGenerativeAI(model='gemini-1.5-pro')

prompt1 = ChatPromptTemplate.from_template("Give the description on topics related to AI {topic} in 100 words")

add_routes(
    app,
    prompt1|model,
    path = "/explainAI"
)

if __name__ == "__main__":
    uvicorn.run(app,host="localhost",port=8000)