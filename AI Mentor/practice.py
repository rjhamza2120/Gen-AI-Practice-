from itertools import zip_longest
import streamlit as st
from streamlit_chat import message
import google.generativeai as genai
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

API_KEY = "AIzaSyAiY0437L_hp1xudDS12ARZk5W9UCxTn7A"

# Set Streamlit page configuration
st.set_page_config(page_title="AI Mentor ChatBot")
st.title("AI Mentor")

# Initialize session state variables
if 'entered_prompt' not in st.session_state:
    st.session_state['entered_prompt'] = ""  # Store the latest user input

if 'generated' not in st.session_state:
    st.session_state['generated'] = []  # Store AI-generated responses

if 'past' not in st.session_state:
    st.session_state['past'] = []  # Store past user inputs

# Define function to submit user input
def submit():
    # Set entered_prompt to the current value of prompt_input
    st.session_state.entered_prompt = st.session_state.prompt_input
    # Clear prompt_input
    st.session_state.prompt_input = ""

# Configure the Gemini API
genai.configure(api_key=API_KEY)

def build_message_list():
    """
    Build a list of messages including system, human, and AI messages.
    """
    # Start zipped_messages with the SystemMessage
    zipped_messages = [SystemMessage(
        content = """Your name is AI Mentor. You are an AI Technical Expert for Artificial Intelligence, here to guide and assist students with their AI-related questions and concerns. Please provide accurate and helpful information, and always maintain a polite and professional tone.

1. Greet the user politely, ask their name, and inquire how you can assist them with AI-related queries.
2. Provide informative and relevant responses to questions about artificial intelligence, machine learning, deep learning, natural language processing, computer vision, and related topics.
3. Avoid discussing sensitive, offensive, or harmful content. Refrain from engaging in any form of discrimination, harassment, or inappropriate behavior.
4. If the user asks about a topic unrelated to AI, politely steer the conversation back to AI or inform them that the topic is outside the scope of this conversation.
5. Be patient and considerate when responding to user queries, and provide clear explanations.
6. If the user expresses gratitude or indicates the end of the conversation, respond with a polite farewell.
7. Do not generate long paragraphs in response. Maximum Words should be 100.

Remember, your primary goal is to assist and educate students in the field of Artificial Intelligence. Always prioritize their learning experience and well-being."""
    )]

    # Zip together the past and generated messages
    for human_msg, ai_msg in zip_longest(st.session_state['past'], st.session_state['generated']):
        if human_msg is not None:
            zipped_messages.append(HumanMessage(content=human_msg))  # Add user messages
        if ai_msg is not None:
            zipped_messages.append(AIMessage(content=ai_msg))  # Add AI messages

    return zipped_messages

def generate_response():
    """
    Generate AI response using the Gemini model.
    """
    # Build the list of messages
    zipped_messages = build_message_list()

    # Generate response using the Gemini chat model
    ai_response = genai.generate_message(
        model="gemini-1.5-flash",
        messages=zipped_messages
    )

    response_text = ai_response["candidates"][0]["message"]["content"]

    return response_text

# Create a text input for the user
st.text_input('YOU: ', key='prompt_input', on_change=submit)

if st.session_state.entered_prompt != "":
    # Get user query
    user_query = st.session_state.entered_prompt

    # Append user query to past queries
    st.session_state.past.append(user_query)

    # Generate response
    output = generate_response()

    # Append AI response to generated responses
    st.session_state.generated.append(output)

# Display the chat history
if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        # Display AI response
        message(st.session_state["generated"][i], key=str(i))
        # Display user message
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
