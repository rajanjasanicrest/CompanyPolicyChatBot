import streamlit as st
from streamlit_chat import message
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
import openai
import os
import time
import random

openai.api_key = os.getenv("OPENAI_API_KEY")
loader = PyPDFLoader("company_policy.pdf")
data = loader.load()

st.title('Policy Chatbot')

# st.write(data)

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(data, embeddings)

chain = ConversationalRetrievalChain.from_llm(
    llm = ChatOpenAI(temperature=0.0,model='gpt-3.5-turbo'),
    retriever = vectorstore.as_retriever(),
)


def conversational_chat(query):
    
    result = chain.invoke({"question": query, 
    "chat_history": st.session_state['history']})

    st.session_state['history'].append((query, result["answer"]))
    
    for word in result["answer"].split():
        yield word + " "
        time.sleep(0.05)


def gen_initial():
    response = random.choice(
        [
            "Hello there! How can I assist you today?",
            "Hi, human! Is there anything I can help you with?",
            "Do you need help?",
        ]
    )
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hello ! Ask me anything about Company Policy 🤗"]

if 'past' not in st.session_state:
    st.session_state['past'] = ["Hey ! 👋"]

    
if user_input := st.chat_input("Ask me anything about company policy"):
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_input)
    # Add user message to chat history
    st.session_state['past'].append({"role": "user", "content": user_input})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(conversational_chat(user_input))

    st.session_state['past'].append(user_input)
    st.session_state['generated'].append(response)
    # Add assistant response to chat history
    st.session_state['generated'].append({"role": "assistant", "content": response})
else:
    with st.chat_message('assistant'):
        start_message = st.write_stream(gen_initial())
    st.session_state['generated'].append({'role':'assistant', "content": start_message})
