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

openai.api_key = "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"
loader = PyPDFLoader("company_policy.pdf")
data = loader.load()

st.title('Policy Chatbot')

# st.write(data)

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
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



response = random.choice(
    [
        "Hello there! How can I assist you today?",
        "Hi, human! Is there anything I can help you with?",
        "Do you need help?",
    ]
)

if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = [response]

if 'past' not in st.session_state:
    st.session_state['past'] = [""]

for i in range(len(st.session_state['generated'])):
    if i != 0:
        with st.chat_message('user'):
            st.markdown(st.session_state['past'][i])
        with st.chat_message('assistant'):
            st.markdown(st.session_state['generated'][i])
    else:
        with st.chat_message('assistant'):
            st.markdown(st.session_state['generated'][i])


if user_input := st.chat_input("Ask me anything about company policy"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_input)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(conversational_chat(user_input))

    st.session_state['past'].append(user_input)
    st.session_state['generated'].append(response)
    
