import streamlit as st
from streamlit_chat import message
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
import tempfile
from langchain_community.document_loaders import PyPDFLoader
import openai
import os


openai.api_key = os.getenv("OPENAI_API_KEY")
loader = PyPDFLoader("company_policy.pdf")
data = loader.load()

st.title('Policy Chatbot')

# st.write(data)

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(data, embeddings)

chain = ConversationalRetrievalChain.from_llm(
    llm = ChatOpenAI(temperature=0.0,model_name='gpt-3.5-turbo'),
    retriever = vectorstore.as_retriever()
)


def conversational_chat(query):
    
    result = chain.invoke({"question": query, 
    "chat_history": st.session_state['history']})

    st.session_state['history'].append((query, result["answer"]))
    
    return result["answer"]



if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hello ! Ask me anything about Company Policy 🤗"]

if 'past' not in st.session_state:
    st.session_state['past'] = ["Hey ! 👋"]
    
#container for the chat history
response_container = st.container()
#container for the user's text input
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        
        user_input = st.text_input("Query:", placeholder="Ask me anything about company policy (:", key='input')
        submit_button = st.form_submit_button(label='Send')
        
    if submit_button and user_input:
        output = conversational_chat(user_input)
        
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
            message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")



