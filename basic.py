
import streamlit as st

#OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

#PaLM
import google.generativeai as palm
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm

#HF
from hugchat import hugchat
from hugchat.login import Login

#Local LLM
import openai
import os

option = st.sidebar.selectbox(
    'Select AI',
    ('OpenAI', 'PaLM', 'Hugging Face' ,'Local LLM'))
if option == 'OpenAI':
    openai_key = st.sidebar.text_input('Insert an OpenAI API key: ')    
    if not (openai_key):
        st.sidebar.error('Please enter your OpenAI API key!', icon='⚠️')
        st.title("DeltaPi Chatbot")
        st.subheader("Your AI-Powered Conversation Companion!")
        st.write("Welcome to DeltaPi Chatbot, your new AI companion that's here to listen, engage, and assist. Combining cutting-edge AI technologies from OpenAI, PaLM, and Hugging Face, DeltaPi offers a unique chatting experience that's both informative and friendly. Whether you have questions, need advice, or just want to explore the world of AI, DeltaPi is ready for a warm and engaging conversation. Choose your preferred AI platform and start a delightful journey of interaction and discovery. DeltaPi is more than a chatbot - it's a friend in the realm of AI, always here to chat and learn with you.")

    else:
        st.sidebar.success('Proceed to entering your prompt message!', icon='👉')
        llm = OpenAI(openai_api_key=openai_key)
    
        # Optionally, specify your own session_state key for storing messages
        msgs = StreamlitChatMessageHistory(key="special_app_key")

        memory = ConversationBufferMemory(memory_key="history", chat_memory=msgs)
        if len(msgs.messages) == 0:
            msgs.add_ai_message("How can I help you?")


        template = """You are DeltaPi AI having a conversation with a user.

        {history}
        user: {user_input}
        DeltaPi AI: """
        prompt = PromptTemplate(input_variables=["history", "user_input"], template=template)

        # Add the memory to an LLMChain as usual
        llm_chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

        st.title("DeltaPi Chatbot")

        for msg in msgs.messages:
            st.chat_message(msg.type).write(msg.content)

        if prompt := st.chat_input("Ask to the DeltaPi: "):
            st.chat_message("user").write(prompt)

            # As usual, new messages are added to StreamlitChatMessageHistory when the Chain is called.
            response = llm_chain.run(prompt)
            st.chat_message("DeltaPi AI").write(response)


if option == 'PaLM':
    palm_api_key = st.sidebar.text_input('Insert a PaLM API key: ')

    if not (palm_api_key):
        st.sidebar.error('Please enter your PaLM API key!', icon='⚠️')
        st.title("DeltaPi Chatbot")
        st.subheader("Your AI-Powered Conversation Companion!")
        st.write("Welcome to DeltaPi Chatbot, your new AI companion that's here to listen, engage, and assist. Combining cutting-edge AI technologies from OpenAI, PaLM, and Hugging Face, DeltaPi offers a unique chatting experience that's both informative and friendly. Whether you have questions, need advice, or just want to explore the world of AI, DeltaPi is ready for a warm and engaging conversation. Choose your preferred AI platform and start a delightful journey of interaction and discovery. DeltaPi is more than a chatbot - it's a friend in the realm of AI, always here to chat and learn with you.")

    else:
        st.sidebar.success('Proceed to entering your prompt message!', icon='👉')
        palm.configure(api_key=palm_api_key)
    

        models = [
        m for m in palm.list_models() if "generateText" in m.supported_generation_methods
        ]
        for m in models:
            st.sidebar.write(f"Model Name: {m.name}")

        model = models[0].name

        llm = GooglePalm()
        llm.temperature = 0.1

        # Optionally, specify your own session_state key for storing messages
        msgs = StreamlitChatMessageHistory(key="special_app_key")

        memory = ConversationBufferMemory(memory_key="history", chat_memory=msgs)
        if len(msgs.messages) == 0:
            msgs.add_ai_message("How can I help you?")


        template = """You are DeltaPi AI having a conversation with a user.

        {history}
        user: {user_input}
        DeltaPi AI: """
        prompt = PromptTemplate(input_variables=["history", "user_input"], template=template)

        # Add the memory to an LLMChain as usual
        llm_chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

        st.title("DeltaPi Chatbot")

        for msg in msgs.messages:
            st.chat_message(msg.type).write(msg.content)

        if prompt := st.chat_input("Ask to the DeltaPi: "):
            st.chat_message("user").write(prompt)

            # As usual, new messages are added to StreamlitChatMessageHistory when the Chain is called.
            response = llm_chain.run(prompt)
            st.chat_message("DeltaPi AI").write(response)

if option == 'Hugging Face':
    # App title
    st.title("DeltaPi Chatbot")

    # Hugging Face Credentials
    with st.sidebar:
        hf_email = st.text_input('Enter E-mail:', type='password')
        hf_pass = st.text_input('Enter password:', type='password')
        if not (hf_email and hf_pass):
            st.error('Please enter your credentials!', icon='⚠️')
        else:
            st.success('Proceed to entering your prompt message!', icon='👉')
        
    # Store LLM generated responses
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Function for generating LLM response
    def generate_response(prompt_input, email, passwd):
        # Hugging Face Login
        sign = Login(email, passwd)
        cookies = sign.login()
        # Create ChatBot                        
        chatbot = hugchat.ChatBot(cookies=cookies.get_dict())
        return chatbot.chat(prompt_input)

    # User-provided prompt
    if prompt := st.chat_input(disabled=not (hf_email and hf_pass)):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_response(prompt, hf_email, hf_pass) 
                st.write(response) 
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)
