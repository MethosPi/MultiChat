
import streamlit as st
import openai
import base64
import os

#OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.agents.openai_assistant import OpenAIAssistantRunnable

#PaLM
import google.generativeai as palm
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm

#HF
from hugchat import hugchat
from hugchat.login import Login

#Autogen
import asyncio
import autogen

#PandasAI
import pandas as pd
import matplotlib.pyplot as plt
from pandasai import SmartDataframe
from pandasai.llm import OpenAI
from pandasai.helpers.openai_info import get_openai_callback
import csv
import io

delta = '\u0394'
pi = '\u03C0'

option = st.sidebar.selectbox(
    'Select AI',
    ('OpenAI', 'PaLM (coming soon)', 'Hugging Face'))
if option == 'OpenAI': 
    libr = st.sidebar.selectbox('Select library', ('Autogen','Langchain','PandasAI (Data analysis)'))

    if libr == 'PandasAI (Data analysis)':

        prompt = st.text_input('Type "Plot" to plot')
        sidebar = st.sidebar
        # Create a sidebar menu
        apikey = sidebar.text_input("Insert your OpenAI API key")
        sidebar.text(' ')
        sidebar.text(' ')
        sidebar.text('Supported file types:')
        sidebar.text('CSV | EXCEL')
        uploaded_files = sidebar.file_uploader("Upload files", accept_multiple_files=True)
        sidebar.text(' ')

        delimiter_input = sidebar.text_input("CSV delimiter", max_chars=1)
        delimiter = delimiter_input
        dataframes = []
        

        
        llm = OpenAI(api_token=apikey)

        
        if uploaded_files:  # Controlla se ci sono file caricati
            columns = st.columns(len(uploaded_files))
            if len(uploaded_files) > 1:
                # Controlla se ci sono più di un file caricato
                        if st.button('PROMPT ALL', key='promptall_button'):
                                    for uploaded_file in uploaded_files:
                                        if uploaded_file.size > 0:  # Verifica se il file non è vuoto
                                            if uploaded_file.name.endswith('.csv'):
                                                def detect_delimiter(uploaded_file):
                                                    with io.StringIO(uploaded_file.getvalue().decode('utf-8')) as file:
                                                        content = '\n'.join(file.readlines()[:5])  # Ottieni solo le prime 5 righe del contenuto
                                                        dialect = csv.Sniffer().sniff(content)
                                                        return dialect.delimiter
        
                                                delimiter = detect_delimiter(uploaded_file)
                                                df = pd.read_csv(uploaded_file, delimiter=delimiter)
        
                                                if df.shape[1] == 1:
                                                    st.write('Wrong delimiter, please insert it manually')                          
                                            if not df.empty:  # Verifica se il DataFrame non è vuoto dopo la lettura del file
                                                dataframes.append(df)
                                        else:
                                            st.write(f'File {uploaded_file.name} is empty.')
        
                                    for i, df in enumerate(dataframes):
                                        st.write(f'File {i+1} {uploaded_file.name}:')
                                        df = SmartDataframe(df, config={"llm": llm, "conversational": False})
                                        with get_openai_callback() as cb:
                                            response = df.chat(prompt)
                                            if 'Plot' in prompt or 'chart' in prompt:
                                                plt.title(f'Chart {i+1} {uploaded_file.name}')
                                                st.pyplot(plt)
                                            else:
                                                st.write(response)
                                                st.write('---')   # Separatore tra i risultati dei prompt  # Separatore tra i risultati dei prompt
            for i, uploaded_file in enumerate(uploaded_files):       
                with columns[i]:
                    if uploaded_file.name.endswith('.csv'):
                        def detect_delimiter(uploaded_file):
                            with io.StringIO(uploaded_file.getvalue().decode('utf-8')) as file:
                                content = '\n'.join(file.readlines()[:5])  # Ottieni solo le prime 5 righe del contenuto
                                dialect = csv.Sniffer().sniff(content)
                                return dialect.delimiter
                        
                        delimiter = detect_delimiter(uploaded_file)
                        df = pd.read_csv(uploaded_file, delimiter=delimiter)
                        
                        if df.shape[1] == 1:
                            st.write('Wrong delimiter, please insert it manually')
        
                        if df.empty:  # Verifica se il DataFrame è vuoto dopo la lettura del file
                            st.write(f'File {i+1} ({uploaded_file.name}) is empty.')
                        else:
                            dataframes.append(df)
                            if st.button(f'Prompt {uploaded_file.name}', key=f'promptcsv_button_{i}'):
                                df = SmartDataframe(df, config={"llm": llm, "conversational": False})
                                with get_openai_callback() as cb:
                                    response = df.chat(prompt)
                                    if 'Plot' in prompt:
                                        # Plot the data
                                        plt.title('Chart')     
                                        # Display the plot
                                        st.pyplot(plt)
                                    elif 'chart' in prompt:
                                        # Plot the data
                                        plt.title('Plot')     
                                        # Display the plot
                                        st.pyplot(plt)
                                    else:
                                        st.write(response)
                                    # Buttons
        
                            if st.button('Show first 10 rows', key=f'10rcsv_button_{i}'):
                                st.write('First 10 rows:')
                                st.write(df.head(10))
        
                            if st.button('Describe', key=f'describecsv_button_{i}'):
                                st.write('Description:')
                                st.write(df.describe())
        
                            if st.button('Show number of rows and columns', key=f'numbercsv_button_{i}'):
                                st.write(f'Rows: {df.shape[0]}')
                                st.write(f'Columns: {df.shape[1]}')
        
                            if st.button('Duplicates rows', key=f'duplicatescsv_button_{i}'):
                                duplicates = df.duplicated().sum()
                                st.write(f'Duplicate rows: {duplicates}')
        
                            if st.button('Show CSV delimiter', key=f'delimiter_button_{i}'):
                                if df.shape[1] == 1 and delimiter == ',':
                                    st.write(f'Possible incorrect delimiter, please verify the delimiter in the "Show first 10 rows" section or insert either ";" or "|".')
                                else:
                                    st.write(delimiter)
                    #excel
                    elif uploaded_file.name.endswith('.xlsx'):
                        df = pd.read_excel(uploaded_file)
                        dataframes.append(df)
                        if st.button(f'Prompt {uploaded_file.name}', key='promptxlsx_button_(2)'):
                            df = SmartDataframe(df, config={"llm": llm, "conversational": False})
                            with get_openai_callback() as cb:
                                response = df.chat(prompt)
                                if 'Plot' in prompt:
                                    # Plot the data
                                    plt.title('Chart')     
                                    # Display the plot
                                    st.pyplot(plt)
                                elif 'chart' in prompt:
                                    # Plot the data
                                    plt.title('Plot')     
                                    # Display the plot
                                    st.pyplot(plt)
                                else:
                                    st.write(response) 
                        #Buttons
                        if st.button('Show first 10 rows', key='10rxlsx_button'):
                            st.write('First 10 rows:')
                            st.write(df.head(10))
                        
                        if st.button('Describe', key='describexlsx_button'):
                            st.write('Description:')
                            st.write(df.describe())
        
                        if st.button('Show number of rows and columns', key='numberxlsx_button'):
                            st.write(f'Rows: {df.shape[0]}')
                            st.write(f'Columns: {df.shape[1]}')
        
                        if st.button('Duplicates rows', key='duplicatesxlsx_button'):
                            duplicates = df.duplicated().sum()
                            st.write(f'Duplicate rows: {duplicates}')

    elif libr == 'Autogen':

        st.write("# DeltaPi Chat Company")

        os.environ["OPENAI_API_KEY"] = st.sidebar.text_input("OpenAI API Key", type="password")
        
        if os.environ["OPENAI_API_KEY"]:
            # Configurazione JSON come oggetto
            config_data = [
                {
                  "model": "gpt-4-1106-preview"
                }
            ]
            
            # Converti l'oggetto JSON in una stringa
            config_str = json.dumps(config_data)
        
            # Creazione di un file temporaneo
            with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json') as temp_file:
                temp_file.write(config_str)
                config_path = temp_file.name
        
            default_llm_config = {'temperature': 0.75}
        
            # Utilizza config_path con AgentBuilder
            config_list = autogen.config_list_from_json(config_path)
            default_llm_config = {'temperature': 0.75}
        
            # 2. Initialising Builder
            builder = AgentBuilder(config_path=config_path)
        
            # 3. Building agents
            building_task = st.sidebar.text_input("Company Builder:")
            if building_task:                 
                        
                agent_list, agent_configs = builder.build(building_task, default_llm_config)
                
                
            
                # 4. Multi-agent group chat
                group_chat = autogen.GroupChat(agents=agent_list, messages=[], max_round=10)
                manager = autogen.GroupChatManager(groupchat=group_chat, llm_config={"config_list": config_list, **default_llm_config})
        
                agents = agent_configs['agent_configs']
                with st.sidebar.status("Generating Agents..."):
                    for agent in agents:
                        name = agent['name']
                        system_message = agent['system_message']
                
                        # Stampa o visualizza i dettagli nell'interfaccia utente
                        st.write(f"Role: {name}")
                        st.write(f"Instructions: {system_message}")
                        st.write("-----")
                        
                user_input = st.chat_input("Ask to DeltaPi Agents: ")
        
                if user_input:
                    # Inizia la chat di gruppo con un messaggio dall'utente
                    response = agent_list[0].initiate_chat(
                        manager, 
                        message=user_input
                    )
        
        
                    # Mostra la risposta del chatbot
                    for messages in group_chat.messages:
                        with st.chat_message(messages["name"]):
                            st.markdown(messages["content"])

    elif libr == 'Langchain':
        openai_key = st.sidebar.text_input('Insert an OpenAI API key: ')    
        if not (openai_key):
            st.sidebar.error('Please enter your OpenAI API key!', icon='⚠️')
            st.title("DeltaPi Chatbot")
            st.subheader("Your AI-Powered Conversation Companion!")
            st.write("Welcome to DeltaPi Chatbot, your new AI companion that's here to listen, engage, and assist. Combining cutting-edge AI technologies from OpenAI and Hugging Face, DeltaPi offers a unique chatting experience that's both informative and friendly. Whether you have questions, need advice, or just want to explore the world of AI, DeltaPi is ready for a warm and engaging conversation. Choose your preferred AI platform and start a delightful journey of interaction and discovery. DeltaPi is more than a chatbot - it's a friend in the realm of AI, always here to chat and learn with you.")
    
        else:
            st.sidebar.success('Proceed to entering your prompt message!', icon='👉')
            llm = OpenAI(openai_api_key=openai_key)
    
    
    
            uploaded_files = st.sidebar.file_uploader("Upload Images", accept_multiple_files=True)
    
        
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
    
            if st.sidebar.toggle("Code Interpreter"):
                os.environ["OPENAI_API_KEY"] = openai_key
                assistant = OpenAIAssistantRunnable.create_assistant(
                    name="Code Interpreter Assistant", 
                    instructions="Code interpreter",
                    tools=[{"type": "code_interpreter"}],
                    model="gpt-4-1106-preview"
                )
                if prompt := st.chat_input("Ask to the DeltaPi Code Interpreter: "):
                    response = llm_chain.run(prompt)
                    output = assistant.invoke({"content": "Prompt:"+ prompt + "Write, execute and finally explain the respose:" + response})
                    st.chat_message("user").write(prompt)
                    for message in output:
                        # Iterate over the content list
                        for content_item in message.content:
                            # Check if the content item is of type 'text'
                            if content_item.type == 'text':
                                st.chat_message("DeltaPi AI").write(content_item.text.value)
    
    
    
            else:
                if prompt := st.chat_input("Ask to the DeltaPi: "):
                    st.chat_message("user").write(prompt)
                    response = llm_chain.run(prompt)
                    st.chat_message("DeltaPi AI").write(response)
                
            if uploaded_files:
    
                def analyze_image_with_gpt4(image_data):
                    chat = ChatOpenAI(temperature=0, openai_api_key=openai_key, model="gpt-4-vision-preview", max_tokens=256)
                
                    output = chat.invoke(
                        [
                            HumanMessage(
                                content=[
                                    {"type": "text", "text": f"Describe this image based on the prompt: {prompt}"},
                                    {"type": "image_url", "image_url": image_data}
                                ]
                            )
                        ]
                    )
                    return output
        
                for uploaded_file in uploaded_files:
                    # Convert the uploaded image to base64 for analysis
                    base64_image = base64.b64encode(uploaded_file.getvalue()).decode('utf-8')
                    image_data = f"data:image/png;base64,{base64_image}"
        
        
                    # Analyze the image with GPT-4 Vision
                    if prompt := st.chat_input("Ask to the DeltaPi about the image: "):
                        st.chat_message("user").write(prompt)
                        # As usual, new messages are added to StreamlitChatMessageHistory when the Chain is called.
                        result = analyze_image_with_gpt4(image_data)
                        st.chat_message("DeltaPi AI").write(result)


#if option == 'PaLM':
#    palm_api_key = st.sidebar.text_input('Insert a PaLM API key: ')
#
#    if not (palm_api_key):
#        st.sidebar.error('Please enter your PaLM API key!', icon='⚠️')
#        st.title("DeltaPi Chatbot")
#        st.subheader("Your AI-Powered Conversation Companion!")
#        st.write("Welcome to DeltaPi Chatbot, your new AI companion that's here to listen, engage, and assist. Combining cutting-edge AI technologies from OpenAI, PaLM, and Hugging Face, DeltaPi offers a unique chatting experience that's both informative and friendly. Whether you have questions, need advice, or just want to explore the world of AI, DeltaPi is ready for a warm and engaging conversation. Choose your preferred AI platform and start a delightful journey of interaction and discovery. DeltaPi is more than a chatbot - it's a friend in the realm of AI, always here to chat and learn with you.")
#
#    else:
#        st.sidebar.success('Proceed to entering your prompt message!', icon='👉')
#        palm.configure(api_key=palm_api_key)
#    
#
#       models = [
   #     m for m in palm.list_models() if "generateText" in m.supported_generation_methods
    #    ]
  #      for m in models:
 #           st.sidebar.write(f"Model Name: {m.name}")

#        model = models[0].name

       # llm = GooglePalm()
      #  llm.temperature = 0.1

        # Optionally, specify your own session_state key for storing messages
     #   msgs = StreamlitChatMessageHistory(key="special_app_key")

    #    memory = ConversationBufferMemory(memory_key="history", chat_memory=msgs)
   #     if len(msgs.messages) == 0:
  #          msgs.add_ai_message("How can I help you?")


  #      template = """You are DeltaPi AI having a conversation with a user.

     #   {history}
    #    user: {user_input}
   #     DeltaPi AI: """
  #      prompt = PromptTemplate(input_variables=["history", "user_input"], template=template)

        # Add the memory to an LLMChain as usual
 #       llm_chain = LLMChain(llm=llm, prompt=prompt, memory=memory)
#
#        st.title("DeltaPi Chatbot")
#
#        for msg in msgs.messages:
#            st.chat_message(msg.type).write(msg.content)
#
#        if prompt := st.chat_input("Ask to the DeltaPi: "):
#            st.chat_message("user").write(prompt)
#
#            # As usual, new messages are added to StreamlitChatMessageHistory when the Chain is called.
#            response = llm_chain.run(prompt)
#            st.chat_message("DeltaPi AI").write(response)

elif option == 'Hugging Face':
    # App title
    st.title("DeltaPi Chatbot")

    # Hugging Face Credentials
    with st.sidebar:
        hf_email = st.text_input('Enter E-mail:', type='password')
        hf_pass = st.text_input('Enter password:', type='password')
        if not (hf_email and hf_pass):
            st.error('Please enter your HF credentials!', icon='⚠️')
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
