
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


option = st.sidebar.selectbox(
    'Select AI',
    ('OpenAI', 'PaLM (coming soon)', 'Hugging Face'))
if option == 'OpenAI': 
    libr = st.sidebar.selectbox('Select library', ('Autogen','Langchain'))

    if libr == 'Autogen':

        st.write("# DeltaPi Chat Company")


        # class TrackableAssistantAgent(Agent):
        #     def _process_received_message(self, message, sender, silent):
        #         with st.chat_message(sender.name):
        #             st.markdown(message)
        #         return super()._process_received_message(message, sender, silent)

        # class TrackableUserProxyAgent(ConversableAgent):
        #     def _process_received_message(self, message, sender, silent):
        #         with st.chat_message(sender.name):
        #             st.markdown(message)
        #         return super()._process_received_message(message, sender, silent)

        # class TrackableGPTAssistantAgent(GPTAssistantAgent):
        #     def _process_received_message(self, message, sender, silent):
        #         with st.chat_message(sender.name):
        #             st.markdown(message)
        #         return super()._process_received_message(message, sender, silent)


        # selected_model = None
        # selected_key = None

        

        # with st.sidebar:
        #     st.header("AI Configuration")
        #     selected_model = st.selectbox("GPT Model", ['gpt-3.5-turbo', 'gpt-4'], index=1)
        #     selected_key = st.text_input("API_Key", type="password")
        #     st.sidebar.text(' ')
        #     st.header("Agent Configuration")
        #     assistant_GPT_name = st.text_input('Agent GPT name ')
        #     assistant_GPT_inst = st.text_area('Agent GPT instructions ')
        #     st.sidebar.text(' ')
        #     uploaded_files = st.sidebar.file_uploader("Upload CSV files", accept_multiple_files=True)



        # with st.container():
        #     #for message in st.session_state["messages"]:
        #     #   st.markdown(message)

        #     user_input = st.chat_input("Task:")
        #     st.warning("Hello and welcome to DeltaPi Chat! 🌟 Need help? Just ask and let's make magic happen! 🚀")
        #     # Create an event loop
        #     if user_input:  
        #         if not selected_key or not selected_model:
        #             st.warning(
        #                 'You must provide valid OpenAI API key and choose preferred model', icon="⚠️")
        #             st.stop()

        #     llm_config = {
        #         "config_list": [
        #             {
        #                 "model": selected_model,
        #                 "api_key": selected_key
        #             }
        #         ]
        #     }

            

        # # The user agent
        # user_proxy = TrackableUserProxyAgent(
        #     name="DeltaPi_User",
        #     system_message="A human user of DeltaPi app.",
        #     code_execution_config={
        #         "work_dir": "chat"
        #     },
        #     max_consecutive_auto_reply=5,
        #     is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
        #     human_input_mode="NEVER"
        # )


        # # The agent playing the role of the product manager (PM)
        # gpt_assistant = TrackableGPTAssistantAgent(
        #     name=assistant_GPT_name,
        #     instructions=assistant_GPT_inst,
        #     llm_config={
        #         "config_list":  [
        #                 {
        #                     "model": selected_model,
        #                     "api_key": selected_key
        #                 }
        #             ],
        #         "assistant_id": None,
        #         "tools": [
        #             {
        #                 "type": "code_interpreter"
        #             }
        #         ],
        #     })



        # # Create an event loop
        # loop = asyncio.new_event_loop()
        # asyncio.set_event_loop(loop)


        # if user_input:
        #     async def initiate_chat():
        #         await user_proxy.a_initiate_chat(
        #             gpt_assistant,
        #             message=user_input,
        #         )

        #     # Run the asynchronous function within the event loop
        #     loop.run_until_complete(initiate_chat())

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
    
    
    
            uploaded_files = st.sidebar.file_uploader("Upload files", accept_multiple_files=True)
    
        
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
                    instructions=st.sidebar.text_area("Agent instructions: "),
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
