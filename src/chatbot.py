import streamlit as st
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks.base import BaseCallbackHandler


class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self._buffer = ""
        self._placeholder = None

    def set_placeholder(self, placeholder):
        self._placeholder = placeholder

    def on_llm_new_token(self, token: str, **kwargs):
        self._buffer += token
        if self._placeholder:
            self._placeholder.markdown(self._buffer + "▌")

class ChatBot:
    def __init__(self, model=None):
        
        self.llm = ChatOpenAI(model=model, streaming=True)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            ("human", "{question}"),
        ])
        self.chain = self.prompt | self.llm | StrOutputParser()

    def stream_response(self, user_input, handler):
        self.llm.callbacks = [handler]
        return self.chain.stream({"question": user_input})

class ChatPipeline:
    def __init__(self):
        
        self.init_session_state()
        self.chatbot = ChatBot("gpt-4o-mini")

    def init_session_state(self):
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
    def chat_history(self):
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
    
    def run(self):
       
       self.chat_history()
       user_input = st.chat_input("Say Something !!")
       if user_input:

            st.session_state.messages.append({
                "role": "user",
                "content": user_input
                })

            with st.chat_message("user"):
                st.write(user_input)

            with st.chat_message("assistant"):
                placeholder = st.empty()
                stream_handler = StreamlitCallbackHandler()
                stream_handler.set_placeholder(placeholder)

                response = ""
                for token in self.chatbot.stream_response(user_input, stream_handler):
                    response += token  # collect full response as you stream

                # Save to session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })
 

