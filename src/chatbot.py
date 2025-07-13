import streamlit as st
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_community.chat_message_histories import ChatMessageHistory
from src.ragfusion import RagFusion


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
    def __init__(self, 
                 config: dict = None):
        
        self.ragfusion = RagFusion(llm=config["MODELS"]["LOCAL"]["MODEL_NAME"],
                                   base_url=config["MODELS"]["LOCAL"]["BASE_URL"],
                                   db_path=config["VECTOR_DB"]["DBPATH"],
                                   emb_model_name=config["MODELS"]["EMBEDDINGMODEL"]["MODEL_NAME"])
        
        self.llm = ChatOpenAI(model=config["MODELS"]["REMOTE"]["MODEL_NAME"])
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            ("human", "{question}"),
        ])
        self.chain = self.prompt | self.llm | StrOutputParser()

    def build_response(self, user_input):
        print(self.ragfusion.fusion_chain(user_input))
        return self.chain.invoke({"question": user_input})


class ChatPipeline:
    def __init__(self, config: dict = None):

        self.init_session_state()
        self.chatbot = ChatBot(config)

    def init_session_state(self):
        if "messages" not in st.session_state:
            st.session_state.messages = []
        else:
            st.session_state.messages.clear()
        if "history" not in st.session_state:
            st.session_state.history = ChatMessageHistory()

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
                st.markdown(user_input)
            st.session_state.history.add_user_message(user_input)
            print(st.session_state.history)
            with st.spinner("Thinking..."):
                response = self.chatbot.build_response(user_input)
                with st.chat_message("assistant"):

                    st.markdown(response)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response
                    })
                    st.session_state.history.add_ai_message(response)
 