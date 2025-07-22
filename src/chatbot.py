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
        
        self.ragfusion = RagFusion(llm=config["MODELS"]["GEMINI"]["MODEL_NAME"],
                                   db_path=config["VECTOR_DB"]["DBPATH"],
                                   emb_model_name=config["MODELS"]["EMBEDDINGMODEL"]["MODEL_NAME"]) 
            
        self.llm = ChatOpenAI(model=config["MODELS"]["OPENAI"]["MODEL_NAME"])
        #self.prompt = ChatPromptTemplate.from_messages([
        #    ("system", "You are a helpful assistant."),
        #    ("human", "{question}"),
        #])
        #self.chain = self.prompt | self.llm | StrOutputParser()

    def build_response(self, user_input, history):
        prompt_template= """
            You are a polite and helpful assistant that answers questions related to Singapore Tourism. Follow the guidelines to answer the question.

            Guidelines:
            - Use the context provided below to answer the question.
            - If text is not found in the context, then answer "Apologies for the inconvenience, I could not find the information you were looking for."
            - The user question should be viewed in relation to the conversation history.

            Conext: {context}

            Conversation History: {history}

            Current Question: {user_input}
            """
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm | StrOutputParser()
        context = self.ragfusion.fusion_chain(user_input)
        return chain.invoke({"history": history, "context": context, "user_input": user_input})


class ChatPipeline:
    def __init__(self, config: dict = None):
        
        self.chatbot = ChatBot(config)
        self.init_session_state()

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
            history = st.session_state.history
            with st.spinner("Thinking..."):
                response = self.chatbot.build_response(user_input, history)
                with st.chat_message("assistant"):

                    st.markdown(response)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response
                    })
                    st.session_state.history.add_ai_message(response)
 