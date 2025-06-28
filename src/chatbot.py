import streamlit as st

class ChatBot:
    def __init__(self):
        #self.messages = [{"role": "system", "content": "You are a helpful assistant."}]
        self.init_session_state()

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
                response = f"You said: {user_input}"
                st.write(response)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                    })
