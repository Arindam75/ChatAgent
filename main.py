import streamlit as st
from dotenv import load_dotenv
from src.chatbot import ChatBot
import os
import yaml

def generate_page():
    st.set_page_config(
        page_title="Welcome To Singapore Tourism",
        layout="wide",
        initial_sidebar_state="expanded",
    )

def build_sidebar():
    with st.sidebar:
        st.image("holiday.jpg")
        st.title("Content")
        option = st.sidebar.button("Clear Chat",
                           on_click=st.session_state.messages.clear)

def build_header():
    st.header("Welcome To Singapore Tourism ")



if "chatbot" not in st.session_state:
    st.session_state.chatbot = ChatBot()

def main():

    generate_page()
    build_sidebar()
    build_header()

    load_dotenv()

    st.session_state.chatbot.run()

if __name__ == "__main__":

    main()