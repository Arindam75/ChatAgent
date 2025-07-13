import streamlit as st
from dotenv import load_dotenv
from src.chatbot import ChatPipeline
import os
import yaml


def generate_page():
    st.set_page_config(
        page_title="Welcome To Singapore Tourism",
        layout="wide",
        initial_sidebar_state="expanded",
    )

def load_config():
    with open("./config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config

def clear_chat():
    print(st.session_state.history)
    st.session_state.chatpipeline.init_session_state()


def build_sidebar():
    with st.sidebar:
        st.image("holiday.jpg")
        st.title("Content")
        st.sidebar.button("Clear Chat", on_click=clear_chat)
        st.sidebar.selectbox(
                    "Older Chats:",
                    options=['a', 'b', 'c'],
                    index=0, 
                    help="Choose a text or JSON file to display"
                )


def build_header():
    st.header("Welcome To Singapore Tourism ")


load_dotenv()
config = load_config()

if "chatpipeline" not in st.session_state:
    st.session_state.chatpipeline = ChatPipeline(config)


def main():

    generate_page()
    build_sidebar()
    build_header()
    
    st.session_state.chatpipeline.run()


if __name__ == "__main__":

    main()
