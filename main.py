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


def clear_chat():
    print(st.session_state.history)
    st.session_state.chatbot.init_session_state()


def build_sidebar():
    with st.sidebar:
        st.image("holiday.jpg")
        st.title("Content")
        st.sidebar.button("Clear Chat", on_click=clear_chat)


def build_header():
    st.header("Welcome To Singapore Tourism ")


load_dotenv()
if "chatbot" not in st.session_state:
    st.session_state.chatbot = ChatPipeline()


def main():

    generate_page()
    build_sidebar()
    build_header()

    st.session_state.chatbot.run()


if __name__ == "__main__":

    main()
