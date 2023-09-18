import os
import streamlit as st
import pandas as pd
from google.oauth2 import service_account
import gspread
import openai
import numpy as np
from transformers import GPT2TokenizerFast
from sentence_transformers import SentenceTransformer
import datetime
from streamlit_chat import message

def check_password():
    """Returns `True` if the user had a correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if (
            st.session_state["username"] in st.secrets["passwords"]
            and st.session_state["password"]
            == st.secrets["passwords"][st.session_state["username"]]
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show inputs for username + password.
        st.text_input("Username", on_change=password_entered, key="username")
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input("Username", on_change=password_entered, key="username")
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 User not known or password incorrect")
        return False
    else:
        # Password correct.
        return True

if check_password():
    
    if 'kept_username' not in st.session_state:
        st.session_state['kept_username'] = st.session_state['username']

    # (adapted from https://medium.com/@avra42/build-your-own-chatbot-with-openai-gpt-3-and-streamlit-6f1330876846)
    st.set_page_config(page_title="Ask Me Anything (AMA), Francois Ascani's chatbot")
    st.title('Ask Me Anything!')
    st.subheader('A chatbot by and about Francois Ascani')
    st.markdown('Aloha! Here is a chatbot I built for you to ask questions about my professional journey. Like any other chatbot, \nit might hallucinate but \
    I kept its "freedom of speech" (aka temperature) pretty low so, hopefully, it does not hallucinate too much \U0001f600. If in doubt, check my CV. Have fun!'
