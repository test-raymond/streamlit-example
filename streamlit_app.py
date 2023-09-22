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
    I kept its "freedom of speech" (aka temperature) pretty low so, hopefully, it does not hallucinate too much \U0001f600. If in doubt, check my CV. Have fun!')
    
    # Read database on Google sheet
    ###############################
    @st.cache_resource
    def access_sheet(sheet_name):
        '''
        Access the Google's spreadsheet. 
        '''
        # From local computer
        scope = ['https://spreadsheets.google.com/feeds',
                 'https://www.googleapis.com/auth/drive']
        credentials = service_account.Credentials.from_service_account_info(
                    st.secrets["gcp_service_account"], scopes = scope)
        gc = gspread.authorize(credentials)
        sheet = gc.open('ama-chatbot-db').worksheet(sheet_name)
        return sheet

    def parse_numbers(s):
        if s != '':
            return [float(x) for x in s.strip('[]').split(',')]
        else:
            return ''

    @st.cache_data
    def get_data():
        '''
        Read the biographical information from the Google sheet
        Returns
        -------
        df : Pandas dataframe
            Contains columns 'section', 'content', 'embeddings' & 'num_tokens'
        '''
        sheet = access_sheet('info')
        data = sheet.get_all_values()
        df = pd.DataFrame(data[1:], columns=['section', 'content', 'num_tokens', 'embeddings'])
        for col in ['embeddings']:
            df[col] = df[col].apply(lambda x: parse_numbers(x))
        for col in ['num_tokens']:
            df[col] = df[col].apply(lambda x: int(x) if x != '' else '')
        return df
    
    
    # Write database with embeddings back to Google sheet
    # This function should be run only when a new entry is
    # made in the database
    #####################################################
    @st.cache_data
    def record_embeddings(df):
        '''
        Write the embeddings and number of tokens of each entry in the Google
        sheet. This function should be done only one time or anytime a new entry
        is made in the database.
        '''
        sheet = access_sheet('info')
        for row in range(len(df)):
            sheet.update_cell(row+2, 3, str(df.loc[row, 'num_tokens']))
            sheet.update_cell(row+2, 4, str(df.loc[row, 'embeddings']))

    # Calculate embeddings of the combined section+content
    ######################################################

    # Set the OpenAI API key
    openai.api_key = st.secrets["openai_api_key"]

    def get_embeddings(text, method='text-embedding-ada-002'):
        '''
        Calculate embeddings.

        Parameters
        ----------
        text : str
            Text to calculate the embeddings for.
        method : str
            Method indicates which model from OpenAI to use.
        Returns
        -------
            List of the embeddings
        '''

        model = method
        result = openai.Embedding.create(
          model=model,
          input=text
        )
        embedding = result["data"][0]["embedding"]

        return embedding

    @st.cache_data
    def add_embeddings(df, method):
        '''
        Calculate embeddings and number of tokens of the combined section+content. 
        Parameters
        ----------
        df : Pandas dataframe
            Biographical info without embeddings
        method : str
            Method indicates which model from OpenAI to use.
        Returns
        -------
        df : Pandas dataframe
            Biographical info with embeddings
        '''
        # Combine title and body
        df['combined'] = "Title: " + df.section.str.strip() + "; Content: " + df.content.str.strip()

        # Caculate the embeddings for the combined title + body
        df['embeddings'] = df.combined.apply(lambda x: get_embeddings(x, method))

        # Calculate number of tokens and remove posts that are too long
        df['num_tokens'] = df.combined.apply(lambda x: len(tokenizer.encode(x)))

        return df

    def update_data(method):
        '''
        If new entries were made in the biographical database, we need to calculate
        the new embeddings. Apply this function to do so.
        '''
        df = get_data()
        df = add_embeddings(df, method)
        record_embeddings(df)

    def vector_similarity(x, y):
        '''
        Calculate the dot product between two vectors.
        Parameters
        ----------
        x : Numpy array
        y : Numpy array
        Returns
        -------
        Float
            Dot product
        '''
        return np.dot(np.array(x), np.array(y))

    def order_entries_by_similarity(query, df, method):
        '''
        Calculate the similarity measure for each biographical entry compared to
        a given query.
        Parameters
        ----------
        query : str
            Query.
        df : Pandas dataframe
            Biographical info with embeddings
        method : str
            Method indicates which model to use
        Returns
        -------
        df : Pandas dataframe
            Biographical info with a new column 'similarity'.
        '''
        query_embedding = get_embeddings(query, method)
        df['similarity'] = df['embeddings'].apply(lambda x: vector_similarity(x, query_embedding))
        df.sort_values(by='similarity', inplace=True, ascending=False)
        df.reset_index(drop=True, inplace=True)

        return df

        # Construct the prompt
    ######################

    # Set the tokenizer
    @st.cache_resource
    def load_tokenizer():
        return GPT2TokenizerFast.from_pretrained("gpt2")

    def get_max_num_tokens():
        '''
        Max number of tokens a pre-trained NLP model can take.
        '''
        return 2046

    def construct_prompt(query, df, method):
        '''
        Construct the prompt to answer the query. The prompt is composed of the
        query (from the user) and a context containing  the biographical entries
        the most relevant (similar) to the query.
        Parameters
        ----------
        query : str
            Query.
        df : Pandas dataframe
            Biographical info with embeddings.
        method : str
            Method indicates which model to use
        Returns
        -------
        prompt : str
            Prompt.
        '''

        MAX_SECTION_LEN = get_max_num_tokens()
        SEPARATOR = "\n* "
        tokenizer = load_tokenizer()
        separator_len = len(tokenizer.tokenize(SEPARATOR))

        chosen_sections = []
        chosen_sections_len = 0

        # Order posts_df by their similarity with the query
        df = order_entries_by_similarity(query, df, method)

        for section_index in range(len(df)):
            # Add contexts until we run out of space.        
            document_section = df.loc[section_index]

            chosen_sections_len += document_section.num_tokens + separator_len
            if chosen_sections_len > MAX_SECTION_LEN:
                break

            chosen_sections.append(SEPARATOR + document_section.content.replace("\n", " "))

        header = """
        Answer as Francois, a French-American data scientist who likes to be humorous
        and speak candidly.
        Context:\n
        """
        prompt = header + "".join(chosen_sections) + "\n\n Q: " + query + "\n A:"

        return prompt

    def record_question_answer(user, query, answer):
        '''
        Record the query, prompt and answer in the database
        '''
        sheet = access_sheet('Q&A')
        # Read how many records we have
        data = sheet.get_all_values()
        df = pd.DataFrame(data[1:], columns=['user', 'date', 'query', 'answer'])
        num_records = len(df)
        today_str = datetime.datetime.strftime(datetime.datetime.today(), '%Y-%m-%d')
        sheet.update_cell(num_records+2, 1, user)
        sheet.update_cell(num_records+2, 2, today_str)
        sheet.update_cell(num_records+2, 3, query)
        sheet.update_cell(num_records+2, 4, answer)

    def ama_chatbot(query, df, method):
        '''
        Use a pre-trained NLP method to answer a question given a database
        of information.
        The function also records the query, the prompt, and the answer in
        the database.
        Parameters
        ----------
        query : str
            Query
        df : Pandas dataframe
            Biographical info with embeddings.
        method : str
            Method indicates which model to use, either 'openai' for using the OpenAI
        API for 'text-embedding-ada-002', or 'huggingface' for using locally
        'paraphrase-MiniLM-L6-v2'. In the former case, the output is only a string
        that will be used via the API. In the latter case, it is an actual model
        object.
        Returns
        -------
        answer : str
            Answer from the model.
        prompt : str
            Actual prompt built.
        '''

        # Construct the prompt
        prompt = construct_prompt(query, df, method)

        # Ask the question with the context with GPT3 text-davinci-003
        COMPLETIONS_MODEL = "text-davinci-003"

        response = openai.Completion.create(
            prompt=prompt,
            temperature=0.9,
            max_tokens=300,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            model=COMPLETIONS_MODEL
        )

        answer = response["choices"][0]["text"].strip(" \n")

        return answer, prompt
    
    # Prepare engine
    method = 'openai'
    df = get_data()    
    
    # Storing the chat
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []

    # Get user's input
    hello_message = "Hello, how are you?"
    def get_text(hello_message):
        input_text = st.text_input("You: ", hello_message, key="input")
        return input_text

    user_input = get_text(hello_message)

    # Get the answer
    if user_input:
        answer, prompt = ama_chatbot(user_input, df, method)
        # Store the output 
        st.session_state.past.append(user_input)
        st.session_state.generated.append(answer)
        # Record the interaction if not the hello message
        if user_input != hello_message:
            record_question_answer(st.session_state['kept_username'], user_input, answer)

    # Display the chat    
    if st.session_state['generated']:

        for i in range(len(st.session_state['generated'])-1, -1, -1):
            message(st.session_state["generated"][i], key=str(i), avatar_style="bottts")
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user', avatar_style='human')
