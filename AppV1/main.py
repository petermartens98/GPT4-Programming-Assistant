import os
from dotenv import load_dotenv
import streamlit as st
import openai
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


def init_ses_states():
    if 'chatHistory' not in st.session_state: 
        st.session_state['chatHistory'] = []


def sidebar():
    global language, scenario, temperature
    languages = ['Python',
                'JavaScript',
                'GoLang',
                'C',
                'C++',
                'C#'
                ]
    scenarios = ['Code Correction',
                'Snippet Completion',
                ]
    with st.sidebar:
        with st.expander(label="Settings", expanded=True):
            language = st.selectbox(label="Language", options=languages)
            scenario = st.selectbox(label="Scenario", options=scenarios)
            temperature = st.slider(label="Temperature", min_value=0.0, max_value=1.0,value=0.5)


def main():
    st.set_page_config(page_title="GPT-4 Coding Asssistant", page_icon="computer")
    init_ses_states()
    sidebar()
    st.title("GPT-4 Coding Assistant")
    st.caption("Powered by OpenAI, LangChain, Streamlit")
    template = PromptTemplate(
        input_variables=['input','language','scenario'],
        template='''
        You are an AI Coding Assistant specializing in the "{language}" programming language.
        \nThe user has specified the mode to "{scenario}"
        \nUSER {language} CODE INPUT: 
        \n"{input}"
        '''
    )
    memory = ConversationBufferMemory(input_key="input", memory_key="chat_history")
    llm = OpenAI(temperature=temperature, model_name="gpt-4")
    llm_chain = LLMChain(llm=llm, prompt=template, memory=memory)
    user_input = st.text_area(label=f"Input {language} Code", height=400)
    if (st.button('Submit') and user_input):
        with st.spinner('Generating Response...'):   
            response = llm_chain.run(input=user_input, language=language, scenario=scenario)
            st.write(response)


if __name__ == '__main__':
    load_dotenv()
    main()

