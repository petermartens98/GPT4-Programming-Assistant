import os
from dotenv import load_dotenv
import streamlit as st
import openai
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from htmlTemplates import css, bot_template, user_template


def init_ses_states():
    if 'chat_history' not in st.session_state: 
        st.session_state['chat_history'] = []


def sidebar():
    global language, scenario, temperature
    languages = ['Python','JavaScript','GoLang','C','C++','C#']
    scenarios = ['Code Correction','Snippet Completion']
    with st.sidebar:
        with st.expander(label="Settings", expanded=True):
            language = st.selectbox(label="Language", options=languages)
            scenario = st.selectbox(label="Scenario", options=scenarios)
            temperature = st.slider(label="Temperature", min_value=0.0, max_value=1.0,value=0.5)


def display_convo():
    with st.container():
        for i, message in enumerate(reversed(st.session_state.chat_history)):
            if i % 2 == 0:
                 st.markdown(bot_template.replace("{{MSG}}", message), unsafe_allow_html=True)
            else:
                st.markdown(user_template.replace("{{MSG}}", message), unsafe_allow_html=True)



def main():
    st.set_page_config(page_title="GPT-4 Coding Asssistant", page_icon="computer")
    st.write(css, unsafe_allow_html=True)
    init_ses_states()
    sidebar()
    st.title("GPT-4 Coding Assistant")
    st.caption("Powered by OpenAI, LangChain, Streamlit")

    initial_template = PromptTemplate(
        input_variables=['input','language','scenario'],
        template='''
        You are a GPT-4 AI Coding Assistant specializing in the "{language}" programming language.
        \nThe user has specified the mode to "{scenario}"
        \nUSER {language} CODE INPUT: 
        \n{input}"
        \n Be sure to end your response by asking user if they need any further assistance
        \n\nAI {language} GPT4 CHATBOT RESPONSE HERE:\n
        '''
    )
    chat_template = PromptTemplate(
        input_variables=['input','language','scenario','chat_history'],
        template='''
        You are a GPT-4 AI Coding Assistant specializing in the "{language}" programming language.
        \nThe user has specified the mode to "{scenario}"
        \nINITIAL USER {language} INPUT: 
        \n"{input}"
        \nCHAT HISTORY:
        \n{chat_history}
        \n Be sure to end your response by asking user if they need any further assistance
        \n\nAI {language} GPT4 CHATBOT RESPONSE HERE:\n
        '''
    )
    memory = ConversationBufferMemory(input_key="input", memory_key="chat_history")
    llm = OpenAI(temperature=temperature, model_name="gpt-4")
    initial_llm_chain = LLMChain(llm=llm, prompt=initial_template, memory=memory)
    chat_llm_chain = LLMChain(llm=llm, prompt=chat_template, memory=memory)
    initial_input = st.text_area(label=f"Input {language} Code", height=300)
    if (st.button(f'Submit {language} Code') and initial_input):
        st.session_state['chat_history'] = []
        with st.spinner('Generating Response...'):   
            initial_response = initial_llm_chain.run(input=initial_input, language=language, scenario=scenario)
            st.session_state.chat_history.append(initial_response)
    if st.session_state.chat_history != []:
        user_message = st.text_input("Further Questions for Coding AI?", key="user_input")
        if st.button("Submit Message") and user_message:
            st.session_state['chat_history'].append(user_message)
            with st.spinner('Generating Response...'):  
                chat_response = chat_llm_chain.run(input=user_message, 
                                                    language=language, 
                                                    scenario=scenario, 
                                                    chat_history=st.session_state.chat_history)
                st.session_state['chat_history'].append(chat_response)
    display_convo()

if __name__ == '__main__':
    load_dotenv()
    main()

