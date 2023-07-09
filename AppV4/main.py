import os
from PIL import Image
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import streamlit as st
import openai
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from prompts import CHAT_TEMPLATE, INITIAL_TEMPLATE
from prompts import CORRECTION_CONTEXT, COMPLETION_CONTEXT, OPTIMIZATION_CONTEXT, GENERAL_ASSISTANT_CONTEXT, GENERATION_CONTEXT, COMMENTING_CONTEXT, EXPLANATION_CONTEXT


def page_config():
    st.set_page_config(page_title="GPT-4 Coding Asssistant", page_icon="computer")
    st.write(css, unsafe_allow_html=True)


def init_ses_states():
    st.session_state.setdefault('chat_history', [])
    st.session_state.setdefault('initial_input', "")


def page_title_header():
    top_image = Image.open('trippyPattern.png')
    st.image(top_image)
    st.title("GPT-4 Coding Assistant")
    st.caption("Powered by OpenAI, LangChain, Streamlit")


def sidebar():
    global language, scenario, temperature, scenario_context
    languages = ['Python', 'GoLang', 'JavaScript', 'Java', 'C', 'C++', 'C#']
    scenarios = ['General Assistant', 'Code Correction', 'Code Completion', 'Code Commenting', 'Code Optimization', 'Code Generation', 'Code Explanation']
    scenario_context_map = {
        "Code Correction": CORRECTION_CONTEXT,
        "Code Completion": COMPLETION_CONTEXT,
        "Code Optimization": OPTIMIZATION_CONTEXT,
        "General Assistant": GENERAL_ASSISTANT_CONTEXT,
        "Code Generation": GENERATION_CONTEXT,
        "Code Commenting": COMMENTING_CONTEXT,
        "Code Explanation": EXPLANATION_CONTEXT
    }

    with st.sidebar:
        with st.expander(label="Settings", expanded=True):
            language = st.selectbox(label="Language", options=languages, index=0)
            scenario = st.selectbox(label="Scenario", options=scenarios, index=0)
            scenario_context = scenario_context_map.get(scenario, "")
            temperature = st.slider(label="Temperature", min_value=0.0, max_value=1.0, value=0.5)


def display_convo():
    with st.container():
        for i, message in enumerate(reversed(st.session_state.chat_history)):
            if i % 2 == 0:
                 st.markdown(bot_template.replace("{{MSG}}", message), unsafe_allow_html=True)
            else:
                st.markdown(user_template.replace("{{MSG}}", message), unsafe_allow_html=True)


def handle_initial_submit():
    initial_template = PromptTemplate(
        input_variables=['input','language','scenario','scenario_context'],
        template= INITIAL_TEMPLATE
    )
    initial_llm_chain = create_llm_chain(prompt_template=initial_template)
    initial_input = st.text_area(label=f"User Input", height=300)
    if (st.button(f'Submit Initial Input') and initial_input):
        st.session_state.initial_input = initial_input
        st.session_state['chat_history'] = []
        with st.spinner('Generating Response...'):   
            initial_response = initial_llm_chain.run(input=initial_input, 
                                                    language=language, 
                                                    scenario=scenario,
                                                    scenario_context=scenario_context)
        st.session_state.chat_history.append(initial_response)


def handle_user_message():
    chat_template = PromptTemplate(
        input_variables=['input','user_message','language','scenario','chat_history'],
        template=CHAT_TEMPLATE
    )
    chat_llm_chain = create_llm_chain(prompt_template=chat_template)
    if st.session_state.chat_history:
        user_message = st.text_input("Further Questions for Coding AI?", key="user_input")
        if st.button("Submit Message") and user_message:
            st.session_state['chat_history'].append(user_message)
            with st.spinner('Generating Response...'):  
                chat_response = chat_llm_chain.run(input=st.session_state['initial_input'],
                                                   user_message=user_message,
                                                    language=language, 
                                                    scenario=scenario,
                                                    chat_history=st.session_state.chat_history)
                st.session_state['chat_history'].append(chat_response)


def create_llm_chain(prompt_template):
    memory = ConversationBufferMemory(input_key="input", memory_key="chat_history", )
    llm = OpenAI(temperature=temperature, model_name="gpt-4")
    return LLMChain(llm=llm, prompt=prompt_template, memory=memory)


def main():
    page_config()
    init_ses_states()
    page_title_header()
    sidebar()
    handle_initial_submit()
    handle_user_message()
    display_convo()


if __name__ == '__main__':
    load_dotenv()
    main()

