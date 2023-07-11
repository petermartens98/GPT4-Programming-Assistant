import os
from PIL import Image
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import streamlit as st
from streamlit_extras.let_it_rain import rain
import openai
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain, ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from htmlTemplates import css, bot_template, user_template
from prompts import CHAT_TEMPLATE, INITIAL_TEMPLATE
from prompts import CORRECTION_CONTEXT, COMPLETION_CONTEXT, OPTIMIZATION_CONTEXT, GENERAL_ASSISTANT_CONTEXT, \
    GENERATION_CONTEXT, COMMENTING_CONTEXT, EXPLANATION_CONTEXT, LEETCODE_CONTEXT, SHORTENING_CONTEXT


def page_config():
    st.set_page_config(page_title="GPT-4 Coding Asssistant", page_icon="computer")
    st.write(css, unsafe_allow_html=True)


def init_ses_states():
    st.session_state.setdefault('chat_history', [])
    st.session_state.setdefault('initial_input', "")
    st.session_state.setdefault("initial_context", "")
    st.session_state.setdefault('scenario_context', "")
    st.session_state.setdefault('docs_processed', False)
    st.session_state.setdefault('docs_chain', None)
    st.session_state.setdefault('user_authenticated', False)


def page_title_header():
    top_image = Image.open('trippyPattern.png')
    st.image(top_image)
    st.title("GPT-4 Coding Assistant")
    st.caption("Powered by OpenAI, LangChain, Streamlit")


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def create_retrieval_chain(vectorstore):
    llm = ChatOpenAI(temperature=temperature, model_name=model)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def create_llm_chain(prompt_template):
    memory = ConversationBufferMemory(input_key="input", memory_key="chat_history", )
    llm = OpenAI(temperature=temperature, model_name=model)
    return LLMChain(llm=llm, prompt=prompt_template, memory=memory)


def sidebar():
    global language, scenario, temperature, model, scenario_context, libraries, pdf_docs
    languages = sorted(['Python', 'GoLang','TypeScript', 'JavaScript', 'Java', 'C', 'C++', 'C#', 'R', 'SQL'])
    python_libs = sorted(["Pandas",'Numpy','Scipy','Scikit-Learn','PyTorch','TensorFlow','Streamlit','Flask','FastAPI'])
    scenarios = ['General Assistant', 'Code Correction', 'Code Completion', 'Code Commenting', 'Code Optimization', 
                 'Code Shortener','Code Generation', 'Code Explanation', 'LeetCode Solver']
    scenario_context_map = {
        "Code Correction": CORRECTION_CONTEXT,
        "Code Completion": COMPLETION_CONTEXT,
        "Code Optimization": OPTIMIZATION_CONTEXT,
        "General Assistant": GENERAL_ASSISTANT_CONTEXT,
        "Code Generation": GENERATION_CONTEXT,
        "Code Commenting": COMMENTING_CONTEXT,
        "Code Explanation": EXPLANATION_CONTEXT,
        "LeetCode Solver": LEETCODE_CONTEXT,
        "Code Shortener": SHORTENING_CONTEXT,
    }

    with st.sidebar:
        with st.expander(label="Coding Settings", expanded=True):
            language = st.selectbox(label="Language", options=languages, index=0)
            if language == "Python":
                libraries = st.multiselect(label="Libraries",options=python_libs)
            else:
                libraries=""
            scenario = st.selectbox(label="Scenario", options=scenarios, index=0)
            scenario_context = scenario_context_map.get(scenario, "")
        
        with st.expander("Chatbot Settings", expanded=True):
            model = st.selectbox("Language Model", options=['gpt-4','gpt-4-0613','gpt-3.5-turbo'])
            temperature = st.slider(label="Temperature", min_value=0.0, max_value=1.0, value=0.5)
        
        with st.expander(label="User Docs",expanded=True):
            pdf_docs = st.file_uploader("Upload Docs Here",type=['PDF'],accept_multiple_files=True)
            if pdf_docs:
                if st.button("Process Docs"):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                    initial_llm_chain = create_retrieval_chain(vectorstore=vectorstore)
                    st.session_state.docs_chain = initial_llm_chain
                    st.session_state.docs_processed = True
                    st.write("Docs Succesfully Processed")
            else:
                st.session_state.docs_processed = False


def display_convo():
    with st.container():
        for i, message in enumerate(reversed(st.session_state.chat_history)):
            if i % 2 == 0:
                 st.markdown(bot_template.replace("{{MSG}}", message), unsafe_allow_html=True)
            else:
                st.markdown(user_template.replace("{{MSG}}", message), unsafe_allow_html=True)


def handle_initial_submit():
    global code_input, code_context
    initial_template = PromptTemplate(
        input_variables=['input','language','scenario','scenario_context','code_context','libraries'],
        template= INITIAL_TEMPLATE
    )
    if pdf_docs and st.session_state.docs_processed:
        initial_llm_chain =st.session_state.docs_chain
    else:
        initial_llm_chain = create_llm_chain(prompt_template=initial_template)
    code_input = st.text_area(label=f"User Code", height=300)
    code_context = st.text_area(label="Additional Context", height=60)
    if (st.button(f'Submit Initial Input') and (code_input or code_context)):
        st.session_state.initial_input = code_input
        st.session_state.initial_context = code_context
        st.session_state['chat_history'] = []
        with st.spinner('Generating Response...'):
            if st.session_state.docs_processed:
                llm_input = initial_template.format(input=code_input,
                                                code_context=code_context, 
                                                language=language, 
                                                scenario=scenario,
                                                scenario_context=scenario_context,
                                                libraries=libraries)
                initial_response = initial_llm_chain({'question':llm_input})['answer']
            else: 
                initial_response = initial_llm_chain.run(input=code_input,
                                                        code_context=code_context, 
                                                        language=language, 
                                                        scenario=scenario,
                                                        scenario_context=scenario_context,
                                                        libraries=libraries)
        st.session_state.chat_history.append(f"AI: {initial_response}")


def handle_user_message():
    chat_template = PromptTemplate(
        input_variables=['input','code_context','user_message','language','scenario','scenario_context','chat_history','libraries'],
        template=CHAT_TEMPLATE
    )
    if st.session_state.docs_processed:
        chat_llm_chain = st.session_state.docs_chain
    else:
        chat_llm_chain = create_llm_chain(prompt_template=chat_template)
    if st.session_state.chat_history:
        user_message = st.text_area("Further Questions for Coding AI?", key="user_input", height=60)
        if st.button("Submit Message") and user_message:
            st.session_state['chat_history'].append(f"USER: {user_message}\n")
            with st.spinner('Generating Response...'):
                if st.session_state.docs_processed: # Fix this
                    chat_input = chat_template.format(input=st.session_state.initial_input,
                                                    code_context=st.session_state.initial_context,
                                                    user_message=user_message, 
                                                    language=language, 
                                                    scenario=scenario,
                                                    scenario_context=scenario_context,
                                                    libraries=libraries)
                    chat_response = chat_llm_chain({'question':chat_input})['answer']
                else:
                    chat_response = chat_llm_chain.run(input=st.session_state['initial_input'],
                                                        user_message=user_message,
                                                        code_context=st.session_state.initial_context,
                                                        language=language, 
                                                        scenario=scenario,
                                                        scenario_context=scenario_context,
                                                        chat_history=st.session_state.chat_history,
                                                        libraries=libraries)
                st.session_state['chat_history'].append(f"AI: {chat_response}\n")


def display_file_code(filename):
    with open(filename, "r") as file:
        with st.expander(filename, expanded=False):
            st.code(file.read(), language='python')


def display_all_code():
    st.subheader("GitHub Link")
    st.write("https://github.com/petermartens98/GPT4-Programming-Assistant")
    st.subheader("Source Code")
    files = ["main.py", "htmlTemplates.py", "prompts.py", "requirements.txt"]
    for file in files:
        display_file_code(file)


def main():
    page_config()
    init_ses_states()
    page_title_header()
    sidebar()
    if st.session_state.docs_processed:
        deploy_tab, code_tab, docs_tab = st.tabs(['Deployment','Source Code','Doc Analysis'])
        with docs_tab:
            st.caption("Coming Soon")
    else:
        deploy_tab, code_tab = st.tabs(['Deployment','Source Code'])
    with deploy_tab:
        handle_initial_submit()
        handle_user_message()
        display_convo()
    with code_tab: 
        display_all_code()


if __name__ == '__main__':
    load_dotenv()
    main()

