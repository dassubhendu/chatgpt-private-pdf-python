import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
# from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
    chunks = text_splitter.split_text(text=text)
    return chunks


def get_vectorStore(chunks, pdf_docs):
    doc_name_list = []
    for pdf_name in pdf_docs:
        store_name = pdf_name.name[:-4]
        doc_name_list.append(store_name)

    for item in doc_name_list:
        print(item)
        print(f"trained_data/{item}.pkl")
        if os.path.exists(f"trained_data/{item}.pkl"):
            with open(f"trained_data/{item}.pkl", "rb") as f:
                vectorStore = pickle.load(f)
            st.write("Embedding loaded from Disk for " + item + ".")
        else:
            embeddings = OpenAIEmbeddings()
            vectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"trained_data/{item}.pkl", "wb") as f:
                pickle.dump(vectorStore, f)
            st.write("Embedding computation completed for " + item + ".")
    return vectorStore


def get_conversation_chain(vectorStore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorStore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userInput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":printer:")

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None


    st.header("Chat with multiple PDfs :printer:")
    user_question = st.text_input("Ask a question about your documents.")
    if user_question:
        handle_userInput(user_question)

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload your PDF files here and click on Process.", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):

                # ====================== Get PDF text ======================= #
                raw_text = get_pdf_text(pdf_docs)

                # ====================== Get the text chunks ================ #
                text_chunks = get_text_chunks(raw_text)

                # ====================== Create vector store ================ #
                vectorStore = get_vectorStore(text_chunks, pdf_docs)

                # =================== Create a conversation chain =========== #
                st.session_state.conversation = get_conversation_chain(vectorStore)
                


if __name__ == '__main__':
    main()