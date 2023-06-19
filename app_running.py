import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os

# Sidebar Contents
with st.sidebar:
    st.title('LLM chat app')
    st.markdown('''
        ## About
        This app is an LLM-Powered chatbot built using:
        - [Streamlit](https://streamlit.io)
        - [Langchain](https://python.langchain.com/)
        - [OpenAI](https://platform.openai.com/docs/models) LLM model
    ''')
    add_vertical_space(5)
    st.write('Made by Subhendu')


def main():
    st.header("Chat with PDF")
    load_dotenv()

    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF files", type='pdf')
    print("The initail pdf loaded as: ", pdf)

    if pdf is not None:
        # Loading the PDF. 
        pdf_reader = PdfReader(pdf)

        # Extracting text from the PDF.
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Splitting text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        chunks = text_splitter.split_text(text=text)
        # st.write(chunks)

        # Embedding the chunks using OpenAI embedding
        # Whenever we upload a document for embedding that will incurred charges
        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                vectorStore = pickle.load(f)
            st.write("Embedding loaded from Disk.")
        else:
            embeddings = OpenAIEmbeddings()
            vectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(vectorStore, f)
            st.write("Embedding computation completed.")

        # Accept user questions / query
        query = st.text_input("Ask question regarding your pdf documents.")

        if query:   
            docs = vectorStore.similarity_search(query=query, k=3)
            # llm = OpenAI(temperature=0,)
            llm = OpenAI(model_name='gpt-3.5-turbo')
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)


if __name__ == '__main__':
    main()