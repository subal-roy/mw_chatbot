import streamlit as st
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import time

from data_processor import process_data

load_dotenv()

PDF_DIR = "data"

def get_conversational_chain():
    prompt_template = """
                        You are a helpful and professional AI assistant representing Mediusware Ltd.

                        Answer the user's question using **only** the information provided in the context below.

                        If the context does not contain a sufficient answer, just say "I don't have enough information to answer that. Please visit [ mediusware.com ]( https://mediusware.com ) to learn more."

                        ---
                        Context:
                        {context}
                        ---

                        User Question:
                        {question}

                        Your Answer:
                        """

    model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash", temperature = 0.3)
    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type = "stuff", prompt = prompt)
    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    st.session_state.messages.append({"role": "user", "content": user_question})

    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        try:
            # Load FAISS index
            new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            docs_with_scores = new_db.similarity_search_with_score(user_question, k=10)
            docs = [doc for doc, score in docs_with_scores if score < 0.9]
            top_docs = docs[:3]

            chain = get_conversational_chain()

            response = chain({"input_documents": top_docs, "question": user_question}, return_only_outputs=True)
            assistant_response = response["output_text"]

            full_response = ""
            for chunk in assistant_response.split():
                full_response += chunk + " "
                message_placeholder.markdown(full_response + "▌")
                time.sleep(0.05)
            message_placeholder.markdown(full_response)

        except Exception as e:
            st.error(f"An error occurred while generating a response: {e}")
            st.exception(e)
            full_response = "Sorry, I couldn't generate a response due to an error."
            st.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})


def main():
    st.set_page_config(page_title="MW Chatbot", layout="centered")
    st.markdown("<h1 style='text-align: center;'>MW Chatbot 🤖</h1>", unsafe_allow_html=True)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Load and process knowledgebase once (on first run only)
    if not os.path.exists("faiss_index"):
        process_data(pdf_dir=PDF_DIR, index_dir="faiss_index")

    # Get user input
    user_question = st.chat_input("Ask a question about company policies, HR, holidays, etc.")
    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    main()