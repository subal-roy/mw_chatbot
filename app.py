import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import time
from retriever_singleton import get_hybrid_retriever
from search_fallback import google_search
from langchain.schema import Document
import os
import logging
import cohere

logger = logging.getLogger(__name__)

load_dotenv()

co = cohere.Client(os.getenv("COHERE_API_KEY"))

def rerank_documents(docs, query, top_k=4):
    rerank_response = co.rerank(
        query=query,
        documents=[doc.page_content for doc in docs],
        top_n = top_k,
        model="rerank-english-v3.0"
    )
    results = rerank_response.results
    reranked_docs = [docs[result.index] for result in results]
    return reranked_docs

      
def get_conversational_chain():
    prompt_template = """
                        You are Mediusware Ltd.'s internal HR assistant. Answer based only on the context as if it's your own knowledge.

                        - Use a natural, first-person tone (e.g., “Based on my knowledge”). But don't always include it. Sometimes you may use "We", "Our" -based on the question.
                        - Don't mention context, documents, or sources.
                        - Flexibly interpret the question and align your answer with the meaning of the context, even if terms differ.
                        - If nothing relevant is found, say:
                        “Sorry, I don't have enough information. Please contact Mediusware HR or visit [mediusware.com](https://mediusware.com).”

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


def process_user_input(user_question, indicator_placeholder):
    try:
        hybrid_retriever = get_hybrid_retriever()

        docs = hybrid_retriever.get_relevant_documents(user_question)
        logger.info(docs)
        top_docs = rerank_documents(docs, user_question, 4)

        #print top docs
        for doc in top_docs:
            logger.info(doc)

        chain = get_conversational_chain()

        response = chain({"input_documents": top_docs, "question": user_question}, return_only_outputs=True)
        assistant_response = response["output_text"]

        fallback_trigger = "Sorry, I don't have enough information"

        if fallback_trigger in assistant_response:
            logger.info(f"Inside fallback trigger")
            indicator_placeholder.markdown(
                f"<p style='font-size: 16px; color: gray;'>Searching on the web...</p>",
                unsafe_allow_html=True
            )
            search_results = google_search(user_question)
            search_docs = []
            if search_results:
                search_docs = [
                    Document(
                        page_content=entry["snippet"],
                         metadata={"source":entry["link"], "title": entry["title"]}
                    )
                    for entry in search_results
                ]
            if search_docs:
                logger.info(f"Search docs found: , {search_docs}")
                search_docs = rerank_documents(search_docs, user_question, 4)
                search_response = chain(
                    {"input_documents": search_docs, "question": user_question},
                    return_only_outputs=True
                )
                assistant_response = search_response["output_text"]
            indicator_placeholder.empty()
            
        return assistant_response

    except Exception as e:
        logger.error(f"Error processing user input: {e}")
        return f"Sorry, I couldn't generate a response due to an error: {str(e)}"


def main():
    st.set_page_config(page_title="MW Chatbot", layout="centered")
    st.markdown("<h1 style='text-align: center;'>MW Chatbot 🤖</h1>", unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processing" not in st.session_state:
        st.session_state.processing = False

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Get user input
    user_question = st.chat_input("Ask a question about company policies, HR, holidays, etc.")
    if user_question and not st.session_state.processing:
        st.session_state.processing = True
        st.session_state.messages.append({"role": "user", "content": user_question})
        st.rerun()
    
    # Process the latest question if processing
    if st.session_state.processing and st.session_state.messages:
        latest_message = st.session_state.messages[-1]
        if latest_message["role"] == "user":
            with st.chat_message("assistant"):
                indicator_placeholder = st.empty()
                message_placeholder = st.empty()
                indicator_placeholder.markdown(
                    f"<p style='font-size: 16px; color: gray;'>Thinking...</p>",
                    unsafe_allow_html=True
                )
                assistant_response = process_user_input(latest_message["content"], indicator_placeholder)
                indicator_placeholder.empty()

                # Stream the response
                full_response = ""
                for char  in assistant_response:
                    full_response += char
                    message_placeholder.markdown(full_response )
                    time.sleep(0.01)

                # Append assistant response to messages
                st.session_state.messages.append({"role": "assistant", "content": full_response})

            # Reset processing flag
            st.session_state.processing = False
            st.rerun()

if __name__ == "__main__":
    main()