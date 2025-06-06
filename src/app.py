import logging

import streamlit as st

from file_loader import PDFLoader, WebLoader
from rag_class import (
    AIMessage,
    HumanMessage,
    State,
    StateGraph,
    WorkFlow,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
)


def generate(workflow: StateGraph, state: State) -> State:
    """Generate response using the workflow and return updated state"""
    try:
        response_state = workflow.invoke(state)
        logging.info(
            f"Workflow returned state with {len(response_state['messages'])} messages"
        )
        return response_state
    except Exception as e:
        logging.error(f"Error during workflow execution: {e}")
        state["messages"].append(AIMessage(content=f"An error occurred: {str(e)}"))
        return state


def clear_history():
    if "messages" in st.session_state:
        st.session_state.messages = []
    if "state" in st.session_state:
        st.session_state.state = State(
            {
                "messages": [],
                "rewrite_times": 0,
            }
        )
    st.success("History cleared!")


def main():
    st.title("Retrieval-Augmented Generation (RAG) Chatbot")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "state" not in st.session_state:
        st.session_state.state = State(
            {
                "messages": [],
                "rewrite_times": 0,
            }
        )

    with st.sidebar:
        st.header("Settings")

        model_provider = st.selectbox(
            "Select a model",
            options=["None", "google_genai", "openai", "local_llmstudio"],
        )

        RAG_workflow = WorkFlow(model_provider=model_provider)
        workflow = RAG_workflow.get_workflow()

        if st.button("Clear History"):
            clear_history()

        if st.button("Clear Workspace"):
            with st.spinner("Clearing workspace..."):
                try:
                    RAG_workflow.vector_store.clear_vectorstore()
                    clear_history()
                    st.success("Vectorstore cleared and reset with placeholder!")
                except Exception as e:
                    logging.error(f"Error clearing vectorstore: {e}")
                    st.error(f"Error clearing vectorstore: {str(e)}")

        st.subheader("Upload PDF")
        pdf_files = st.file_uploader(
            "Upload PDF", type="pdf", accept_multiple_files=True
        )
        if pdf_files:
            # Then add a button to process files that are already uploaded
            if st.button("Process PDFs") and pdf_files:
                with st.spinner("Processing PDFs..."):
                    loader = PDFLoader()
                    all_splits = []
                    for pdf_file in pdf_files:
                        try:
                            splits = loader.load(
                                pdf_file=pdf_file, original_filename=pdf_file.name
                            )
                            all_splits.extend(splits)
                            logging.info(
                                f"Loaded {len(splits)} documents from {pdf_file.name}"
                            )
                        except Exception as e:
                            logging.error(f"Error loading PDF: {str(e)}")
                            st.error(f"Error loading PDF: {str(e)}")

                    if all_splits:
                        RAG_workflow.vector_store.add_documents(documents=all_splits)
                        st.success("Added document to vector store!")

        st.subheader("Add Web Content")
        urls = st.text_area("Enter website URLs (one per line):")
        if urls:
            if st.button("Process URLs") and urls:
                with st.spinner("Processing URLs..."):
                    urls_list = [url.strip() for url in urls.split("\n") if url.strip()]
                    logging.info(
                        f"Processing {len(urls_list)} URLs: {', '.join(urls_list)}"
                    )
                    if urls_list:
                        loader = WebLoader()

                        try:
                            all_splits = loader.load(urls_list)
                            logging.info(
                                f"Loading {len(all_splits)} documents from URLs"
                            )
                            RAG_workflow.vector_store.add_documents(
                                documents=all_splits
                            )
                        except Exception as e:
                            logging.error(f"Error loading URL content: {str(e)}")
                            st.error(f"Error loading URL: {str(e)}")

                        st.success("Added document to vector store!")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input("Ask me anything!"):
        # add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # add user message to State
        st.session_state.state["messages"].append(HumanMessage(content=prompt))

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Get updated state from workflow
                updated_state = generate(workflow, st.session_state.state.copy())

                if updated_state and updated_state["messages"]:
                    # Update the session state with new state
                    st.session_state.state = updated_state

                    # Get the last AI message to display
                    last_message = updated_state["messages"][-1]
                    response_content = (
                        last_message.content
                        if hasattr(last_message, "content")
                        else str(last_message)
                    )

                    # Update UI message history
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response_content}
                    )

                    # Display the message
                    st.markdown(response_content)


if __name__ == "__main__":
    main()
