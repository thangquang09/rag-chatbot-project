import logging

import streamlit as st

from crawl import check_valid_url, load_web_content
from rag import AIMessage, HumanMessage, State, SystemMessage, get_workflow
from setting import MAX_HISTORY

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
)


@st.cache_resource
def load_workflow():
    try:
        workflow = get_workflow()
        return workflow
    except Exception as e:
        logging.error(f"Error loading workflow: {e}")
        st.error(f"Error loading workflow: {e}")
        return None


def add_message(role, content):
    # Thêm vào định dạng Streamlit
    st.session_state.messages.append({"role": role, "content": content})

    # Thêm vào định dạng Langchain
    if role == "user":
        st.session_state.langchain_messages.append(HumanMessage(content=content))
    elif role == "assistant":
        st.session_state.langchain_messages.append(AIMessage(content=content))
    elif role == "system":
        st.session_state.langchain_messages.append(SystemMessage(content=content))


def generate_text(prompt: str) -> str:
    workflow = load_workflow()

    # Limit number of chat history
    messages = st.session_state.langchain_messages[-MAX_HISTORY:].copy()

    # ensure that user prompt in context
    if not messages or messages[-1].content != prompt:
        messages.append(HumanMessage(content=prompt))

    state = State({"messages": messages, "rewrite_times": 0})

    try:
        logging.info(f"Running workflow with prompt: {prompt}")
        result = workflow.invoke(state)

        st.session_state.last_workflow_state = result

        if result and "messages" in result and result["messages"]:
            response = result["messages"][-1]

            if hasattr(response, "content"):
                return response.content
            return str(response)
        else:
            return "I couldn't generate a response."
    except Exception as e:
        logging.error(f"Error during workflow execution: {e}")
        return f"An error occurred: {str(e)}"


def main():
    st.title("Chatbot")
    st.write("This is a simple chatbot application.")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "langchain_messages" not in st.session_state:
        st.session_state.langchain_messages = []

    with st.sidebar:
        # st.header("Settings")
        # url = st.text_input("Search URL:")
        # flag = check_valid_url(url)
        # if flag:
        #     try:
        #         all_splits = load_web_content(url)
        #         logging.info(f"Loaded {len(all_splits)} documents from {url}")
        #         st.logging(f"Loaded {len(all_splits)} documents")
        #         ### code vector .add
        #     except Exception as e:
        #         logging.error(f"Error loading URL: {str(e)}")
        #         st.error(f"Error loading URL: {str(e)}")
        # else:
        #     st.error("Invalid URL provided.")
        #     ### code
        # if st.button("Upload PDF"):
        #     pdf = st.file_uploader("Upload PDF", type="pdf")
        #     ### code
        st.header("Debug Information")
        if st.checkbox("Show debug info"):
            st.subheader("Streamlit Messages")
            st.write(f"Count: {len(st.session_state.messages)}")
            st.json(st.session_state.messages)

            st.subheader("Langchain Messages")
            st.write(f"Count: {len(st.session_state.langchain_messages)}")
            for i, msg in enumerate(st.session_state.langchain_messages):
                st.write(
                    f"{i + 1}. Type: {type(msg).__name__}, Content: {msg.content[:50]}..."
                )

            st.subheader("Last Workflow State")
            if "last_workflow_state" in st.session_state:
                st.write(st.session_state.last_workflow_state)
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input("Ask me anything!"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_text(prompt)
                if response:
                    add_message("assistant", response)
                    st.markdown(response)


if __name__ == "__main__":
    main()
