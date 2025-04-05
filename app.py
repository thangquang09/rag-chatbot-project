import logging

import streamlit as st

from crawl import check_valid_url, load_web_content
from rag import AIMessage, HumanMessage, State, SystemMessage, get_workflow

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

def convert_to_langchain_messages(streamlit_messages):
    """Convert Streamlit message format to Langchain message objects"""
    langchain_messages = []
    for msg in streamlit_messages:
        if msg["role"] == "user":
            langchain_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            langchain_messages.append(AIMessage(content=msg["content"]))
        elif msg["role"] == "system":
            langchain_messages.append(SystemMessage(content=msg["content"]))
    return langchain_messages


def convert_to_streamlit_messages(langchain_messages):
    """Convert Langchain message objects to Streamlit message format"""
    streamlit_messages = []
    for msg in langchain_messages:
        if isinstance(msg, HumanMessage):
            streamlit_messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            streamlit_messages.append({"role": "assistant", "content": msg.content})
        elif isinstance(msg, SystemMessage):
            streamlit_messages.append({"role": "system", "content": msg.content})
    return streamlit_messages


def generate_text(prompt: str, history=None) -> str:
    workflow = load_workflow()

    if history:
        messages = convert_to_langchain_messages(history)
        if not messages or messages[-1].content != prompt:
            messages.append(HumanMessage(content=prompt))
    else:
        messages = [HumanMessage(content=prompt)]

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
        return "An error occurred while generating the response."


def main():
    st.title("Chatbot")
    st.write("This is a simple chatbot application.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

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
            st.subheader("Session Messages")
            st.json(st.session_state.messages)

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
                response = generate_text(prompt, st.session_state.messages)
                if response:
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )
                    st.markdown(response)


if __name__ == "__main__":
    main()
