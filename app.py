import logging

import streamlit as st

from crawl import check_valid_url, load_web_content
from rag import AIMessage, HumanMessage, State, SystemMessage, get_workflow, StateGraph
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
def load_workflow() -> StateGraph:
    try:
        workflow = get_workflow()
        return workflow
    except Exception as e:
        logging.error(f"Error loading workflow: {e}")
        st.error(f"Error loading workflow: {e}")
        return None



# def generate_text(prompt: str) -> str:
#     workflow = load_workflow()

#     # Limit number of chat history
#     messages = st.session_state.langchain_messages[-MAX_HISTORY:].copy()

#     # ensure that user prompt in context
#     if not messages or messages[-1].content != prompt:
#         messages.append(HumanMessage(content=prompt))

#     state = State({"messages": messages, "rewrite_times": 0})

#     try:
#         logging.info(f"Running workflow with prompt: {prompt}")
#         result = workflow.invoke(state)

#         st.session_state.last_workflow_state = result

#         if result and "messages" in result and result["messages"]:
#             response = result["messages"][-1]

#             if hasattr(response, "content"):
#                 return response.content
#             return str(response)
#         else:
#             return "I couldn't generate a response."
#     except Exception as e:
#         logging.error(f"Error during workflow execution: {e}")
#         return f"An error occurred: {str(e)}"

def generate(workflow: StateGraph, state: State) -> State:
    """Generate response using the workflow and return updated state"""
    try:
        response_state = workflow.invoke(state)
        logging.info(f"Workflow returned state with {len(response_state['messages'])} messages")
        return response_state
    except Exception as e:
        logging.error(f"Error during workflow execution: {e}")
        state["messages"].append(AIMessage(content=f"An error occurred: {str(e)}"))
        return state


def main():
    st.title("Chatbot")
    st.write("This is a simple chatbot application.")
    
    workflow = load_workflow()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "state" not in st.session_state:
        st.session_state.state = State({
            "messages": [],
            "rewrite_times": 0,
        })

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
                
        if st.button("Clear History"):
            # st.session_state.messages = []
            # st.session_state.langchain_messages = []
            # st.session_state.last_workflow_state = None
            st.success("History cleared!")
            

            
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
                    response_content = last_message.content if hasattr(last_message, "content") else str(last_message)
                    
                    # Update UI message history
                    st.session_state.messages.append({"role": "assistant", "content": response_content})
                    
                    # Display the message
                    st.markdown(response_content)


if __name__ == "__main__":
    main()
