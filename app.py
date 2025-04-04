import streamlit as st
from crawl import *
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
)

def generate_text(prompt: str) -> str:
    response = "This is a placeholder response."
    return response



def main():
    st.title("Chatbot")
    st.write("This is a simple chatbot application.")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    with st.sidebar:
        st.header("Settings")
        url = st.text_input("Search URL:")
        flag = check_valid_url(url)
        if flag:
            try:
                all_splits = load_web_content(url)
                logging.info(f"Loaded {len(all_splits)} documents from {url}")
                st.logging(f"Loaded {len(all_splits)} documents")
                ### code vector .add
            except Exception as e:
                logging.error(f"Error loading URL: {str(e)}")
                st.error(f"Error loading URL: {str(e)}")
        else:
            st.error("Invalid URL provided.")
            ### code
        if st.button("Upload PDF"):
            pdf = st.file_uploader("Upload PDF", type="pdf")
            ### code
        
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    if prompt := st.chat_input("Ask me anything!"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_text(prompt)
                if response:
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.markdown(response)

    
if __name__ == "__main__":
    main()
        