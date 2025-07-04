from loguru import logger
import atexit

import streamlit as st

from dotenv import load_dotenv

import os

from file_loader import PDFLoader, WebLoader
from rag_class import (
    AIMessage,
    HumanMessage,
    State,
    StateGraph,
    WorkFlow,
)
from vectorstore import QdrantClientManager
load_dotenv()

# Register cleanup function to close Qdrant clients on app shutdown
def cleanup_qdrant_clients():
    """Cleanup function to close all Qdrant clients on app shutdown."""
    try:
        QdrantClientManager.close_all_clients()
        logger.info("All Qdrant clients closed successfully")
    except Exception as e:
        logger.warning(f"Error during Qdrant cleanup: {e}")

atexit.register(cleanup_qdrant_clients)

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
)


def generate(workflow: StateGraph, state: State) -> State:
    """Generate response using the workflow and return updated state"""
    try:
        response_state = workflow.invoke(state)
        logger.info(
            f"Workflow returned state with {len(response_state['messages'])} messages"
        )
        return response_state
    except Exception as e:
        logger.error(f"Error during workflow execution: {e}")
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
            options=["None", "google_genai", "google_vertexai", "openai", "local_llmstudio"],
        )
        
        st.subheader("Embedding Settings")
        
        embedding_type = st.selectbox(
            "Embedding Type",
            options=["vertexai", "huggingface"],
            index=0,
            help="Choose between Google VertexAI or local HuggingFace embeddings"
        )
        
        if embedding_type == "huggingface":
            embedding_model = st.selectbox(
                "HuggingFace Model",
                options=[
                    "BAAI/bge-base-en",
                    "sentence-transformers/all-MiniLM-L6-v2",
                    "sentence-transformers/all-mpnet-base-v2",
                    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                ],
                index=0,
                help="Select HuggingFace embedding model"
            )
        else:
            embedding_model = st.selectbox(
                "VertexAI Model", 
                options=["text-embedding-004", "textembedding-gecko@003"],
                index=0,
                help="Select Google VertexAI embedding model"
            )
        
        # Enable hybrid search by default for better retrieval quality
        enable_hybrid_search = True
        
        chunk_type = st.selectbox(
            "Text Chunking Strategy",
            options=["recursive", "character"],
            index=0,
            help="Choose text splitting strategy: recursive (smart) or character (simple)"
        )
        
        use_memory = st.checkbox(
            "Use Memory Storage",
            value=True,
            help="Use in-memory storage to avoid file locking issues during development (data won't persist)"
        )

        # Cache WorkFlow in session state to avoid recreating Qdrant client on every refresh
        workflow_key = f"workflow_{model_provider}_{embedding_type}_{embedding_model}_{enable_hybrid_search}_{chunk_type}_{use_memory}"
        
        if workflow_key not in st.session_state:
            st.session_state[workflow_key] = WorkFlow(
                model_provider=model_provider,
                embedding_type=embedding_type,
                embedding_model=embedding_model,
                enable_hybrid_search=enable_hybrid_search,
                chunk_type=chunk_type,
                use_memory=use_memory
            )
        
        RAG_workflow = st.session_state[workflow_key]
        workflow = RAG_workflow.get_workflow()
        
        # Display current configuration
        with st.expander("Current Configuration", expanded=False):
            st.write(f"**Model Provider:** {model_provider}")
            st.write(f"**Embedding Type:** {embedding_type}")
            st.write(f"**Embedding Model:** {embedding_model}")
            st.write(f"**Hybrid Search:** Enabled (default)")
            st.write(f"**Chunk Type:** {chunk_type}")
            st.write(f"**Storage:** {'Memory (temporary)' if use_memory else 'Persistent'}")
            
            # Show device for HuggingFace (forced to CPU to avoid compatibility issues)
            if embedding_type == "huggingface":
                st.write(f"**Device:** CPU (stable compatibility)")
            
            # Show current sources
            sources = RAG_workflow.vector_store.get_unique_sources()
            if sources and sources != ["No sources available"]:
                st.write(f"**Sources:** {len(sources)} document(s)")
                # Show first few sources inline, with option to see all
                if len(sources) <= 3:
                    for source in sources:
                        st.write(f"  - {source}")
                else:
                    for source in sources[:2]:
                        st.write(f"  - {source}")
                    st.write(f"  - ... and {len(sources) - 2} more")
        
        # Separate expander for viewing all sources (outside the config expander)
        sources = RAG_workflow.vector_store.get_unique_sources()
        if sources and sources != ["No sources available"] and len(sources) > 3:
            with st.expander("📂 View All Sources"):
                for i, source in enumerate(sources, 1):
                    st.write(f"{i}. {source}")

        if st.button("Clear History"):
            clear_history()

        if st.button("Clear Workspace"):
            with st.spinner("Clearing workspace..."):
                try:
                    RAG_workflow.vector_store.clear_vectorstore()
                    clear_history()
                    st.success("Vectorstore cleared and reset with placeholder!")
                except Exception as e:
                    logger.error(f"Error clearing vectorstore: {e}")
                    st.error(f"Error clearing vectorstore: {str(e)}")
        
        if st.button("🔄 Force Refresh Configuration"):
            # Clear all workflow-related session state
            keys_to_remove = [key for key in st.session_state.keys() if key.startswith("workflow_")]
            for key in keys_to_remove:
                del st.session_state[key]
            st.success("Configuration refreshed! The page will reload with new settings.")
            st.rerun()

        st.subheader("Upload PDF")
        
        # Add debug mode option
        debug_mode = st.checkbox(
            "Enable Debug Mode",
            value=False,
            help="Show detailed logging information during PDF processing"
        )
        
        # Add credential path input
        # check if 'GOOGLE_APPLICATION_CREDENTIALS' is set
        if 'GOOGLE_APPLICATION_CREDENTIALS' in os.environ:
            credential_path = os.environ['GOOGLE_APPLICATION_CREDENTIALS']
            st.info(f"✅ Using credentials from environment: {credential_path}")
        else:   
            credential_path = st.text_input(
                "Google Service Account Credentials Path (Optional)",
                placeholder="/path/to/service-account-key.json",
                help="Required for advanced table extraction features (merge_span_tables, enrich). Leave empty for basic table extraction."
            )
        
        # Add custom temp directory option
        temp_dir = st.text_input(
            "Custom Temporary Directory (Optional)",
            placeholder="/tmp/pdf_processing",
            help="Specify a custom directory for temporary files. Leave empty to use system default."
        )
        
        pdf_files = st.file_uploader(
            "Upload PDF", type="pdf", accept_multiple_files=True
        )
        if pdf_files:
            # Show file information
            st.write(f"📁 **{len(pdf_files)} file(s) selected:**")
            for pdf_file in pdf_files:
                file_size = len(pdf_file.getvalue()) / (1024 * 1024)  # Size in MB
                st.write(f"- {pdf_file.name} ({file_size:.2f} MB)")
            
            # Then add a button to process files that are already uploaded
            if st.button("Process PDFs") and pdf_files:
                with st.spinner("Processing PDFs..."):
                    # Use credential_path if provided, otherwise None
                    cred_path = credential_path.strip() if credential_path and credential_path.strip() else None
                    
                    # Validate credential path if provided
                    if cred_path and not os.path.exists(cred_path):
                        st.error(f"❌ Credentials file not found: {cred_path}")
                        st.stop()
                    
                    # Validate temp directory if provided
                    temp_dir_path = temp_dir.strip() if temp_dir and temp_dir.strip() else None
                    if temp_dir_path and not os.path.exists(temp_dir_path):
                        try:
                            os.makedirs(temp_dir_path, exist_ok=True)
                            st.info(f"📁 Created temporary directory: {temp_dir_path}")
                        except Exception as e:
                            st.error(f"❌ Failed to create temp directory {temp_dir_path}: {e}")
                            st.stop()
                    
                    loader = PDFLoader(
                        credential_path=cred_path,
                        debug=debug_mode,
                        temp_dir=temp_dir_path
                    )
                    all_splits = []
                    
                    # Create a progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, pdf_file in enumerate(pdf_files):
                        try:
                            # Update progress
                            progress = (i + 1) / len(pdf_files)
                            progress_bar.progress(progress)
                            status_text.text(f"Processing {pdf_file.name}... ({i+1}/{len(pdf_files)})")
                            
                            splits = loader.load(
                                pdf_file=pdf_file, original_filename=pdf_file.name
                            )
                            all_splits.extend(splits)
                            logger.info(
                                f"Loaded {len(splits)} documents from {pdf_file.name}"
                            )
                            
                            # Show success for individual file
                            if debug_mode:
                                st.success(f"✅ {pdf_file.name}: {len(splits)} documents extracted")
                                
                        except Exception as e:
                            logger.error(f"Error loading PDF {pdf_file.name}: {str(e)}")
                            st.error(f"❌ Error processing {pdf_file.name}: {str(e)}")

                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    
                    if all_splits:
                        RAG_workflow.vector_store.add_documents(documents=all_splits)
                        
                        # Show summary
                        text_docs = len([doc for doc in all_splits if doc.metadata.get("type") == "text"])
                        table_docs = len([doc for doc in all_splits if doc.metadata.get("type") == "table"])
                        
                        st.success(
                            f"🎉 Successfully processed {len(pdf_files)} PDF(s)!\n\n"
                            f"📄 **Total documents added:** {len(all_splits)}\n"
                            f"📝 **Text documents:** {text_docs}\n"
                            f"📊 **Table documents:** {table_docs}"
                        )
                    else:
                        st.error("❌ No documents were extracted from the PDF files.")

        st.subheader("Add Web Content")
        urls = st.text_area("Enter website URLs (one per line):")
        if urls:
            if st.button("Process URLs") and urls:
                with st.spinner("Processing URLs..."):
                    urls_list = [url.strip() for url in urls.split("\n") if url.strip()]
                    logger.info(
                        f"Processing {len(urls_list)} URLs: {', '.join(urls_list)}"
                    )
                    if urls_list:
                        loader = WebLoader()

                        try:
                            all_splits = loader.load(urls_list)
                            logger.info(
                                f"Loading {len(all_splits)} documents from URLs"
                            )
                            RAG_workflow.vector_store.add_documents(
                                documents=all_splits
                            )
                        except Exception as e:
                            logger.error(f"Error loading URL content: {str(e)}")
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
