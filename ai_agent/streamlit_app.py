import streamlit as st
import os
from typing import List, Dict, Any
import pandas as pd
import plotly.express as px
from datetime import datetime

# Import agent components
from agent import SmartAgent, Config

from dotenv import load_dotenv
load_dotenv()

def get_vector_db_documents(vector_db):
    """Simple function to get documents from vector database"""
    try:
        if hasattr(vector_db, 'vectorstore') and vector_db.vectorstore:
            if hasattr(vector_db.vectorstore, '_collection'):
                collection = vector_db.vectorstore._collection
                results = collection.get()
                
                documents = []
                if results and 'documents' in results:
                    for i, content in enumerate(results['documents']):
                        doc = {'content': content}
                        if 'metadatas' in results and i < len(results['metadatas']):
                            metadata = results['metadatas'][i] or {}
                            doc.update(metadata)
                        documents.append(doc)
                return documents
        return []
    except:
        return []

def render_vector_db_tab():
    """Simple vector database viewer"""
    st.subheader("📚 Vector Database Contents")
    
    stats = st.session_state.agent.vector_db.get_collection_stats()
    doc_count = stats.get("document_count", 0)
    
    st.metric("Total Documents", doc_count)
    
    if doc_count == 0:
        st.info("No documents in database")
        return
    
    # Get and display documents
    documents = get_vector_db_documents(st.session_state.agent.vector_db)
    
    if documents:
        search = st.text_input("🔍 Search documents")
        
        for i, doc in enumerate(documents):
            if search and search.lower() not in doc['content'].lower():
                continue
                
            with st.expander(f"Document {i+1}: {doc['content'][:80]}..."):
                st.write("**Content:**")
                st.write(doc['content'])
                
                if any(k != 'content' for k in doc.keys()):
                    st.write("**Metadata:**")
                    for k, v in doc.items():
                        if k != 'content':
                            st.write(f"- {k}: {v}")
    else:
        st.warning("Could not retrieve documents")

def init_session_state():
    """Initialize Streamlit session state"""
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "pending_approval" not in st.session_state:
        st.session_state.pending_approval = None
    if "config" not in st.session_state:
        st.session_state.config = Config()

def streamlit_approval_callback(name: str, description: str, args: tuple, kwargs: dict) -> bool:
    """Streamlit-specific approval callback"""
    st.session_state.pending_approval = {
        "name": name,
        "description": description,
        "args": args,
        "kwargs": kwargs,
        "approved": None
    }
    return False  # Will be handled by the UI

def setup_page():
    """Setup Streamlit page configuration"""
    config = st.session_state.config
    
    st.set_page_config(
        page_title=config.get("streamlit.page_title", "Smart AI Agent"),
        page_icon=config.get("streamlit.page_icon", "🤖"),
        layout=config.get("streamlit.layout", "wide")
    )

def render_sidebar():
    """Render the sidebar with configuration and stats"""
    st.sidebar.title("🤖 Agent Configuration")
    
    # LLM Selection
    llm_type = st.sidebar.selectbox(
        "Select LLM",
        ["llama", "claude"],
        index=0 if st.session_state.config.get("llm.default_type") == "llama" else 1
    )
    
    # API Key input for Claude
    if llm_type == "claude":
        api_key = st.sidebar.text_input(
            "Anthropic API Key",
            type="password",
            value=os.getenv("ANTHROPIC_API_KEY", "")
        )
        if api_key:
            os.environ["ANTHROPIC_API_KEY"] = api_key
    
    # Initialize Agent button
    if st.sidebar.button("Initialize Agent"):
        try:
            # Update config
            st.session_state.config.config["llm"]["default_type"] = llm_type
            
            # Create agent with Streamlit approval callback
            st.session_state.agent = SmartAgent(
                config=st.session_state.config,
                approval_callback=streamlit_approval_callback
            )
            
            # Initialize with sample data
            st.session_state.agent.initialize_with_sample_data()
            
            st.sidebar.success("Agent initialized successfully!")
        except Exception as e:
            st.sidebar.error(f"Error initializing agent: {e}")
    
    # Agent Stats
    if st.session_state.agent:
        st.sidebar.subheader("📊 Agent Stats")
        stats = st.session_state.agent.get_stats()
        
        st.sidebar.metric("LLM Type", stats["llm_type"])
        st.sidebar.metric("Tools Available", stats["tools_count"])
        
        if "document_count" in stats["vector_db_stats"]:
            st.sidebar.metric("Documents in DB", stats["vector_db_stats"]["document_count"])

def render_approval_dialog():
    """Render approval dialog if pending"""
    if st.session_state.pending_approval and st.session_state.pending_approval["approved"] is None:
        approval = st.session_state.pending_approval
        
        st.warning("🤖 Agent Approval Required")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Action:** {approval['name']}")
            st.write(f"**Description:** {approval['description']}")
            st.write(f"**Arguments:** {approval['args']}")
            st.write(f"**Keywords:** {approval['kwargs']}")
        
        with col2:
            col_approve, col_deny = st.columns(2)
            
            with col_approve:
                if st.button("✅ Approve", key="approve_btn"):
                    st.session_state.pending_approval["approved"] = True
                    st.rerun()
            
            with col_deny:
                if st.button("❌ Deny", key="deny_btn"):
                    st.session_state.pending_approval["approved"] = False
                    st.rerun()

def render_chat_interface():
    """Render the main chat interface"""
    st.title("💬 Smart AI Agent Chat")
    
    if not st.session_state.agent:
        st.info("👈 Please initialize the agent using the sidebar configuration.")
        return
    
    # Display chat history
    chat_container = st.container()
    
    with chat_container:
        for i, (role, message) in enumerate(st.session_state.chat_history):
            if role == "user":
                with st.chat_message("user"):
                    st.write(message)
            else:
                with st.chat_message("assistant"):
                    st.write(message)
    
    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        # Add user message to history
        st.session_state.chat_history.append(("user", prompt))
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Convert chat history to LangChain format
                from langchain.schema import HumanMessage, AIMessage
                
                langchain_history = []
                for role, msg in st.session_state.chat_history[:-1]:  # Exclude current message
                    if role == "user":
                        langchain_history.append(HumanMessage(content=msg))
                    else:
                        langchain_history.append(AIMessage(content=msg))
                
                # Get response from agent
                response = st.session_state.agent.chat(prompt, langchain_history)
                
                st.write(response)
                
                # Add assistant response to history
                st.session_state.chat_history.append(("assistant", response))
        
        # Keep chat history manageable
        max_history = st.session_state.config.get("agent.max_chat_history", 20)
        if len(st.session_state.chat_history) > max_history:
            st.session_state.chat_history = st.session_state.chat_history[-max_history:]

def render_tools_page():
    """Render tools and database management page"""
    st.title("🔧 Tools & Database Management")
    
    if not st.session_state.agent:
        st.info("👈 Please initialize the agent using the sidebar configuration.")
        return
    
    # Simple tabs - just add one more
    tools_tab, db_tab, add_tab = st.tabs(["🛠️ Tools", "📚 Database", "➕ Add Document"])
    
    with tools_tab:
        # Your existing tools code
        tools_info = []
        for tool in st.session_state.agent.tools:
            tools_info.append({
                "Tool Name": tool.name,
                "Description": tool.description[:100] + "..." if len(tool.description) > 100 else tool.description
            })
        
        df = pd.DataFrame(tools_info)
        st.dataframe(df, use_container_width=True)
    
    with db_tab:
        # New simple vector DB viewer
        render_vector_db_tab()
    
    with add_tab:
        # Your existing add document code
        with st.form("add_document_form"):
            doc_text = st.text_area("Document Text", height=150)
            doc_source = st.text_input("Source (optional)")
            doc_topic = st.text_input("Topic (optional)")
            
            if st.form_submit_button("Add Document"):
                if doc_text.strip():
                    metadata = {}
                    if doc_source:
                        metadata["source"] = doc_source
                    if doc_topic:
                        metadata["topic"] = doc_topic
                    metadata["added_at"] = datetime.now().isoformat()
                    
                    result = st.session_state.agent.vector_db.add_documents([doc_text], [metadata])
                    st.success(f"Document added: {result}")
                else:
                    st.error("Please enter document text")

def main():
    """Main Streamlit application"""
    init_session_state()
    setup_page()
    
    # Sidebar
    render_sidebar()
    
    # Handle pending approvals
    render_approval_dialog()
    
    # Main content area
    tab1, tab2 = st.tabs(["💬 Chat", "🔧 Tools & Database"])
    
    with tab1:
        render_chat_interface()
    
    with tab2:
        render_tools_page()
    
    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using LangChain and Streamlit")

if __name__ == "__main__":
    main()
