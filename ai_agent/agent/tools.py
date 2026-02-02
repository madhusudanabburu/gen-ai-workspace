# File: agent/tools.py - COMPREHENSIVE BUT CLEAN
from typing import List, Callable
from datetime import datetime
from langchain.tools import Tool
from .vector_db import VectorDBManager
from .web_search import WebSearchManager
from .config import Config

class AgentTools:
    """Comprehensive tools for the agent"""
    
    def __init__(self, config: Config, vector_db: VectorDBManager, 
                 web_search: WebSearchManager, approval_callback: Callable = None):
        self.config = config
        self.vector_db = vector_db
        self.web_search = web_search
        self.approval_callback = approval_callback
    
    def create_tools(self) -> List[Tool]:
        """Create comprehensive, reliable tools"""
        
        def search_knowledge_base(query: str) -> str:
            """Search the knowledge base for stored information"""
            try:
                # Use focused search for better results
                result = self.vector_db.search_similar_focused(query)
                if "No similar documents found" in result or "No highly relevant documents found" in result:
                    return f"No relevant information found in knowledge base for '{query}'. Consider searching the web for current information."
                return result
            except Exception as e:
                # Fallback to regular search
                try:
                    result = self.vector_db.search_similar(query, k=1)  # Just get the top result
                    if "No similar documents found" in result:
                        return f"No relevant information found in knowledge base for '{query}'."
                    return result
                except Exception as e2:
                    return f"Knowledge base search failed: {str(e2)}"
        
        def search_web(query: str) -> str:
            """Search the web for current information"""
            try:
                result = self.web_search.search_web(query)
                # Check for rate limiting
                if "rate limit" in result.lower() or "ratelimit" in result.lower():
                    return f"Web search is currently rate limited for '{query}'. The information may be temporarily unavailable."
                return result
            except Exception as e:
                return f"Web search failed for '{query}': {str(e)}"
        
        def get_current_time(query: str = "") -> str:
            """Get the current date and time"""
            return f"Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        def add_to_knowledge_base(text: str) -> str:
            """Add new information to the knowledge base"""
            try:
                if len(text.strip()) < 10:
                    return "Text is too short to add to knowledge base. Please provide more substantial content."
                
                metadata = {
                    "source": "user_input",
                    "timestamp": datetime.now().isoformat(),
                    "type": "user_provided"
                }
                
                result = self.vector_db.add_documents([text], [metadata])
                return f"Successfully added to knowledge base: {result}"
            except Exception as e:
                return f"Failed to add to knowledge base: {str(e)}"
        
        def get_database_stats(query: str = "") -> str:
            """Get statistics about the knowledge base"""
            try:
                stats = self.vector_db.get_collection_stats()
                doc_count = stats.get('document_count', 'unknown')
                collection_name = stats.get('collection_name', 'unknown')
                return f"Knowledge base '{collection_name}' contains {doc_count} documents."
            except Exception as e:
                return f"Failed to get database stats: {str(e)}"
        
        return [
            Tool(
                name="search_knowledge_base",
                description="Search the knowledge base for stored information. Use this FIRST for any query to check existing knowledge.",
                func=search_knowledge_base
            ),
            Tool(
                name="search_web", 
                description="Search the web for current information. Use this when knowledge base doesn't have sufficient information or for latest/current information.",
                func=search_web
            ),
            Tool(
                name="get_current_time",
                description="Get the current date and time",
                func=get_current_time
            ),
            Tool(
                name="add_to_knowledge_base",
                description="Add new information to the knowledge base for future reference",
                func=add_to_knowledge_base
            ),
            Tool(
                name="get_database_stats",
                description="Get statistics about the knowledge base contents",
                func=get_database_stats
            )
        ]