# File: agent/core.py - COMPREHENSIVE BUT CLEAN
from typing import List, Dict, Any, Optional, Callable
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain.schema import BaseMessage
from langchain_anthropic import ChatAnthropic

try:
    from langchain_ollama import OllamaLLM
except ImportError:
    try:
        from langchain_community.llms import Ollama as OllamaLLM
    except ImportError:
        OllamaLLM = None

from .config import Config
from .vector_db import VectorDBManager
from .web_search import WebSearchManager
from .tools import AgentTools

class SmartAgent:
    """Comprehensive AI Agent with Smart Search Strategy"""
    
    def __init__(self, config: Config = None, approval_callback: Callable = None):
        self.config = config or Config()
        
        # Initialize components
        self.vector_db = VectorDBManager(self.config)
        self.web_search = WebSearchManager(self.config)
        self.agent_tools = AgentTools(self.config, self.vector_db, self.web_search, approval_callback)
        
        # Initialize LLM and agent
        self.llm = self._initialize_llm()
        self.tools = self.agent_tools.create_tools()
        self.agent_executor = self._create_agent_executor()
    
    def _initialize_llm(self):
        """Initialize the language model"""
        llm_type = self.config.get("llm.default_type", "llama")
        
        if llm_type.lower() == "claude":
            api_key = self.config.get_api_key("anthropic")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY required for Claude")
            
            return ChatAnthropic(
                anthropic_api_key=api_key,
                model=self.config.get("llm.claude.model"),
                temperature=self.config.get("llm.claude.temperature", 0.1)
            )
        
        elif llm_type.lower() == "llama":
            if OllamaLLM is None:
                raise ValueError("Ollama package not available")
                
            return OllamaLLM(
                model=self.config.get("llm.llama.model"),
                temperature=self.config.get("llm.llama.temperature", 0.1)
            )
        
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")
    
    def _create_agent_executor(self):
        """Create agent with comprehensive search strategy"""
        prompt = PromptTemplate.from_template("""You are a helpful AI assistant with access to multiple information sources. Follow this search strategy:

SEARCH STRATEGY:
1. For any question, FIRST search your knowledge base
2. If knowledge base doesn't have sufficient info, search the web for current information
3. Use reasoning to synthesize information from multiple sources when needed

AVAILABLE TOOLS:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do next
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat as needed)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}""")
        
        agent = create_react_agent(llm=self.llm, tools=self.tools, prompt=prompt)
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=40
        )
    
    def chat(self, message: str, chat_history: List[BaseMessage] = None) -> str:
        """Comprehensive chat method"""
        try:
            print("Input from user " + message)
            response = self.agent_executor.invoke({"input": message})
            return response["output"]
        except Exception as e:
            print(f"Agent error: {e}")
            # Only use fallback for serious errors, not normal issues
            if "parsing" in str(e).lower() or "format" in str(e).lower():
                return f"I encountered a formatting issue. Please try rephrasing your question."
            else:
                return f"I encountered an error: {str(e)}. Please try rephrasing your question."
    
    def initialize_with_sample_data(self):
        """Load comprehensive sample data"""
        sample_docs = [
            "LangChain is a framework for developing applications powered by language models. It provides tools for building agents, chains, and retrieval systems. LangChain simplifies the process of working with LLMs by providing abstractions and utilities.",
            "Python is a popular programming language for AI development, offering libraries like LangChain, OpenAI, TensorFlow, PyTorch, and HuggingFace Transformers. It's known for its simplicity and extensive ecosystem.",
            "Streamlit is a Python library for creating web applications for machine learning and data science projects. It allows developers to create interactive dashboards and apps with minimal code.",
            "JP Morgan Chase is one of the largest banks in the United States, offering investment banking, financial services, and asset management. The bank has been investing heavily in technology and AI.",
            "AI agents are autonomous systems that can perceive their environment, make decisions, and take actions to achieve specific goals. They can be simple rule-based systems or complex learning systems.",
            "Vector databases store high-dimensional vectors and enable similarity search. They're essential for retrieval-augmented generation (RAG) systems and semantic search applications.",
            "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It includes supervised, unsupervised, and reinforcement learning."
        ]
        
        metadata = [
            {"source": "langchain_docs", "topic": "framework", "type": "technical"},
            {"source": "programming_guide", "topic": "python", "type": "technical"},
            {"source": "streamlit_docs", "topic": "web_framework", "type": "technical"},
            {"source": "financial_info", "topic": "banking", "type": "business"},
            {"source": "ai_concepts", "topic": "agents", "type": "technical"},
            {"source": "vector_db_guide", "topic": "database", "type": "technical"},
            {"source": "ai_concepts", "topic": "machine_learning", "type": "technical"}
        ]
        
        return self.vector_db.add_documents(sample_docs, metadata)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive stats"""
        return {
            "llm_type": self.config.get("llm.default_type"),
            "tools_count": len(self.tools),
            "max_iterations": 15,
            "vector_db_stats": self.vector_db.get_collection_stats()
        }