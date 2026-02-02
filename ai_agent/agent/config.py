import os
import yaml
from typing import Dict, Any
from pathlib import Path

class Config:
    """Configuration manager for the AI agent"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    return yaml.safe_load(f)
            else:
                return self._get_default_config()
        except Exception as e:
            print(f"Error loading config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "llm": {
                "default_type": "llama",
                "claude": {
                    "model": "claude-sonnet-4-20250514",
                    "temperature": 0.1
                },
                "llama": {
                    "model": "local_Meta_Llama_LLM_8B",
                    "temperature": 0.1
                }
            },
            "vector_db": {
                "collection_name": "knowledge_base",
                "persist_directory": "./chroma_db",
                "embedding_model": "all-MiniLM-L6-v2"
            },
            "web_search": {
                "max_results": 5,
                "max_content_length": 2000
            },
            "agent": {
                "max_iterations": 10,
                "verbose": True,
                "max_chat_history": 20
            },
            "streamlit": {
                "page_title": "Smart AI Agent",
                "page_icon": "🤖",
                "layout": "wide"
            }
        }
    
    def get(self, key: str, default=None):
        """Get configuration value by key"""
        keys = key.split('.')
        value = self.config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_api_key(self, service: str) -> str:
        """Get API key from environment variables"""
        env_vars = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY"
        }
        
        env_var = env_vars.get(service.lower())
        if env_var:
            return os.getenv(env_var, "")
        return ""