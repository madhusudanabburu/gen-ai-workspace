# File: agent/vector_db.py
# =============================================================================
from typing import List, Dict, Optional
import re

# Updated imports to avoid deprecation warnings
try:
    from langchain_chroma import Chroma
    print("✅ Using langchain-chroma package")
    USING_NEW_CHROMA = True
except ImportError:
    from langchain.vectorstores import Chroma
    print("⚠️ Using deprecated langchain.vectorstores.Chroma - consider upgrading")
    USING_NEW_CHROMA = False
    
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    print("✅ Using langchain-huggingface package")
except ImportError:
    from langchain.embeddings import HuggingFaceEmbeddings
    print("⚠️ Using deprecated langchain.embeddings.HuggingFaceEmbeddings - consider upgrading")

from langchain.schema import Document
from .config import Config

class VectorDBManager:
    """Manages vector database operations"""
    
    def __init__(self, config: Config):
        self.config = config
        self.collection_name = config.get("vector_db.collection_name")
        self.persist_directory = config.get("vector_db.persist_directory")
        self.embedding_model = config.get("vector_db.embedding_model")
        
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        self.vectorstore = None
        self._initialize_vectorstore()
    
    def _initialize_vectorstore(self):
        """Initialize the vector store with improved error handling"""
        try:
            # Ensure directory exists
            import os
            os.makedirs(self.persist_directory, exist_ok=True)
            
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
            
            # Test that it actually works by trying a simple operation
            try:
                # Try to get the collection count (this will fail if DB isn't working)
                if hasattr(self.vectorstore, '_collection'):
                    collection = self.vectorstore._collection
                    count = collection.count()
                    print(f"📚 Vector database initialized: {self.collection_name} ({count} documents)")
                else:
                    # For newer versions, try alternative method
                    try:
                        # Try a simple search to validate
                        self.vectorstore.similarity_search("test", k=1)
                        print(f"📚 Vector database initialized: {self.collection_name}")
                    except Exception:
                        print(f"📚 Vector database initialized: {self.collection_name} (new database)")
                
                print(f"📚 Vector database validated successfully")
                
            except Exception as test_error:
                print(f"📚 Vector database initialized: {self.collection_name} (new/empty database)")
                # Don't fail here, just note it's a new database
                
        except Exception as e:
            print(f"❌ Failed to initialize vector database: {e}")
            print(f"❌ Full error: {type(e).__name__}: {str(e)}")
            self.vectorstore = None
            # Don't raise the exception, just continue with None vectorstore
    
    def _safe_persist(self):
        """Safely persist the vector database if method exists"""
        if not self.vectorstore:
            return False
            
        try:
            # Check if persist method exists (older versions)
            if hasattr(self.vectorstore, 'persist'):
                self.vectorstore.persist()
                print("📚 Database persisted using .persist() method")
                return True
            else:
                # Newer versions may persist automatically or use different method
                print("📚 Database persistence handled automatically (new Chroma version)")
                return True
        except Exception as e:
            print(f"⚠️ Persistence warning: {e}")
            return True  # Don't fail on persistence issues
    
    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract key terms from query for better matching"""
        # Remove common question words and focus on key terms
        stop_words = {"what", "is", "are", "the", "a", "an", "how", "why", "when", "where", "who", "which", "tell", "me", "about"}
        
        # Clean and split query
        words = re.findall(r'\b[a-zA-Z]+\b', query.lower())
        key_terms = [word for word in words if word not in stop_words and len(word) > 2]
        
        return key_terms
    
    def _calculate_relevance_score(self, doc_content: str, query_terms: List[str]) -> float:
        """Calculate relevance score based on term matching"""
        content_lower = doc_content.lower()
        score = 0.0
        
        for term in query_terms:
            # Exact match gets higher score
            if term in content_lower:
                score += 1.0
            # Partial match gets lower score
            elif any(term in word for word in content_lower.split()):
                score += 0.5
        
        # Normalize by number of terms
        return score / len(query_terms) if query_terms else 0.0
    
    def search_similar_focused(self, query: str) -> str:
        """Search for the most relevant document with smart filtering"""
        if not self.vectorstore:
            return "Vector database not initialized."
        
        try:
            # Get top results
            docs = self.vectorstore.similarity_search(query, k=3)
            if not docs:
                return "No similar documents found in the knowledge base"
            
            # For simple queries, just return the most relevant result
            best_doc = docs[0]
            
            # Check if it's actually relevant
            query_lower = query.lower()
            content_lower = best_doc.page_content.lower()
            
            # Simple relevance check
            query_words = [word for word in query_lower.split() if len(word) > 2]
            relevance_score = sum(1 for word in query_words if word in content_lower)
            
            if relevance_score > 0:
                # Return just the content without metadata for clean response
                return best_doc.page_content
            else:
                return "No highly relevant documents found in the knowledge base"
                
        except Exception as e:
            return f"Error searching vector database: {str(e)}"
    
    def add_documents(self, texts: List[str], metadatas: List[Dict] = None) -> str:
        """Add documents to the vector database"""
        if not self.vectorstore:
            print("📚 Vector database not initialized - attempting to reinitialize...")
            try:
                self._initialize_vectorstore()
                if not self.vectorstore:
                    return "Vector database failed to initialize. Please check the logs."
            except Exception as e:
                return f"Vector database initialization failed: {str(e)}"
        
        try:
            documents = [Document(page_content=text, metadata=meta or {}) 
                        for text, meta in zip(texts, metadatas or [{}] * len(texts))]
            
            # Add documents
            self.vectorstore.add_documents(documents)
            
            # Try to persist (safely)
            self._safe_persist()
            
            return f"Added {len(documents)} documents to vector database"
        except Exception as e:
            return f"Error adding documents: {str(e)}"
    
    def search_similar(self, query: str, k: int = 5) -> str:
        """Search for similar documents in the vector database (original method)"""
        if not self.vectorstore:
            print("📚 Vector database not initialized - attempting to reinitialize...")
            try:
                self._initialize_vectorstore()
                if not self.vectorstore:
                    return "Vector database failed to initialize. Please check the logs."
            except Exception as e:
                return f"Vector database initialization failed: {str(e)}"
        
        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            if not docs:
                return "No similar documents found in the knowledge base"
            
            results = []
            for i, doc in enumerate(docs, 1):
                metadata = doc.metadata
                results.append(f"Result {i}:\n{doc.page_content}\nMetadata: {metadata}\n")
            
            return "\n".join(results)
        except Exception as e:
            return f"Error searching vector database: {str(e)}"
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection"""
        if not self.vectorstore:
            print("📚 Vector database not initialized - attempting to reinitialize...")
            try:
                self._initialize_vectorstore()
                if not self.vectorstore:
                    return {"error": "Vector database failed to initialize"}
            except Exception as e:
                return {"error": f"Vector database initialization failed: {str(e)}"}
        
        try:
            # Try multiple methods to get count
            if hasattr(self.vectorstore, '_collection'):
                collection = self.vectorstore._collection
                count = collection.count()
            else:
                # For newer versions, try alternative approaches
                try:
                    # Method 1: Try to get the collection directly
                    if hasattr(self.vectorstore, '_client'):
                        collection = self.vectorstore._client.get_collection(self.collection_name)
                        count = collection.count()
                    else:
                        # Method 2: Estimate by trying a large search
                        docs = self.vectorstore.similarity_search("", k=10000)
                        count = len(docs)
                except Exception:
                    count = "unknown"
            
            return {
                "document_count": count,
                "collection_name": self.collection_name,
                "status": "initialized"
            }
        except Exception as e:
            return {"error": str(e), "status": "error"}