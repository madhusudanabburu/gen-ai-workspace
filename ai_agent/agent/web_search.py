# File: agent/web_search.py - ENHANCED VERSION WITH COMPATIBILITY
import requests
from bs4 import BeautifulSoup
import re
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import json
import time

try:
    from langchain_community.tools import DuckDuckGoSearchRun
    from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
except ImportError:
    try:
        # Use the new ddgs package
        from ddgs import DDGS
        # Create a wrapper for compatibility
        class DuckDuckGoSearchRun:
            def run(self, query):
                with DDGS() as ddgs:
                    results = list(ddgs.text(query, max_results=5))
                    return str(results)
    except ImportError:
        # Fallback to old package
        from langchain.tools import DuckDuckGoSearchRun
        from langchain.utilities import DuckDuckGoSearchAPIWrapper

from .config import Config

class WebSearchManager:
    """Enhanced web search manager with backward compatibility"""
    
    def __init__(self, config: Config):
        self.config = config
        self.max_results = config.get("web_search.max_results", 5)
        self.max_content_length = config.get("web_search.max_content_length", 2000)
        self.timeout = config.get("web_search.timeout", 15)
        
        # Initialize search tools
        self.ddg_search = DuckDuckGoSearchRun()
        
        # Rate limiting
        self.last_search_time = 0
        self.min_search_interval = 2  # seconds between searches
        
        # Result caching (simple in-memory cache)
        self.search_cache = {}
        self.cache_duration = 300  # 5 minutes
    
    def search_web(self, query: str) -> str:
        """Main search method with enhanced processing"""
        try:
            # Check cache first
            cached_result = self._get_cached_result(query)
            if cached_result:
                return cached_result
            
            # Rate limiting
            self._enforce_rate_limit()
            
            # Determine search type and strategy
            search_type = self._determine_search_type(query)
            
            # Perform enhanced search
            raw_results = self._perform_search(query, search_type)
            
            # Process and enhance results
            enhanced_result = self._process_and_format_results(raw_results, query)
            
            # Cache the result
            self._cache_result(query, enhanced_result)
            
            return enhanced_result
            
        except Exception as e:
            return self._handle_search_error(e, query)
    
    def _determine_search_type(self, query: str) -> str:
        """Determine the type of search based on query content"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["latest", "recent", "news", "current", "today", "2024", "2025"]):
            return "news"
        elif any(word in query_lower for word in ["research", "study", "paper", "academic", "journal"]):
            return "academic"
        elif any(word in query_lower for word in ["compare", "vs", "versus", "difference", "comparison"]):
            return "comparison"
        else:
            return "general"
    
    def _perform_search(self, query: str, search_type: str) -> str:
        """Perform search with strategy based on type"""
        try:
            if search_type == "news":
                enhanced_query = f"{query} latest news 2024"
            elif search_type == "academic":
                enhanced_query = f"{query} research study"
            elif search_type == "comparison":
                enhanced_query = f"compare {query}"
            else:
                enhanced_query = self._optimize_query(query)
            
            result = self.ddg_search.run(enhanced_query)
            return result
            
        except Exception as e:
            # Fallback to basic query
            return self.ddg_search.run(query)
    
    def _optimize_query(self, query: str) -> str:
        """Optimize query for better search results"""
        # Remove unnecessary question words
        stop_words = {"what", "is", "are", "how", "why", "when", "where", "who", "can", "you"}
        words = query.lower().split()
        
        # Keep stop words if query is very short
        if len(words) <= 3:
            return query
        
        optimized_words = [word for word in words if word not in stop_words]
        return " ".join(optimized_words) if optimized_words else query
    
    def _process_and_format_results(self, raw_results: str, query: str) -> str:
        """Process and format search results for better presentation"""
        try:
            # Extract meaningful sentences
            sentences = self._extract_relevant_sentences(raw_results, query)
            
            if not sentences:
                return f"Web search results for '{query}':\n\n{raw_results[:self.max_content_length]}"
            
            # Format as clean, readable response
            formatted_response = self._format_as_answer(sentences, query)
            
            return formatted_response
            
        except Exception as e:
            # Fallback to basic formatting
            return f"Web search results for '{query}':\n\n{raw_results[:self.max_content_length]}"
    
    def _extract_relevant_sentences(self, text: str, query: str) -> List[str]:
        """Extract sentences most relevant to the query"""
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        
        # Get query keywords
        query_words = set(word.lower() for word in query.split() if len(word) > 2)
        
        # Score sentences by relevance
        scored_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 30:  # Skip very short sentences
                continue
            
            sentence_words = set(re.findall(r'\b\w+\b', sentence.lower()))
            relevance_score = len(query_words.intersection(sentence_words))
            
            if relevance_score > 0:
                scored_sentences.append((sentence, relevance_score))
        
        # Sort by relevance and return top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        return [sentence for sentence, score in scored_sentences[:4]]
    
    def _format_as_answer(self, sentences: List[str], query: str) -> str:
        """Format sentences as a coherent answer"""
        if not sentences:
            return f"No specific information found for '{query}'."
        
        # Create a structured response
        response_parts = [f"Based on current web information about '{query}':"]
        response_parts.append("")
        
        # Add the most relevant information as numbered points
        for i, sentence in enumerate(sentences[:3], 1):
            clean_sentence = sentence.strip()
            if not clean_sentence.endswith('.'):
                clean_sentence += '.'
            response_parts.append(f"{i}. {clean_sentence}")
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        response_parts.append("")
        response_parts.append(f"(Search performed at {timestamp})")
        
        return "\n".join(response_parts)
    
    def _enforce_rate_limit(self):
        """Enforce rate limiting between searches"""
        current_time = time.time()
        time_since_last = current_time - self.last_search_time
        
        if time_since_last < self.min_search_interval:
            sleep_time = self.min_search_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_search_time = time.time()
    
    def _get_cached_result(self, query: str) -> Optional[str]:
        """Get cached search result if available"""
        cache_key = query.lower().strip()
        if cache_key in self.search_cache:
            cached_data = self.search_cache[cache_key]
            if time.time() - cached_data['timestamp'] < self.cache_duration:
                return cached_data['result']
            else:
                del self.search_cache[cache_key]
        return None
    
    def _cache_result(self, query: str, result: str):
        """Cache search result"""
        cache_key = query.lower().strip()
        self.search_cache[cache_key] = {
            'result': result,
            'timestamp': time.time()
        }
        
        # Simple cache cleanup
        if len(self.search_cache) > 20:
            oldest_key = min(self.search_cache.keys(), 
                           key=lambda k: self.search_cache[k]['timestamp'])
            del self.search_cache[oldest_key]
    
    def _handle_search_error(self, error: Exception, query: str) -> str:
        """Handle search errors gracefully"""
        error_str = str(error).lower()
        
        if any(term in error_str for term in ["rate", "limit", "429", "ratelimit"]):
            return f"Web search is currently rate limited for '{query}'. Please try again in a few moments."
        elif any(term in error_str for term in ["timeout", "connection", "network"]):
            return f"Web search connection failed for '{query}'. Please check your internet connection."
        else:
            return f"Web search error for '{query}': {str(error)[:100]}. Please try again or rephrase your query."
    
    def fetch_webpage_content(self, url: str) -> str:
        """Enhanced webpage content extraction"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "header", "footer"]):
                script.decompose()
            
            # Get text and clean it
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Limit text length
            if len(text) > self.max_content_length:
                text = text[:self.max_content_length] + "..."
            
            return f"Content from {url}:\n{text}"
        except Exception as e:
            return f"Error fetching webpage: {str(e)}"
    
    # Enhanced methods for better functionality
    def search_web_synthesized(self, query: str) -> str:
        """Alias for enhanced search (backward compatibility)"""
        return self.search_web(query)
    
    def synthesize_search_results(self, raw_results: str, query: str) -> str:
        """Synthesize raw results (backward compatibility)"""
        return self._process_and_format_results(raw_results, query)
    
    def get_search_stats(self) -> Dict:
        """Get search statistics"""
        return {
            "cached_searches": len(self.search_cache),
            "cache_duration_minutes": self.cache_duration / 60,
            "rate_limit_interval": self.min_search_interval
        }
    
    def clear_cache(self):
        """Clear search cache"""
        self.search_cache.clear()