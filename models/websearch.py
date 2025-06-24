import httpx
import json
import re
from typing import Dict, List, Optional, Any, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DirectSearchService:
    """Service for handling web search operations using a direct approach via OpenAI API"""
    
    def __init__(self, api_key: str):
        """
        Initialize the search service
        
        Args:
            api_key: OpenAI API key
        """
        self.api_key = api_key
        self.base_url = "https://api.openai.com/v1"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
    
    async def search(
        self, 
        query: str, 
        model: str = "gpt-4o", 
        timeout: float = 30.0
    ) -> Tuple[bool, str]:
        """
        Perform a direct search for the given query
        
        Args:
            query: The search query
            model: OpenAI model to use
            timeout: Request timeout in seconds
            
        Returns:
            Tuple of (success, result text)
        """
        try:
            # Create enhanced search query
            enhanced_query = self._enhance_financial_query(query)
            
            # Configure a basic chat completion request
            # We'll use a system prompt that instructs the model to provide current financial information
            data = {
                "model": model,
                "messages": [
                    {
                        "role": "system", 
                        "content": (
                            "You are a financial expert assistant with the most current knowledge about financial regulations, "
                            "contribution limits, tax rates, and financial planning information up to April 2025.\n\n"
                            
                            "When answering questions about financial topics:\n"
                            "1. Provide the most specific, current information available for 2025\n"
                            "2. Include exact numbers, percentages, dates, and thresholds\n"
                            "3. Be precise about contribution limits, tax brackets, and financial deadlines\n"
                            "4. Specify the applicable tax year in your answer\n"
                            "5. If the information would typically be found on official websites like IRS.gov, mention this fact\n\n"
                            
                            "For example, if asked about Roth IRA limits, include the specific contribution limit for 2025, "
                            "any catch-up provisions, and income phaseout ranges."
                        )
                    },
                    {"role": "user", "content": enhanced_query}
                ],
                "temperature": 0.1,  # Low temperature for factual responses
                "max_tokens": 1024
            }
            
            # Execute request
            async with httpx.AsyncClient(timeout=timeout) as client:
                logger.info(f"Sending direct search request: {enhanced_query}")
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=data
                )
            
            # Handle non-200 responses
            if response.status_code != 200:
                logger.error(f"API error: {response.status_code}, {response.text}")
                return False, f"(API error: {response.status_code})"
            
            # Parse response
            result = response.json()
            
            # Extract the assistant's message content
            if "choices" in result and len(result["choices"]) > 0:
                message = result["choices"][0].get("message", {})
                content = message.get("content", "")
                
                if content:
                    return True, content
                else:
                    logger.warning("Search returned empty content")
                    return False, "(No results available)"
            else:
                logger.warning("Unexpected API response structure")
                return False, "(Unexpected response from API)"
            
        except httpx.RequestError as e:
            logger.error(f"HTTP request error during search: {str(e)}")
            return False, "(Network error during search)"
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error during search: {str(e)}")
            return False, "(Error parsing search results)"
        except Exception as e:
            logger.error(f"Unexpected error during search: {str(e)}")
            return False, f"(An error occurred during search: {str(e)})"
    
    def _enhance_financial_query(self, query: str) -> str:
        """
        Enhance the query to focus on financial information
        
        Args:
            query: Original search query
            
        Returns:
            Enhanced query
        """
        # Check if query is already specific enough
        if re.search(r'(in|for|limit|as of) (202[0-9]|the current year)', query, re.IGNORECASE):
            base_query = query
        else:
            # Add year specificity if missing
            current_year_terms = ["current", "latest", "now", "today"]
            if any(term in query.lower() for term in current_year_terms) or "2025" not in query:
                base_query = f"{query} for 2025"
            else:
                base_query = query
        
        # For direct approach, we use a more straightforward enhanced query
        enhanced_query = (
            f"Provide accurate and current information about {base_query}. "
            f"Include specific numbers, dates, limits, thresholds, and percentages "
            f"for the 2025 tax year. Make sure to be precise about any financial limits or regulations."
        )
        
        return enhanced_query

# For backward compatibility, create an alias of DirectSearchService as WebSearchService
WebSearchService = DirectSearchService