"""
LLM Helper - Simple Gemini 2.5 Flash Integration
Loads API key from .env and provides ready-to-use LLM instance.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def get_llm(temperature: float = 0.0):
    """
    Get Gemini 2.5 Flash model instance.
    
    Args:
        temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
    
    Returns:
        LLM instance ready to use, or None if API key not found
    
    Usage:
        llm = get_llm()
        if llm:
            response = llm.invoke("What is 2+2?")
    """
    # Get API key from environment
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        print("⚠️  GOOGLE_API_KEY not found in .env file")
        print("   Please create .env file with: GOOGLE_API_KEY=your_key_here")
        return None
    
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        # Create Gemini 2.5 Flash instance
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",  # Gemini 2.5 Flash (latest fast model)
            google_api_key=api_key,
            temperature=temperature,
            max_tokens=None,  # No limit
            timeout=None,
            max_retries=2,
        )
        
        print(f"✓ Loaded Gemini 2.5 Flash (temperature={temperature})")
        return llm
        
    except ImportError:
        print("⚠️  langchain-google-genai not installed")
        print("   Run: uv add langchain-google-genai")
        return None
    except Exception as e:
        print(f"⚠️  Error loading LLM: {e}")
        return None


def test_llm():
    """Test if LLM is working."""
    llm = get_llm()
    
    if not llm:
        return False
    
    try:
        response = llm.invoke("Say 'Hello' if you can hear me")
        print(f"✓ LLM test successful: {response.content}")
        return True
    except Exception as e:
        print(f"✗ LLM test failed: {e}")
        return False


if __name__ == "__main__":
    print("Testing LLM connection...")
    test_llm()

