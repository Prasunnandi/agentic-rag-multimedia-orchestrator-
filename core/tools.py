import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults

@tool
def web_search(query: str) -> str:
    """Search the web for information using Tavily."""
    try:
        search = TavilySearchResults(max_results=3)
        return str(search.invoke({"query": query}))
    except Exception as e:
        return f"Search failed: {e}. Ensure TAVILY_API_KEY is set."

@tool
def scrape_url(url: str) -> str:
    """Scrape text from a URL."""
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()
        text = soup.get_text(separator="\n", strip=True)
        return text[:5000] # Truncate to save context limits
    except Exception as e:
        return f"Error scraping {url}: {e}"
