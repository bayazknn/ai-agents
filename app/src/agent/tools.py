import os
import requests
from typing import Any, Dict, List
import PyPDF2
import io
from dotenv import load_dotenv
load_dotenv()

from langchain_community.tools import WikipediaQueryRun, TavilySearchResults
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import tool
import pdfplumber
from io import BytesIO
import time
import pymupdf4llm
import pymupdf

# Initialize Wikipedia Tool
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

# Initialize Tavily Tool
tavily = TavilySearchResults(max_results=5)



def extract_pdf_from_url(url: str) -> str:
    """
    Downloads a PDF from the given URL and extracts its text content.
    Args:
        url: The URL of the PDF file.
    Returns:
        The extracted text content of the PDF.
    """
    print(f"Fetching PDF from URL started: {url}")
    start_time = time.time()
    response = requests.get(url)
    response.raise_for_status()
    pdf_bytes = BytesIO(response.content)
    end_time = time.time()
    print(f"Fetching PDF from URL finished: {url} in {end_time - start_time:.2f} seconds")
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    markdown_text = pymupdf4llm.to_markdown(doc)
    return markdown_text



    

@tool
def parse_pdf_from_url(pdf_url: str, file_path: str) -> str:
    """
    Downloads a PDF from the given URL and extracts its text content.
    Args:
        pdf_url: The URL of the PDF file.
    Returns:
        The extracted text content of the PDF.
    """
    try:
        if pdf_url:
            response = requests.get(pdf_url)
            response.raise_for_status() # Raise an exception for HTTP errors
            with io.BytesIO(response.content) as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file)
                text = ""
                for page_num in range(len(reader.pages)):
                    text += reader.pages[page_num].extract_text()
                return text
        else:
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page_num in range(len(reader.pages)):
                    text += reader.pages[page_num].extract_text()
                return text
        

    except Exception as e:
        return f"Error parsing PDF from URL {pdf_url}: {e}"

# Placeholder MCP tool function - you'll need to implement this based on your MCP setup
def use_mcp_tool(server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Placeholder function for MCP tool usage.
    You'll need to implement this based on your MCP server setup.
    """
    print(f"MCP Tool Call: {server_name} -> {tool_name} with args: {arguments}")
    return {"status": "success", "result": "MCP tool executed"}

@tool
def graphiti_create_entities(entities: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create multiple new entities in the knowledge graph using Graphiti.
    Args:
        entities: A list of dictionaries, where each dictionary represents an entity
                  with 'name', 'entityType', and 'observations'.
    """
    return use_mcp_tool(
        server_name="github.com/modelcontextprotocol/servers/tree/main/src/memory",
        tool_name="create_entities",
        arguments={"entities": entities}
    )

@tool
def graphiti_add_observations(observations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Add new observations to existing entities in the knowledge graph using Graphiti.
    Args:
        observations: A list of dictionaries, where each dictionary has 'entityName'
                      and 'contents' (list of strings).
    """
    return use_mcp_tool(
        server_name="github.com/modelcontextprotocol/servers/tree/main/src/memory",
        tool_name="add_observations",
        arguments={"observations": observations}
    )

@tool
def graphiti_create_relations(relations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create multiple new relations between entities in the knowledge graph using Graphiti.
    Args:
        relations: A list of dictionaries, where each dictionary has 'from', 'to',
                   and 'relationType'.
    """
    return use_mcp_tool(
        server_name="github.com/modelcontextprotocol/servers/tree/main/src/memory",
        tool_name="create_relations",
        arguments={"relations": relations}
    )

@tool
def graphiti_search_nodes(query: str) -> Dict[str, Any]:
    """
    Search for nodes in the knowledge graph based on a query using Graphiti.
    Args:
        query: The search query to match against entity names, types, and observation content.
    """
    return use_mcp_tool(
        server_name="github.com/modelcontextprotocol/servers/tree/main/src/memory",
        tool_name="search_nodes",
        arguments={"query": query}
    )

@tool
def graphiti_open_nodes(names: List[str]) -> Dict[str, Any]:
    """
    Open specific nodes in the knowledge graph by their names using Graphiti.
    Args:
        names: An array of entity names to retrieve.
    """
    return use_mcp_tool(
        server_name="github.com/modelcontextprotocol/servers/tree/main/src/memory",
        tool_name="open_nodes",
        arguments={"names": names}
    )

# Export tools for use by agents
arxiv_tools = [wikipedia, tavily, parse_pdf_from_url,
               graphiti_create_entities, graphiti_add_observations,
               graphiti_create_relations, graphiti_search_nodes, graphiti_open_nodes]
