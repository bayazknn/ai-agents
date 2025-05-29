import google.generativeai as genai
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor


def create_cached_chain(pdf_text: str, system_prompt: str):
    """
    Create a cached chain for Gemini with proper parameter handling
    
    Args:
        pdf_text: The PDF content to cache
        system_prompt: System instruction for the model
        
    Returns:
        Tuple of (cached_chain, cached_content)
    """
    # Configure the API
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    
    try:
        # Create cached content
        cached_content = genai.caching.CachedContent.create(
            model="models/gemini-2.0-flash",
            display_name="arxiv_paper_cache",
            system_instruction=system_prompt,
            contents=[{
                "role": "user", 
                "parts": [{"text": f"Analyze this arXiv paper:\n\n{pdf_text}"}]
            }],
            # Optional: Set TTL (time to live) for cache
            # ttl=datetime.timedelta(hours=1)  # Cache for 1 hour
        )
        
        # Create LLM with cached content - only use supported parameters
        gemini_llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.1,  # Slightly higher for more natural responses
            # Put the cached content reference in model_kwargs
            model_kwargs={
                "cached_content": cached_content.name,
                "cache":True,
                # Remove cache_content - it's not needed
            }
        )
        
        # Create a more robust prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("human", "{input}")
        ])
        
        # Create the chain
        cached_chain = prompt | gemini_llm
        
        return (
            cached_chain,
            cached_content
        )
        
    except Exception as e:
        print(f"Error creating cached chain: {e}")
        return None, None


def create_agent_chain_with_tools(llm, system_prompt: str, tools_list=None):
    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
        
    # Create the agent
    if tools_list:
        agent = create_tool_calling_agent(llm, tools_list, prompt)
    else:
        chain = prompt | llm
        return chain
    
    # Create agent executor
    agent_executor = AgentExecutor(agent=agent, tools=tools_list, verbose=True)
    
    return agent_executor


def create_agent_chain(llm, system_prompt: str):
    prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_prompt),
    HumanMessagePromptTemplate.from_template("{input}")])

# Create a chain with the prompt
    chain = prompt_template | llm
    return chain    