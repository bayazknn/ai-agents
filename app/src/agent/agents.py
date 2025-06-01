from locale import strcoll
import os
import operator
from dotenv import load_dotenv
from pydantic.v1 import ConfigDict
load_dotenv()
from typing import Any, Dict, List, TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.runnables import RunnableConfig

from langgraph.prebuilt import create_react_agent
from langgraph.graph.message import add_messages
import google.generativeai as genai
from agent.create_chain import create_cached_chain, create_agent_chain_with_tools, create_agent_chain
from agent.prompts import STUDENT_AGENT_PROMPT, TEACHER_AGENT_PROMPT, OBSERVER_AGENT_PROMPT
from agent.tools import arxiv_tools, graphiti_create_entities, graphiti_add_observations, parse_pdf_from_url, extract_pdf_from_url
import re
import json


gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0.1,

    model_kwargs={
        "max_output_tokens": 1500
    }
)

# --- Agent Definitions ---

class AgentState(TypedDict):
    """
    Represents the state of the agent graph.
    """
    messages: list[Any]
    arxiv_paper_url: str
    arxiv_paper: str
    questions_list: List[str]
    current_turn: int = 0
    observer_insights: List[str]
    final_summary: List[Dict[str, Any]]
    turn_annotations: List[Dict[str, Any]]
    student_chain: Any
    teacher_chain: Any
    observer_chain: Any
    error: Any



def init_node(state: AgentState, config: RunnableConfig):

    arxiv_paper_url = state.get("arxiv_paper_url")
    arxiv_paper = extract_pdf_from_url(arxiv_paper_url)
    
    
    try:
        student_chain = create_agent_chain(gemini_llm, STUDENT_AGENT_PROMPT)
        teacher_chain = create_agent_chain(gemini_llm, TEACHER_AGENT_PROMPT)
        observer_chain = create_agent_chain(gemini_llm, OBSERVER_AGENT_PROMPT)
    except Exception as e:
        return {
            "error": str(e)
        }

    return {
        "arxiv_paper": arxiv_paper,
        "current_turn": 0,
        "student_chain": student_chain,
        "teacher_chain": teacher_chain,
        "observer_chain": observer_chain,
        }

    
    

def student_node(state: AgentState, config: RunnableConfig):
    messages = state.get("messages", [])
    current_turn = state.get("current_turn",0)
    observer_insights = state.get("observer_insights", [])
    questions_list = state.get("questions_list", [])
    student_chain = state.get("student_chain")
    arxiv_paper = state.get("arxiv_paper")
    print(f"Student: Current turn: {current_turn}")

    input_message = ""

    if current_turn == 0:

        prompt = f"""You are question generator for arxiv paper. Generate 6 questions for this paper which content text is below.
                Your questions are responded from experts. You use paper abstract to generate questions.
                Your generated questions cover main points of paper. 
                Dont generate simple questions like "What is this paper about?" or "What is the main contribution of this paper?". 
                Your generated questions are used in phd research. Questions must be creative and reveal deep insights of paper. 
                Your response format is json and dont include any other text. Your responses are directly parsed as json.
                
                #Question Model Format#
                Every question item has title, prompt, category.
                title: short format of question title
                prompt: Question has 50 words max. Prompt has include questions and be formatted as modern llm prompt engineering rules.
                category: One single word to categorize the questions. Category keyword reflects the type of question.
                #End Question Model Format#




                #Response Format#
                [
                    
                    title: "Explain a research papcontenter",
                    prompt: "Can you explain the key findings and methodology of this research paper?",
                    category: "research",
                
                
                    title: "Explain limitations",
                    prompt: "What are the limitations or potential weaknesses of the approach described in this paper?",
                    category: "critique",
                    
                
                    title: "Compare with other papers",
                    prompt: "How does this paper compare to other recent work in the same field?",
                    category: "analysis",
                
                ]
                #End Response Format#




                #Negative Prompt for Response#
                Dont add introdutory wordings. Follow below templates:
                <Comparative Response>
                    <Bad Response>
                    The study compares brute force TSK, cascading GFT, and FCM-based approaches. 
                    What are the trade-offs in terms of accuracy, complexity, and interpretability between these three GFS strategies when applied to the Airfoil Self Noise dataset?
                    </Bad Response>
                    <Good Response>
                    What are the trade-offs in terms of accuracy, complexity, and interpretability between these three GFS strategies when applied to the Airfoil Self Noise dataset?
                    </Good Response>
                </Comparative Response>
                <Comparative Response>
                    <Bad Response>
                    The paper explores three GFS strategies. What are the trade-offs in terms of explainability, computational cost, and predictive accuracy between brute force TSK, cascading GFT, and FCM-based approaches, especially considering their application to the Airfoil Self Noise dataset?
                    </Bad Response>
                    <Good Response>
                    What are the trade-offs in terms of explainability, computational cost, and predictive accuracy between brute force TSK, cascading GFT, and FCM-based approaches, especially considering their application to the Airfoil Self Noise dataset?
                    </Good Response>
                </Comparative Response>
                #End Negative Prompt for Response#
                


                #Paper Content Text#
                {arxiv_paper}
                #End Paper Content Text#
                """
        first_questions_response = student_chain.invoke({"input": prompt})

        cleaned = re.sub(r'^```json\s*', '', first_questions_response.content, flags=re.IGNORECASE)  # Remove ```json at the start
        cleaned = re.sub(r'```$', '', cleaned)  # Remove ``` at the end
        cleaned = cleaned.strip()

        questions_list = json.loads(cleaned)

        question_string_template = f"""<Questions>
        {"\n".join([f"<Question>{question['prompt']}</Question>" for question in questions_list])}
        </Questions>"""

        print("question_string_template", question_string_template)

        return {
        "messages": messages + [HumanMessage(content=question_string_template, name="Student")],
        "current_turn": current_turn + 1,
        "questions_list": questions_list
        }


    else:
        teacher_response = [msg for msg in messages if msg.name == "Teacher"][-1].content

        question_generator_template = f"""
        Generate 3 questions to understand the paper better according to the teacher response below.
        Do not include any other text. questions should be concise and directly. Dont mention and refer. Only return the questions as a xml format like that:

        <Questions>
        <Question>Question 1</Question>
        <Question>Question 2</Question>
        <Question>Question 3</Question>
        </Questions>

        #Teacher Response#
        {teacher_response}
        #End Teacher Response#
        """

        if observer_insights:
            observer_insights_template = f""" Follow these observer insights when generating questions:
            #Observer Insights#
            {"\n\n".join(observer_insights)}
            #End Observer Insights#
            """
            input_template = observer_insights_template + question_generator_template
        else:
            input_template = question_generator_template

        try:
            question_generator_response = student_chain.invoke({"input": input_template})
        except Exception as e:
            return {
                "error": str(e)
            }
        print("question_generator_response", question_generator_response)
        
        # This function now returns the input dictionary for the student_agent
        return {
            "messages": messages + [AIMessage(content=question_generator_response.content, name="Student")],
            "current_turn": current_turn + 1,
            "questions_list": questions_list + [question_generator_response.content] # Keep as list of one string for now, user can clarify if individual questions are needed
        }

def teacher_node(state: AgentState, config: RunnableConfig):
    current_turn = state.get("current_turn", 0) 
    observer_insights = state.get("observer_insights", [])
    last_questions = state.get("questions_list", [])[-1]
    arxiv_paper = state.get("arxiv_paper")
    teacher_chain = state.get("teacher_chain")
    messages = state.get("messages", [])
    input_message = ""

    # Combine observer insights into the input if available

    input_message = f"""
    Respond to the questions below in the context of the arxiv paper. Respond concise, informative and shortly as a xml format like that:

    <Responses>
    <ResponseItem>
    <Question>Question 1</Question>
    <Response>Response 1</Response>
    </ResponseItem>
    <ResponseItem>
    <Question>Question 2</Question>
    <Response>Response 2</Response>
    </ResponseItem>
    <ResponseItem>
    <Question>Question 3</Question>
    <Response>Response 3</Response>
    </ResponseItem>
    </Responses>

    Do not include any other text. just return the xml format.




    #Questions#
    {last_questions}
    #End Questions#


    #Arxiv Paper#
    {arxiv_paper}
    #End Arxiv Paper#
    """

    if observer_insights:
        observer_insights_template = f""" Follow these observer insights when answering questions:
        #Observer Insights#
        {"\n\n".join(observer_insights)}
        #End Observer Insights#
        """

        input_template = observer_insights_template + input_message
    else:
        input_template = input_message

    try:
        teacher_response = teacher_chain.invoke({"input": input_template})
    except Exception as e:
        return {
            "error": str(e)
        }

    # This function now returns the input dictionary for the teacher_agent
    return {
        "messages": messages + [AIMessage(content=teacher_response.content, name="Teacher") ],
        "current_turn": current_turn + 1,
    }

def observer_node(state: AgentState, config: RunnableConfig):
    messages = state.get("messages", [])
    current_turn = state.get("current_turn")
    observer_insights = state.get("observer_insights", [])
    observer_chain = state.get("observer_chain")

    

    conversation_history_template = f"\n\n".join([f"""
    <Response>
    <Turn>{turn}</Turn>
    <Responder>{message.name}</Responder>
    <ResponseContent>
    {message.content}
    </ResponseContent>
    </Response>
    """ for turn, message in enumerate(messages[1:])])
    observer_input = f"""
    Review the conversation history and provide instructions to guide the student and teacher to discuss cached arxiv paper arxiv_paper_cache .
    Give instructions for student to generate better questions. 
    Give instructions for teacher to when answering questions to follow instructions.

    Give only instructions do not include any other text. Response format should be xml format like that:

    <Instructions>
    <Instruction>
        <Student>
            Instructions for student
        </Student>
        <Teacher>
            Instructions for teacher
        </Teacher>
    </Instruction>
    </Instructions>
    
    
    #Conversations#
    {conversation_history_template}
    #End Conversations#
    """
    try:
        observer_response = observer_chain.invoke({"input": observer_input})
    except Exception as e:
        return {
            "error": str(e)
        }


    return {
        "messages": messages + [AIMessage(content=observer_response.content, name="Observer")],
        "current_turn": current_turn + 1,
        "observer_insights": observer_insights + [observer_response.content],
    }
