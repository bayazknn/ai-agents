from typing import Any, Dict, List, TypedDict
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from dotenv import load_dotenv
load_dotenv()

# Import agents from the same package
from agent.agents import student_node, teacher_node, observer_node, init_node, AgentState
from langgraph.checkpoint.memory import InMemorySaver

# # Define the graph state
# class AgentState(MessagesState):
#     arxiv_paper: str
#     questions_list: List[str]
#     current_turn: int = 0
#     observer_insights: List[str]
#     final_summary: List[Dict[str, Any]]
#     turn_annotations: List[Dict[str, Any]]
#     agent_memory: Dict[str, Any]



# Define configurable parameters for the graph
class Configuration(TypedDict):
    """Configurable parameters for the agent.
    Set these when creating assistants OR when invoking the graph.
    """
    pdf_url: str

# Define constants for N and K
N_LOOPS = 9
K_INTERVAL = 3

# Define the graph with custom checkpoint saver
workflow = StateGraph(
    AgentState
)

# Add nodes for each agent
workflow.add_node("initial", init_node )
workflow.add_node("student", student_node )
workflow.add_node("teacher", teacher_node )
workflow.add_node("observer", observer_node)



# Define routing function from student
def route_from_student(state: AgentState):
    current_turn = state["current_turn"]

    if "error" in state:
        print(f"Routing from student: Error occurred. Ending conversation.")
        return "end"


    if current_turn == 0:
        print(f"Routing from student: Initial turn. Going to student.")
        return "student"
    elif current_turn % K_INTERVAL == 0:
        print(f"Routing from student: K interval ({K_INTERVAL}) reached at turn {current_turn}. Going to observer for insights.")
        return "observer"
    elif current_turn < N_LOOPS:
        print(f"Routing from student: Questions remaining. Going to teacher.")
        return "teacher"
    else:
        print(f"Routing from student: No more questions. Ending conversation.")
        return "end"

# Define routing function from teacher
def route_from_teacher(state: AgentState):
    current_turn = state["current_turn"]

    if "error" in state:
        print(f"Routing from teacher: Error occurred. Ending conversation.")
        return "end"
    
    if current_turn % K_INTERVAL == 0:
        print(f"Routing from teacher: K interval ({K_INTERVAL}) reached at turn {current_turn}. Going to observer for insights.")
        return "observer"
    elif current_turn < N_LOOPS:
        print(f"Routing from teacher: Continuing to student.")
        return "student"
    else:
        print(f"Routing from teacher: N loops ({N_LOOPS}) completed. Ending conversation.")
        return "end"


workflow.add_edge(START, "initial")
workflow.add_edge("initial", "student")

# Add conditional edges from student
workflow.add_conditional_edges(
    "student",
    route_from_student,
    {
        "teacher": "teacher",
        "observer": "observer",
        "end": END
    }
)

# Add conditional edges from teacher
workflow.add_conditional_edges(
    "teacher",
    route_from_teacher,
    {
        "student": "student",
        "observer": "observer",
        "end": END
    }
)

# Add edges from observer back to student or END
def route_from_observer(state: AgentState):
    current_turn = state["current_turn"]

    if "error" in state:
        print(f"Routing from observer: Error occurred. Ending conversation.")
        return "end"

    if current_turn >= N_LOOPS:
        print("Routing from observer: N loops completed. Ending.")
        return "end"
    else:
        print("Routing from observer: Insights provided. Returning to student.")
        return "student"

workflow.add_conditional_edges(
    "observer",
    route_from_observer,
    {
        "student": "student",
        "end": END
    }
)


# Compile the graph
graph = workflow.compile()


# if __name__ == "__main__":
#     input_messages = [HumanMessage(content="lets discuss about this paper")]
#     for chunk in graph.stream({"messages": input_messages}, stream_mode="values"):
#         chunk["messages"][-1].pretty_print()
    
