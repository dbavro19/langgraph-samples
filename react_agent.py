"""
LangGraph ReAct Agent - Teaching Example
==========================================

This is a simple ReAct (Reasoning + Acting) agent that:
1. Receives a user question
2. Decides if it needs to use tools
3. Calls tools if needed
4. Returns a final answer

Current capabilities: Weather information for New York
"""

# ============================================================================
# IMPORTS
# ============================================================================
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_aws import ChatBedrock
from weather_tool import get_weather


# ============================================================================
# STEP 1: DEFINE THE STATE
# ============================================================================
# The State is the data structure that flows through the entire graph.
# - messages: Stores the conversation history (user, AI, tool results)
# - add_messages: Special reducer that automatically appends new messages

class State(TypedDict):
    """
    The state of our agent graph.
    
    Attributes:
        messages: List of all messages in the conversation.
                  The Annotated[list, add_messages] means new messages
                  get automatically appended to the list.
    """
    messages: Annotated[list, add_messages]


# ============================================================================
# STEP 2: DEFINE TOOLS
# ============================================================================
# Tools are functions that the LLM can call to get information or perform actions.
# The @tool decorator makes them compatible with LangChain/LangGraph.



# ----------------------------------------------------------------------------
# 📝 HOW TO ADD A NEW TOOL
# ----------------------------------------------------------------------------
# Uncomment the code below to add a calculator tool to the agent:
#
# @tool
# def calculator(expression: str):
#     """Evaluates a mathematical expression.
#     
#     Args:
#         expression: A mathematical expression to evaluate (e.g., "2 + 2", "10 * 5")
#         
#     Returns:
#         The result of the calculation
#     """
#     try:
#         # Safe evaluation of simple math expressions
#         result = eval(expression, {"__builtins__": {}})
#         return f"The result of {expression} is {result}"
#     except Exception as e:
#         return f"Error calculating {expression}: {str(e)}"
#
# Then update the tools list below to include it:
# tools = [get_weather, calculator]  # ← Add calculator here
# ----------------------------------------------------------------------------

# List of all available tools
tools = [get_weather]
# To add more tools: tools = [get_weather, calculator, another_tool]


# ============================================================================
# STEP 3: INITIALIZE THE LLM
# ============================================================================
# We use AWS Bedrock to access Claude models.
# The LLM will be the "brain" that decides when to use tools.

llm = ChatBedrock(
    model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
    region_name="us-east-1",
    model_kwargs={
        "temperature": 0,      # 0 = deterministic, 1 = creative
        "max_tokens": 1024     # Maximum length of response
    }
)

# Bind tools to the LLM so it knows what tools are available
# This tells the LLM about the tools and enables it to call them
llm_with_tools = llm.bind_tools(tools)


# ============================================================================
# STEP 4: CREATE THE GRAPH
# ============================================================================
# Initialize the graph with our State schema
graph = StateGraph(State)


# ============================================================================
# STEP 5: DEFINE NODES
# ============================================================================
# Nodes are the individual steps in our workflow.
# Each node receives the current state and returns an updated state.

def agent_node(state: State) -> State:
    """
    The Agent Node - The "Brain" of the ReAct loop
    
    This node:
    1. Receives the current conversation state
    2. Sends all messages to the LLM
    3. LLM decides: "Do I need a tool?" or "Can I answer directly?"
    4. Returns the LLM's decision (either a tool call or a final answer)
    
    Args:
        state: Current state containing all messages
        
    Returns:
        Updated state with the LLM's response
    """
    new_message = llm_with_tools.invoke(state["messages"])
    return {"messages": [new_message]}


# ToolNode is a built-in LangGraph component that:
# 1. Looks at the last message for tool calls
# 2. Executes the appropriate tool functions
# 3. Returns the tool results as messages
tool_node = ToolNode(tools)


# ============================================================================
# STEP 6: ADD NODES TO THE GRAPH
# ============================================================================
# Register our nodes with the graph

graph.add_node("agent", agent_node)      # The thinking/decision node
graph.add_node("tools", tool_node)       # The tool execution node


# ============================================================================
# STEP 7: DEFINE ROUTING LOGIC
# ============================================================================
# This function determines where to go after the agent thinks

def should_continue(state: State) -> Literal["tools", "__end__"]:
    """
    Conditional Edge - Decides the next step
    
    After the agent node runs, this function decides:
    - If LLM wants to use a tool → route to "tools" node
    - If LLM has a final answer → route to "__end__" (finish)
    
    Args:
        state: Current state containing all messages
        
    Returns:
        "tools" if we need to execute tools, "__end__" if we're done
    """
    last_message = state["messages"][-1]
    
    # Check if the last message has tool calls
    if last_message.tool_calls:
        return "tools"  # LLM wants to use a tool
    else:
        return "__end__"  # LLM has a final answer, we're done


# ============================================================================
# STEP 8: CONNECT THE NODES WITH EDGES
# ============================================================================
# Edges define how the graph flows from one node to another

# Set the starting point of the graph
graph.set_entry_point("agent")

# After the agent node, use conditional routing
graph.add_conditional_edges(
    "agent",           # From this node
    should_continue,   # Use this function to decide where to go
    {
        "tools": "tools",   # If function returns "tools", go to tools node
        "__end__": END      # If function returns "__end__", finish
    }
)

# After tools execute, always return to the agent
# This creates the ReAct loop: agent → tools → agent → tools → ...
graph.add_edge("tools", "agent")


# ============================================================================
# STEP 9: COMPILE THE GRAPH
# ============================================================================
# Compiling converts our graph definition into an executable application

app = graph.compile()

# Optional: Visualize the graph structure
# Uncomment to see a diagram of your graph:
# from IPython.display import Image, display
# display(Image(app.get_graph().draw_mermaid_png()))


# ============================================================================
# STEP 10: RUN THE AGENT
# ============================================================================

# Create a system message to set the agent's behavior
system_message = SystemMessage(content="""
You are a helpful weather assistant. When users ask about weather,
use the get_weather tool to retrieve current information.
Be conversational and friendly in your responses.
""")

# Example 1: Weather query
print("=" * 70)
print("EXAMPLE 1: Weather Query")
print("=" * 70)

result = app.invoke({
    "messages": [
        system_message,
        HumanMessage(content="What is the weather in New York?")
    ]
})

print("\nUser: What is the weather in New York?")
print(f"Assistant: {result['messages'][-1].content}")


# ----------------------------------------------------------------------------
# Example 2: Test the agent without needing a tool
# ----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("EXAMPLE 2: Simple Query (No Tool Needed)")
print("=" * 70)

result = app.invoke({
    "messages": [
        system_message,
        HumanMessage(content="Hello! Can you help me with weather information?")
    ]
})

print("\nUser: Hello! Can you help me with weather information?")
print(f"Assistant: {result['messages'][-1].content}")


# ----------------------------------------------------------------------------
# 📝 EXAMPLE 3: UNCOMMENT TO TEST CALCULATOR TOOL (After adding it above)
# ----------------------------------------------------------------------------
# print("\n" + "=" * 70)
# print("EXAMPLE 3: Calculator Query")
# print("=" * 70)
#
# calculator_system = SystemMessage(content="""
# You are a helpful assistant with access to weather and calculator tools.
# Use the appropriate tool based on the user's question.
# """)
#
# result = app.invoke({
#     "messages": [
#         calculator_system,
#         HumanMessage(content="What is 25 * 4?")
#     ]
# })
#
# print("\nUser: What is 25 * 4?")
# print(f"Assistant: {result['messages'][-1].content}")


print("\n" + "=" * 70)
print("Done! Check the code comments to learn how to add more tools.")
print("=" * 70)


# ============================================================================
# SUMMARY: HOW TO EXTEND THIS AGENT
# ============================================================================
"""
To add a new tool to this agent:

1. Define the tool function with @tool decorator (see STEP 2)
   - Write a clear docstring (LLM uses this to understand the tool)
   - Define parameters with type hints
   
2. Add the tool to the tools list
   - tools = [get_weather, your_new_tool]
   
3. Update the system message if needed
   - Tell the agent about the new capability
   
4. That's it! The graph automatically handles:
   - Tool discovery (LLM knows about it via bind_tools)
   - Tool execution (ToolNode executes it)
   - Routing (conditional edge decides when to use it)

The ReAct loop pattern means the agent can:
- Use multiple tools in sequence
- Decide which tool to use based on context
- Chain tool calls together
- Respond directly when no tools are needed
"""