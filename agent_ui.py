"""
Streamlit UI for LangGraph ReAct Agent
========================================

This app provides a simple interface to interact with the LangGraph agent
and visualize the tool calls in real-time.

To run:
    streamlit run streamlit_app.py
"""

import streamlit as st
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_aws import ChatBedrock


# ============================================================================
# AGENT SETUP (Same as teaching example)
# ============================================================================

class State(TypedDict):
    """Agent state containing conversation messages"""
    messages: Annotated[list, add_messages]


@tool
def get_weather(location: str):
    """Retrieves the weather for a given location"""
    if location.lower() in ["new york", "nyc"]:
        return "The weather in New York is sunny with a high of 75°F."
    else:
        return f"Unable to retrieve weather information for {location}. Try another location (Use New York for testing)."


@tool
def calculator(expression: str):
    """Evaluates a mathematical expression"""
    try:
        result = eval(expression, {"__builtins__": {}})
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"


# List of tools
tools = [get_weather, calculator]

# Initialize LLM
llm = ChatBedrock(
    model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
    region_name="us-east-1",
    model_kwargs={
        "temperature": 0,
        "max_tokens": 1024
    }
)

llm_with_tools = llm.bind_tools(tools)


# Define nodes
def agent_node(state: State) -> State:
    """Agent decides what to do next"""
    new_message = llm_with_tools.invoke(state["messages"])
    return {"messages": [new_message]}


tool_node = ToolNode(tools)


# Routing logic
def should_continue(state: State) -> Literal["tools", "__end__"]:
    """Decide if we need to call tools or finish"""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    else:
        return "__end__"


# Build graph
graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "__end__": END})
graph.add_edge("tools", "agent")

# Compile the app
agent_app = graph.compile()


# ============================================================================
# STREAMLIT UI
# ============================================================================

st.set_page_config(
    page_title="LangGraph ReAct Agent",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 LangGraph ReAct Agent Demo")


# Sidebar with info
with st.sidebar:
    st.header("📚 Add stuff here?")
    
    
    if st.button("🔄 Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

# Initialize session state for conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
chat_container = st.container()

with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        elif msg["role"] == "assistant":
            with st.chat_message("assistant"):
                st.write(msg["content"])
        elif msg["role"] == "tool":
            with st.expander(f"🔧 Tool: {msg['tool_name']}", expanded=False):
                st.code(f"Input: {msg['input']}", language="python")
                st.code(f"Output: {msg['output']}", language="text")

# Input area
user_input = st.text_input(
    "Your question:",
    placeholder="e.g., What's the weather in New York? or What is 100 / 5?",
    key="user_input"
)

col1, col2 = st.columns([1, 5])
with col1:
    submit_button = st.button("🚀 Send", type="primary", use_container_width=True)

if submit_button and user_input:
    # Add user message to history
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    
    # Create system message
    system_message = SystemMessage(content="""
    You are a helpful assistant with access to weather and calculator tools.
    Use the appropriate tool based on the user's question.
    Be conversational and friendly in your responses.
    """)
    
    # Prepare messages for agent
    agent_messages = [system_message]
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            agent_messages.append(HumanMessage(content=msg["content"]))
    
    # Show processing status
    with st.spinner("🤔 Agent is thinking..."):
        # Stream through the graph to capture intermediate steps
        result = None
        tool_calls_made = []
        
        # Invoke the agent
        result = agent_app.invoke({"messages": agent_messages})
        
        # Extract tool calls from the conversation
        for message in result["messages"]:
            # Check if this is an AI message with tool calls
            if isinstance(message, AIMessage) and hasattr(message, 'tool_calls') and message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_calls_made.append({
                        "tool_name": tool_call["name"],
                        "input": tool_call["args"]
                    })
            
            # Check if this is a tool message (result)
            if hasattr(message, 'name') and message.name:
                # Find the corresponding tool call and add the output
                for tc in tool_calls_made:
                    if tc["tool_name"] == message.name and "output" not in tc:
                        tc["output"] = message.content
                        break
        
        # Add tool calls to session state
        for tool_call in tool_calls_made:
            st.session_state.messages.append({
                "role": "tool",
                "tool_name": tool_call["tool_name"],
                "input": tool_call["input"],
                "output": tool_call.get("output", "No output")
            })
        
        # Get final response
        final_response = result["messages"][-1].content
        
        # Add assistant response to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": final_response
        })
    
    # Rerun to update the display
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    Built with LangGraph + Streamlit | 
    <a href='https://python.langchain.com/docs/langgraph' target='_blank'>LangGraph Docs</a>
</div>
""", unsafe_allow_html=True)