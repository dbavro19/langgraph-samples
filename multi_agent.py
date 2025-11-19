"""
Supervisor Multi-Agent System with MCP Integration
===================================================

This demonstrates:
1. A Supervisor agent that routes tasks to specialized workers
2. Two worker agents:
   - Research Agent (uses web search tools)
   - File Agent (uses MCP server for file operations)
3. MCP (Model Context Protocol) integration

Architecture:
                    SUPERVISOR
                         |
          +--------------+--------------+
          |                             |
    RESEARCH AGENT              FILE AGENT
    (Web tools)                 (MCP Server)

"""

from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_aws import ChatBedrock
from langchain_core.tools import tool
import json


# ============================================================================
# STEP 1: DEFINE STATE
# ============================================================================

class SupervisorState(TypedDict):
    """
    State for the Supervisor system
    
    Attributes:
        messages: Conversation history
        next_agent: Which agent should work next
        task_description: What needs to be done
        agent_results: Results from each agent
        final_answer: Complete response
    """
    messages: Annotated[list, add_messages]
    next_agent: str
    task_description: str
    agent_results: dict
    final_answer: str


# ============================================================================
# STEP 2: INITIALIZE LLM
# ============================================================================

llm = ChatBedrock(
    model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
    region_name="us-east-1",
    model_kwargs={
        "temperature": 0,
        "max_tokens": 2048
    }
)


# ============================================================================
# STEP 3: DEFINE TOOLS FOR AGENTS
# ============================================================================

# ──────────────────────────────────────────────────────────────────────────
# Research Agent Tools (Simple web-like tools)
# ──────────────────────────────────────────────────────────────────────────

@tool
def search_web(query: str) -> str:
    """
    Search the web for information
    
    Args:
        query: Search query
        
    Returns:
        Search results (simulated)
    """
    # Simulated search results
    search_db = {
        "aws": "AWS (Amazon Web Services) is a cloud platform offering 200+ services including computing, storage, and AI/ML.",
        "python": "Python is a high-level programming language known for simplicity and readability. Popular for data science and AI.",
        "langgraph": "LangGraph is a framework for building stateful, multi-actor applications with LLMs, created by LangChain.",
        "bedrock": "Amazon Bedrock is a fully managed service providing access to foundation models via API.",
    }
    
    # Simple keyword matching
    for keyword, info in search_db.items():
        if keyword.lower() in query.lower():
            return f"Search results for '{query}':\n{info}"
    
    return f"Search results for '{query}':\nNo specific information found. Try: aws, python, langgraph, or bedrock."


@tool
def get_latest_news(topic: str) -> str:
    """
    Get latest news about a topic
    
    Args:
        topic: Topic to get news about
        
    Returns:
        Latest news (simulated)
    """
    news_db = {
        "ai": "Latest AI News: New foundation models released, focus on efficiency and multimodal capabilities.",
        "cloud": "Cloud News: Major providers expanding edge computing and serverless offerings.",
        "technology": "Tech News: Growth in AI adoption across enterprises, focus on responsible AI.",
    }
    
    for keyword, news in news_db.items():
        if keyword.lower() in topic.lower():
            return news
    
    return f"Latest news about {topic}: General technology updates available."


# ──────────────────────────────────────────────────────────────────────────
# File Agent Tools (MCP Server Simulation)
# ──────────────────────────────────────────────────────────────────────────
# NOTE: In production, these would connect to an actual MCP server
# For this demo, we simulate MCP server responses

class MCPFileServer:
    """
    Simulated MCP Server for file operations
    
    In production, this would be a real MCP server connection:
    - MCP servers expose resources and tools via standard protocol
    - Clients connect over stdio or HTTP
    - Provides structured tool definitions
    
    Example real MCP server: filesystem MCP, database MCP, API MCP
    """
    
    def __init__(self):
        # Simulated file system
        self.files = {
            "notes.txt": "Meeting notes: Discussed Q4 roadmap and new features.",
            "data.csv": "product,sales\nLaptop,1500\nPhone,2300",
            "report.md": "# Q3 Report\nRevenue up 25% YoY.",
        }
    
    def list_files(self) -> str:
        """List available files"""
        files = list(self.files.keys())
        return f"Available files: {', '.join(files)}"
    
    def read_file(self, filename: str) -> str:
        """Read a file"""
        if filename in self.files:
            return f"Contents of {filename}:\n{self.files[filename]}"
        else:
            return f"File '{filename}' not found. Available: {', '.join(self.files.keys())}"
    
    def write_file(self, filename: str, content: str) -> str:
        """Write to a file"""
        self.files[filename] = content
        return f"Successfully wrote to {filename}"
    
    def search_in_files(self, keyword: str) -> str:
        """Search for keyword across all files"""
        results = []
        for filename, content in self.files.items():
            if keyword.lower() in content.lower():
                results.append(f"{filename}: ...{content[:50]}...")
        
        if results:
            return f"Found '{keyword}' in:\n" + "\n".join(results)
        else:
            return f"No files contain '{keyword}'"


# Initialize MCP server (simulated)
mcp_server = MCPFileServer()


@tool
def mcp_list_files() -> str:
    """
    List all available files (via MCP server)
    
    Returns:
        List of files
    """
    return mcp_server.list_files()


@tool
def mcp_read_file(filename: str) -> str:
    """
    Read a file (via MCP server)
    
    Args:
        filename: Name of file to read
        
    Returns:
        File contents
    """
    return mcp_server.read_file(filename)


@tool
def mcp_write_file(filename: str, content: str) -> str:
    """
    Write to a file (via MCP server)
    
    Args:
        filename: Name of file to write
        content: Content to write
        
    Returns:
        Success message
    """
    return mcp_server.write_file(filename, content)


@tool
def mcp_search_files(keyword: str) -> str:
    """
    Search for keyword in all files (via MCP server)
    
    Args:
        keyword: Keyword to search for
        
    Returns:
        Search results
    """
    return mcp_server.search_in_files(keyword)


# ============================================================================
# STEP 4: SUPERVISOR NODE
# ============================================================================

def supervisor_node(state: SupervisorState) -> SupervisorState:
    """
    SUPERVISOR - Routes tasks to appropriate worker agents
    
    This node:
    1. Analyzes the user's request
    2. Decides which agent is best suited for the task
    3. Routes to that agent
    
    Returns:
        Updated state with routing decision
    """
    print("\n" + "="*70)
    print("👔 SUPERVISOR: Analyzing request and delegating...")
    print("="*70)
    
    messages = state["messages"]
    agent_results = state.get("agent_results", {})
    
    # Check if we have results from agents
    if agent_results:
        print("   All agents have completed their work")
        print(f"   Results collected from: {list(agent_results.keys())}")
        return {
            "messages": messages,
            "next_agent": "finish",
            "task_description": state.get("task_description", ""),
            "agent_results": agent_results,
            "final_answer": ""
        }
    
    # Get the user's request
    last_message = messages[-1].content if messages else ""
    
    # Create supervisor decision prompt
    supervisor_prompt = SystemMessage(content="""
You are a supervisor managing two specialized agents:

1. RESEARCH AGENT
   - Searches the web for information
   - Gets latest news and updates
   - Good for: questions about topics, general information, news

2. FILE AGENT
   - Manages files via MCP server
   - Can list, read, write, and search files
   - Good for: file operations, document retrieval, content search

Based on the user's request, decide which agent should handle it.
Respond with ONLY the agent name: "research" or "file"

If unclear, choose "research" as default.
""")
    
    # Get decision from LLM
    decision_messages = [supervisor_prompt, HumanMessage(content=f"User request: {last_message}")]
    response = llm.invoke(decision_messages)
    
    agent_choice = response.content.strip().lower()
    
    # Validate choice
    if "file" in agent_choice:
        agent_choice = "file"
    else:
        agent_choice = "research"
    
    print(f"\n   📋 Decision: Delegate to {agent_choice.upper()} AGENT")
    print(f"   📝 Task: {last_message[:60]}...")
    
    return {
        "messages": messages,
        "next_agent": agent_choice,
        "task_description": last_message,
        "agent_results": agent_results,
        "final_answer": ""
    }


# ============================================================================
# STEP 5: WORKER AGENT NODES
# ============================================================================

def research_agent_node(state: SupervisorState) -> SupervisorState:
    """
    RESEARCH AGENT - Searches web and gathers information
    
    Tools available:
    - search_web
    - get_latest_news
    """
    print("\n" + "-"*70)
    print("🔍 RESEARCH AGENT: Working on task...")
    print("-"*70)
    
    messages = state["messages"]
    task = state["task_description"]
    agent_results = state["agent_results"]
    
    print(f"Task: {task}")
    
    # Use LLM to decide which tool to use
    research_prompt = SystemMessage(content="""
You are a research agent with these tools:
- search_web(query) - Search for general information
- get_latest_news(topic) - Get recent news

Based on the task, decide which tool to use and extract parameters.

Respond with JSON:
{
    "tool": "search_web or get_latest_news",
    "query": "search query or topic"
}
""")
    
    tool_decision = llm.invoke([
        research_prompt,
        HumanMessage(content=f"Task: {task}")
    ])
    
    # Parse tool decision
    try:
        content = tool_decision.content
        if '{' in content:
            start = content.index('{')
            end = content.rindex('}') + 1
            tool_info = json.loads(content[start:end])
        else:
            tool_info = {"tool": "search_web", "query": task}
        
        print(f"   🔧 Using tool: {tool_info['tool']}")
        print(f"   📥 Query: {tool_info['query']}")
        
        # Execute the tool
        if tool_info["tool"] == "search_web":
            result = search_web(tool_info["query"])
        else:
            result = get_latest_news(tool_info["query"])
        
        print(f"   ✅ Result obtained:\n   {result[:100]}...")
        
        # Store result
        agent_results["research"] = result
        
        return {
            "messages": messages,
            "next_agent": "return_to_supervisor",
            "task_description": task,
            "agent_results": agent_results,
            "final_answer": ""
        }
    
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        agent_results["research"] = f"Error: {str(e)}"
        return {
            "messages": messages,
            "next_agent": "return_to_supervisor",
            "task_description": task,
            "agent_results": agent_results,
            "final_answer": ""
        }


def file_agent_node(state: SupervisorState) -> SupervisorState:
    """
    FILE AGENT - Handles file operations via MCP server
    
    Tools available (via MCP):
    - mcp_list_files
    - mcp_read_file
    - mcp_write_file
    - mcp_search_files
    """
    print("\n" + "-"*70)
    print("📁 FILE AGENT: Working on task (via MCP server)...")
    print("-"*70)
    
    messages = state["messages"]
    task = state["task_description"]
    agent_results = state["agent_results"]
    
    print(f"Task: {task}")
    print("   🔌 Connected to MCP File Server")
    
    # Use LLM to decide which MCP tool to use
    file_prompt = SystemMessage(content="""
You are a file agent with MCP server tools:
- mcp_list_files() - List all files
- mcp_read_file(filename) - Read a specific file
- mcp_write_file(filename, content) - Write to a file
- mcp_search_files(keyword) - Search for keyword in files

Based on the task, decide which tool to use.

Respond with JSON:
{
    "tool": "tool_name",
    "filename": "filename or null",
    "content": "content or null",
    "keyword": "keyword or null"
}
""")
    
    tool_decision = llm.invoke([
        file_prompt,
        HumanMessage(content=f"Task: {task}")
    ])
    
    # Parse tool decision
    try:
        content = tool_decision.content
        if '{' in content:
            start = content.index('{')
            end = content.rindex('}') + 1
            tool_info = json.loads(content[start:end])
        else:
            tool_info = {"tool": "mcp_list_files"}
        
        print(f"   🔧 Using MCP tool: {tool_info['tool']}")
        
        # Execute the MCP tool
        tool_name = tool_info.get("tool", "mcp_list_files")
        
        if tool_name == "mcp_list_files":
            result = mcp_list_files()
        elif tool_name == "mcp_read_file":
            filename = tool_info.get("filename", "notes.txt")
            print(f"   📄 Reading: {filename}")
            result = mcp_read_file(filename)
        elif tool_name == "mcp_write_file":
            filename = tool_info.get("filename", "output.txt")
            content = tool_info.get("content", "")
            print(f"   ✍️ Writing to: {filename}")
            result = mcp_write_file(filename, content)
        elif tool_name == "mcp_search_files":
            keyword = tool_info.get("keyword", "")
            print(f"   🔎 Searching for: {keyword}")
            result = mcp_search_files(keyword)
        else:
            result = mcp_list_files()
        
        print(f"   ✅ MCP Result:\n   {result[:100]}...")
        
        # Store result
        agent_results["file"] = result
        
        return {
            "messages": messages,
            "next_agent": "return_to_supervisor",
            "task_description": task,
            "agent_results": agent_results,
            "final_answer": ""
        }
    
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        agent_results["file"] = f"Error: {str(e)}"
        return {
            "messages": messages,
            "next_agent": "return_to_supervisor",
            "task_description": task,
            "agent_results": agent_results,
            "final_answer": ""
        }


# ============================================================================
# STEP 6: RESPONSE GENERATOR NODE
# ============================================================================

def response_generator_node(state: SupervisorState) -> SupervisorState:
    """
    RESPONSE GENERATOR - Creates final answer from agent results
    """
    print("\n" + "="*70)
    print("💬 RESPONSE GENERATOR: Creating final answer...")
    print("="*70)
    
    messages = state["messages"]
    agent_results = state["agent_results"]
    
    # Format results
    results_text = "Agent Results:\n\n"
    for agent_name, result in agent_results.items():
        results_text += f"{agent_name.upper()} AGENT:\n{result}\n\n"
    
    # Generate response
    response_prompt = SystemMessage(content=f"""
Based on the results from the agents, provide a clear, helpful answer to the user's question.

{results_text}

Provide a natural, conversational response that directly answers the question.
""")
    
    final_response = llm.invoke([
        response_prompt,
        messages[-1]  # Original user question
    ])
    
    final_answer = final_response.content
    print(f"✅ Final answer generated")
    
    return {
        "messages": messages + [AIMessage(content=final_answer)],
        "next_agent": "done",
        "task_description": state["task_description"],
        "agent_results": agent_results,
        "final_answer": final_answer
    }


# ============================================================================
# STEP 7: ROUTING LOGIC
# ============================================================================

def route_from_supervisor(state: SupervisorState) -> Literal["research", "file", "finish"]:
    """Router: Where should we go from supervisor?"""
    next_agent = state["next_agent"]
    
    print(f"\n🔀 ROUTING: Supervisor decided → {next_agent}")
    
    if next_agent == "research":
        return "research"
    elif next_agent == "file":
        return "file"
    else:
        return "finish"


def route_from_agent(state: SupervisorState) -> Literal["supervisor", "finish"]:
    """Router: Where should we go after agent completes?"""
    next_agent = state["next_agent"]
    
    if next_agent == "return_to_supervisor":
        print("🔀 ROUTING: Agent done → returning to supervisor")
        return "supervisor"
    else:
        return "finish"


# ============================================================================
# STEP 8: BUILD THE GRAPH
# ============================================================================

def create_supervisor_graph():
    """Build the Supervisor Multi-Agent graph"""
    
    graph = StateGraph(SupervisorState)
    
    # Add nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("research", research_agent_node)
    graph.add_node("file", file_agent_node)
    graph.add_node("responder", response_generator_node)
    
    # Set entry point
    graph.set_entry_point("supervisor")
    
    # Supervisor routes to workers or finish
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "research": "research",
            "file": "file",
            "finish": "responder"
        }
    )
    
    # Workers return to supervisor
    graph.add_conditional_edges(
        "research",
        route_from_agent,
        {
            "supervisor": "supervisor",
            "finish": "responder"
        }
    )
    
    graph.add_conditional_edges(
        "file",
        route_from_agent,
        {
            "supervisor": "supervisor",
            "finish": "responder"
        }
    )
    
    # Responder ends the workflow
    graph.add_edge("responder", END)
    
    return graph.compile()


# ============================================================================
# STEP 9: RUN EXAMPLES
# ============================================================================

def main():
    """Run example queries"""
    
    print("\n" + "🔷"*35)
    print("SUPERVISOR MULTI-AGENT SYSTEM WITH MCP")
    print("🔷"*35 + "\n")
    
    # Create the agent
    agent = create_supervisor_graph()
    
    # Example questions
    questions = [
        "What is LangGraph?",  # Should route to Research Agent
        "Read the notes.txt file",  # Should route to File Agent (MCP)
        # "Search for the word 'roadmap' in all files",  # File Agent (MCP)
    ]
    
    for i, question in enumerate(questions, 1):
        print("\n" + "🔷"*35)
        print(f"QUESTION {i}: {question}")
        print("🔷"*35)
        
        result = agent.invoke({
            "messages": [HumanMessage(content=question)],
            "next_agent": "",
            "task_description": "",
            "agent_results": {},
            "final_answer": ""
        })
        
        print("\n" + "="*70)
        print("🎯 FINAL ANSWER:")
        print("="*70)
        print(result["final_answer"])
        print("\n")


if __name__ == "__main__":
    main()


# ============================================================================
# NOTES ON MCP INTEGRATION
# ============================================================================
"""
🔌 MCP (Model Context Protocol) Integration Notes:

In this example, we SIMULATED an MCP server with the MCPFileServer class.

In PRODUCTION, you would:

1. Run an actual MCP server (e.g., filesystem MCP server):
   ```bash
   npx -y @modelcontextprotocol/server-filesystem /path/to/files
   ```

2. Connect to it using MCP client libraries:
   ```python
   from mcp import MCPClient
   
   client = MCPClient()
   await client.connect_stdio("npx", ["-y", "@modelcontextprotocol/server-filesystem", "/path"])
   
   # List available tools
   tools = await client.list_tools()
   
   # Call a tool
   result = await client.call_tool("read_file", {"path": "notes.txt"})
   ```

3. Wrap MCP tools in LangChain tool format:
   ```python
   @tool
   def mcp_read_file(filename: str) -> str:
       result = await mcp_client.call_tool("read_file", {"path": filename})
       return result["content"]
   ```

Popular MCP Servers:
- Filesystem (read/write files)
- Database (SQL queries)
- Web (fetch URLs, search)
- Git (repo operations)
- Custom (your own tools)

Benefits:
- Standardized tool protocol
- Easy to add new capabilities
- Separation of concerns (tools in separate process)
- Can be written in any language
- Reusable across different AI systems
"""