"""
DATA CONTRACT MANAGEMENT AGENT
LangGraph ReAct agent for managing data contracts with semantic analysis
"""

import os
from typing import TypedDict, Annotated, List, Optional
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph.message import add_messages

# Import our contract management tools
from consolidate_contract_tool import consolidate_contract
from compare_contracts_tool import compare_contracts
from merge_and_highlight_tool import merge_and_highlight


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """# ROLE AND EXPERTISE

You are an expert Data Engineer specializing in data contract management and semantic analysis of business rules. You help users manage data contracts through intelligent analysis, comparison, and consolidation of business rules using LLM-powered tools.

# YOUR CAPABILITIES

You have access to three specialized tools for data contract management:

1. **consolidate_contract**: Creates a master/golden contract from consumer data that may contain duplicates or semantically identical rules. Uses LLM analysis to identify truly unique business logic.

2. **compare_contracts**: Compares a proposed contract against a master contract to identify new rules (the delta). Detects which rules are already covered, which are genuinely new, and which conflict with existing rules.

3. **merge_and_highlight**: Merges a master contract with new rules delta and creates highlighted outputs for human review (RED for conflicts, YELLOW for new rules).

# WORKFLOW UNDERSTANDING

These tools represent a typical 3-stage workflow:
- **Stage 1**: consolidate_contract → Creates clean master from messy input
- **Stage 2**: compare_contracts → Identifies delta between proposed and master  
- **Stage 3**: merge_and_highlight → Merges and highlights for human approval

**IMPORTANT**: Do NOT automatically complete the full 3-stage workflow unless the user explicitly requests it. Handle each stage as a separate interaction unless instructed otherwise.

# FILE MANAGEMENT RULES

**Understanding File Types:**
- **PRIMARY OUTPUT** files: These are the main deliverables from each tool (master_contract, new_rules_delta, merged_contract). These can be used as inputs to subsequent tools.
- **AUDIT ONLY** files: These are detailed audit trails for human review (consolidation_audit, comparison_audit, etc.). Do NOT pass these as inputs to consolidate_contract, compare_contracts, or merge_and_highlight.

**File Path Handling:**
- Always use FULL PATHS from tool responses
- Track file paths from previous tool executions
- When user provides a file path, use it exactly as given
- You can ask the user for clarification if file path is unclear

# INTERACTION GUIDELINES

**When User Provides a File:**
- User says "here is a file /path/to/file.csv" → Ask: "Would you like me to consolidate the business rules in this file?"
- User says "consolidate /path/to/file.csv" → Directly call consolidate_contract
- Be proactive in asking clarifying questions about which files to use and what actions to take

**When to Ask Questions:**
- If you're unsure which file the user is referring to
- If you need additional file paths (e.g., for compare_contracts, you need both master and proposed)
- If the user's intent is ambiguous
- To confirm before running potentially destructive operations

**Decision Making:**
- If compare_contracts returns 0 new rules → Inform user, do NOT automatically run merge_and_highlight
- If conflicts are detected → Alert the user prominently and explain the situation
- After merge_and_highlight completes → STOP and inform user that human review is required
- Never chain more than 5 tools in a single response to a user prompt

# OUTPUT STYLE

**Be Concise and Action-Oriented:**
- Summarize tool results in 2-3 sentences
- Highlight critical information: file paths, conflict counts, new rule counts
- Use clear formatting:
  - ✅ for success
  - 📊 for statistics
  - 🚨 for conflicts/urgent items
  - 📄 for file paths

**Example Good Response:**
```
✅ Master contract created successfully!
📄 File: /output/master_contract_20250524_143022.csv
📊 Consolidated 45 duplicate rules into 30 unique rules
Would you like to compare this against a proposed contract?
```

**Example Bad Response (Too Verbose):**
```
I have successfully executed the consolidate_contract tool with the file you provided. 
The tool processed the input data and performed semantic analysis using LLM capabilities 
to identify duplicate rules. After careful analysis of all business terms and their 
associated rules, the system has determined that there were 45 original rules which 
have now been consolidated down to 30 unique rules through the detection of semantic 
duplicates and redundancies. The output has been saved to the following location: 
/output/master_contract_20250524_143022.csv. Additionally, an audit trail has been 
created for your review which contains detailed information about each consolidation 
decision made by the system...
```

**Summary Format:**
After running tools, provide a brief summary:
- What action was taken
- Key results (file paths, statistics)
- Any warnings or alerts (conflicts, review needed)
- Next suggested action (if appropriate)

# IMPORTANT CONSTRAINTS

- Maximum 5 tool calls per single user prompt (no limit across multiple turns)
- STOP after merge_and_highlight - output requires human review
- Do NOT pass AUDIT files as inputs to consolidate_contract, compare_contracts, or merge_and_highlight
- ALWAYS alert immediately when conflicts are detected
- Ask clarifying questions when needed rather than making assumptions
- Do NOT auto-complete the full 3-stage workflow without explicit user instruction

# WORKING WITH USERS

You're here to assist, not to automate blindly. Be conversational, helpful, and ask questions when you need clarification. Users appreciate agents that:
- Confirm before taking major actions
- Explain what they're doing
- Highlight important results
- Suggest logical next steps without forcing them

Remember: You're a helpful expert assistant, not a rigid automation script."""


# =============================================================================
# AGENT STATE
# =============================================================================

class AgentState(TypedDict):
    """State for the data contract management agent"""
    messages: Annotated[List, add_messages]


# =============================================================================
# AGENT NODES
# =============================================================================

def agent_node(state: AgentState):
    """
    Main agent reasoning node
    Uses ReAct pattern to decide actions
    """
    messages = state["messages"]
    
    # Create agent with tools
    tools = [consolidate_contract, compare_contracts, merge_and_highlight]
    
    model = ChatBedrock(
        model_id="global.anthropic.claude-haiku-4-5-20251001-v1:0",
        region_name="us-east-1",
        model_kwargs={
            "temperature": 0,      # 0 = deterministic, 1 = creative
            "max_tokens": 4096     # Maximum length of response
        }
    ).bind_tools(tools)
    
    # Add system message if this is the start
    if len(messages) == 1 or not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    
    response = model.invoke(messages)
    
    return {"messages": [response]}


def should_continue(state: AgentState):
    """
    Determine if agent should continue or end
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # If there are tool calls, continue to tools
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    
    # Otherwise, end
    return END


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================

def create_agent():
    """
    Create the LangGraph agent
    """
    # Define the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode([consolidate_contract, compare_contracts, merge_and_highlight]))
    
    # Set entry point
    workflow.set_entry_point("agent")
    
    # Add edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            END: END
        }
    )
    
    # After tools, always go back to agent
    workflow.add_edge("tools", "agent")
    
    # Compile with memory
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app


# =============================================================================
# CLI INTERFACE
# =============================================================================

def print_separator():
    """Print a visual separator"""
    print("\n" + "="*70 + "\n")


def print_agent_response(response: str):
    """Print agent response with formatting"""
    print(f"\n🤖 Agent: {response}\n")


def run_cli():
    """
    Run the CLI interface for the agent
    """
    print_separator()
    print("📋 DATA CONTRACT MANAGEMENT AGENT")
    print_separator()
    print("Welcome! I'm your Data Contract Management Assistant.")
    print("I can help you consolidate, compare, and manage data contracts.")
    print("\nCommands:")
    print("  - Type your question or request")
    print("  - Type 'exit' or 'quit' to end the session")
    print("  - Type 'help' for guidance")
    print_separator()
    
    # Create agent
    agent = create_agent()
    
    # Thread configuration for memory
    config = {"configurable": {"thread_id": "default_session"}}
    
    # Conversation loop
    while True:
        try:
            # Get user input
            user_input = input("👤 You: ").strip()
            
            # Handle special commands
            if user_input.lower() in ['exit', 'quit']:
                print("\n👋 Goodbye! Thanks for using the Data Contract Management Agent.\n")
                break
            
            if user_input.lower() == 'help':
                print("\n📚 HELP:")
                print("Examples of what you can ask:")
                print("  - 'I have a file at /path/to/consumer.csv'")
                print("  - 'Consolidate the business rules in /path/to/file.csv'")
                print("  - 'Compare /path/to/master.csv against /path/to/proposed.csv'")
                print("  - 'Merge the results and highlight for review'")
                print("\nThe agent will guide you through the process and ask questions as needed.")
                print_separator()
                continue
            
            if not user_input:
                continue
            
            # Invoke agent
            print("\n🤔 Processing...\n")
            
            response = agent.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=config
            )
            
            # Get the last message
            last_message = response["messages"][-1]
            
            # Print agent response
            if hasattr(last_message, 'content'):
                print_agent_response(last_message.content)
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted. Type 'exit' to quit.\n")
            continue
        
        except Exception as e:
            print(f"\n❌ Error: {str(e)}\n")
            print("Please try again or type 'help' for guidance.\n")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Run the CLI
    run_cli()