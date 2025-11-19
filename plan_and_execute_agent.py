"""
Plan-and-Execute Agent for CSV Data Analysis
==============================================

This agent demonstrates the Plan-and-Execute pattern:
1. Planner creates a step-by-step plan
2. Executor runs each step using tools
3. Agent can revise plan based on results

Use case: Answer questions about data in a CSV file

Pattern:
    User Question
         ↓
    PLANNER (creates steps)
         ↓
    EXECUTOR (runs each step)
         ↓
    Response

"""

from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_aws import ChatBedrock
from langchain_core.tools import tool
import pandas as pd
import json


# ============================================================================
# STEP 1: DEFINE STATE
# ============================================================================

class PlanExecuteState(TypedDict):
    """
    State for Plan-and-Execute agent
    
    Attributes:
        messages: Conversation history
        plan: List of steps to execute
        current_step: Which step we're on
        step_results: Results from each completed step
        csv_data: The loaded CSV dataframe
        final_answer: The complete response
    """
    messages: Annotated[list, add_messages]
    plan: List[str]
    current_step: int
    step_results: dict
    csv_data: pd.DataFrame
    final_answer: str


# ============================================================================
# STEP 2: DEFINE TOOLS (For the Executor to use)
# ============================================================================

@tool
def get_column_names(df_json: str) -> str:
    """
    Get the column names from the CSV data
    
    Args:
        df_json: JSON string representation of dataframe (use "placeholder" as value)
    
    Returns:
        List of column names
    """
    # Note: We'll inject the actual dataframe from state
    # For now, this is a placeholder that will be replaced
    return "Columns will be determined from state"


@tool
def get_summary_statistics(df_json: str, column: str = None) -> str:
    """
    Get summary statistics for the data
    
    Args:
        df_json: JSON string representation of dataframe
        column: Optional specific column to analyze
    
    Returns:
        Summary statistics
    """
    return f"Summary statistics for {column if column else 'all columns'}"


@tool
def filter_data(df_json: str, condition: str) -> str:
    """
    Filter the data based on a condition
    
    Args:
        df_json: JSON string representation of dataframe
        condition: Description of filter condition (e.g., "sales > 1000")
    
    Returns:
        Filtered data summary
    """
    return f"Filtered data based on: {condition}"


@tool
def calculate_aggregation(df_json: str, column: str, operation: str) -> str:
    """
    Calculate aggregations on data
    
    Args:
        df_json: JSON string representation of dataframe
        column: Column to aggregate
        operation: Operation (sum, mean, max, min, count)
    
    Returns:
        Aggregation result
    """
    return f"Calculating {operation} of {column}"


# For executor - we'll actually implement these inline with real dataframe
tools = [get_column_names, get_summary_statistics, filter_data, calculate_aggregation]


# ============================================================================
# STEP 3: INITIALIZE LLM
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
# STEP 4: HELPER FUNCTIONS FOR CSV OPERATIONS
# ============================================================================

def execute_tool_on_dataframe(tool_name: str, df: pd.DataFrame, **kwargs) -> str:
    """
    Actually execute tools on the real dataframe
    
    Args:
        tool_name: Name of the tool to execute
        df: The pandas dataframe
        **kwargs: Additional arguments
    
    Returns:
        String result of the operation
    """
    try:
        if tool_name == "get_column_names":
            return f"Columns: {', '.join(df.columns.tolist())}"
        
        elif tool_name == "get_summary_statistics":
            column = kwargs.get("column")
            if column and column in df.columns:
                stats = df[column].describe()
                return f"Statistics for {column}:\n{stats.to_string()}"
            else:
                return f"Summary:\nRows: {len(df)}\nColumns: {len(df.columns)}\n{df.describe().to_string()}"
        
        elif tool_name == "filter_data":
            condition = kwargs.get("condition", "")
            # Simple filtering (in production, use safer evaluation)
            try:
                filtered = df.query(condition) if condition else df
                return f"Filtered to {len(filtered)} rows. Sample:\n{filtered.head().to_string()}"
            except:
                return f"Could not filter with condition: {condition}"
        
        elif tool_name == "calculate_aggregation":
            column = kwargs.get("column")
            operation = kwargs.get("operation", "sum").lower()
            
            if column not in df.columns:
                return f"Column {column} not found"
            
            if operation == "sum":
                result = df[column].sum()
            elif operation == "mean":
                result = df[column].mean()
            elif operation == "max":
                result = df[column].max()
            elif operation == "min":
                result = df[column].min()
            elif operation == "count":
                result = df[column].count()
            else:
                return f"Unknown operation: {operation}"
            
            return f"{operation.capitalize()} of {column}: {result}"
        
        else:
            return f"Unknown tool: {tool_name}"
    
    except Exception as e:
        return f"Error executing {tool_name}: {str(e)}"


# ============================================================================
# STEP 5: PLANNER NODE
# ============================================================================

def planner_node(state: PlanExecuteState) -> PlanExecuteState:
    """
    The Planner Node - Creates a step-by-step plan
    
    This node:
    1. Analyzes the user's question
    2. Looks at what data is available (CSV columns)
    3. Creates a plan (list of steps) to answer the question
    
    Returns:
        Updated state with a plan
    """
    print("\n" + "="*70)
    print("📋 PLANNER: Creating execution plan...")
    print("="*70)
    
    messages = state["messages"]
    df = state["csv_data"]
    
    # Get available columns for context
    columns_info = f"Available columns: {', '.join(df.columns.tolist())}"
    sample_data = f"Sample data (first 3 rows):\n{df.head(3).to_string()}"
    
    # Create planning prompt
    planning_prompt = SystemMessage(content=f"""
You are a data analysis planner. Create a step-by-step plan to answer the user's question about the data.

{columns_info}

{sample_data}

Available operations:
1. get_column_names - See all column names
2. get_summary_statistics - Get stats for specific column or all data
3. filter_data - Filter rows based on conditions
4. calculate_aggregation - Sum, mean, max, min, count of a column

Create a plan as a numbered list of steps. Each step should be one operation.
Be specific about which columns and operations to use.
Keep it simple - typically 2-4 steps is enough.

Example plan format:
1. Get summary statistics for the 'sales' column
2. Calculate the sum of 'revenue' column
3. Filter data where sales > 1000

Now create a plan for the user's question:
""")
    
    # Get plan from LLM
    plan_messages = [planning_prompt] + messages
    response = llm.invoke(plan_messages)
    
    # Parse the plan (extract numbered steps)
    plan_text = response.content
    print(f"\n📝 Generated Plan:\n{plan_text}\n")
    
    # Extract steps (simple parsing - look for numbered lines)
    steps = []
    for line in plan_text.split('\n'):
        line = line.strip()
        # Look for lines starting with numbers
        if line and (line[0].isdigit() or line.startswith('-')):
            # Remove the number/bullet and clean up
            step = line.lstrip('0123456789.-) ').strip()
            if step:
                steps.append(step)
    
    if not steps:
        # Fallback: treat entire response as one step
        steps = [plan_text]
    
    print(f"✅ Plan extracted: {len(steps)} steps")
    for i, step in enumerate(steps, 1):
        print(f"   {i}. {step}")
    
    return {
        "messages": messages,
        "plan": steps,
        "current_step": 0,
        "step_results": {},
        "csv_data": df,
        "final_answer": ""
    }


# ============================================================================
# STEP 6: EXECUTOR NODE
# ============================================================================

def executor_node(state: PlanExecuteState) -> PlanExecuteState:
    """
    The Executor Node - Executes one step from the plan
    
    This node:
    1. Takes the current step from the plan
    2. Determines which tool to use
    3. Executes the tool on the CSV data
    4. Stores the result
    
    Returns:
        Updated state with step results
    """
    plan = state["plan"]
    current_step = state["current_step"]
    step_results = state["step_results"]
    df = state["csv_data"]
    
    print("\n" + "-"*70)
    print(f"🔧 EXECUTOR: Running step {current_step + 1}/{len(plan)}")
    print("-"*70)
    
    # Get the current step
    step_description = plan[current_step]
    print(f"Step: {step_description}")
    
    # Use LLM to parse the step and determine tool + arguments
    parsing_prompt = SystemMessage(content=f"""
Parse this step and extract the tool name and arguments.

Available tools:
- get_column_names (no args needed)
- get_summary_statistics (optional: column name)
- filter_data (requires: condition as string)
- calculate_aggregation (requires: column name, operation: sum/mean/max/min/count)

Respond with ONLY a JSON object:
{{
    "tool": "tool_name",
    "column": "column_name or null",
    "operation": "operation or null",
    "condition": "condition or null"
}}
""")
    
    # FIX: Add the step as a HumanMessage
    parse_response = llm.invoke([
        parsing_prompt,
        HumanMessage(content=f"Step to parse: {step_description}")
    ])
    
    # Parse the JSON response
    try:
        # Extract JSON from response
        content = parse_response.content
        # Find JSON in the response
        if '{' in content and '}' in content:
            start = content.index('{')
            end = content.rindex('}') + 1
            json_str = content[start:end]
            tool_info = json.loads(json_str)
        else:
            # Fallback
            tool_info = {"tool": "get_summary_statistics"}
        
        print(f"🎯 Tool parsed: {tool_info}")
        
        # Execute the tool
        tool_name = tool_info.get("tool", "get_summary_statistics")
        result = execute_tool_on_dataframe(
            tool_name,
            df,
            column=tool_info.get("column"),
            operation=tool_info.get("operation"),
            condition=tool_info.get("condition")
        )
        
        print(f"✅ Result:\n{result}\n")
        
        # Store result
        step_results[f"step_{current_step + 1}"] = {
            "description": step_description,
            "result": result
        }
        
        return {
            "messages": state["messages"],
            "plan": plan,
            "current_step": current_step + 1,  # Move to next step
            "step_results": step_results,
            "csv_data": df,
            "final_answer": ""
        }
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        # Store error and move on
        step_results[f"step_{current_step + 1}"] = {
            "description": step_description,
            "result": f"Error: {str(e)}"
        }
        
        return {
            "messages": state["messages"],
            "plan": plan,
            "current_step": current_step + 1,
            "step_results": step_results,
            "csv_data": df,
            "final_answer": ""
        }


# ============================================================================
# STEP 7: RESPONSE GENERATOR NODE
# ============================================================================

def response_generator_node(state: PlanExecuteState) -> PlanExecuteState:
    """
    The Response Generator Node - Creates final answer from all step results
    
    This node:
    1. Collects all step results
    2. Uses LLM to synthesize a natural language answer
    3. Returns the final response
    """
    print("\n" + "="*70)
    print("💬 RESPONSE GENERATOR: Creating final answer...")
    print("="*70)
    
    messages = state["messages"]
    step_results = state["step_results"]
    
    # Format all step results
    results_text = "Execution Results:\n\n"
    for step_key, step_data in step_results.items():
        results_text += f"{step_key}:\n"
        results_text += f"  Task: {step_data['description']}\n"
        results_text += f"  Result: {step_data['result']}\n\n"
    
    # Create response generation prompt
    response_prompt = SystemMessage(content=f"""
Based on the execution results below, provide a clear, natural language answer to the user's question.

{results_text}

Provide a concise, helpful answer that directly addresses the question.
Include relevant numbers and insights from the results.
""")
    
    # Get final response
    final_messages = [response_prompt] + messages
    response = llm.invoke(final_messages)
    
    final_answer = response.content
    print(f"\n✅ Final Answer Generated:\n{final_answer}\n")
    
    return {
        "messages": state["messages"] + [AIMessage(content=final_answer)],
        "plan": state["plan"],
        "current_step": state["current_step"],
        "step_results": step_results,
        "csv_data": state["csv_data"],
        "final_answer": final_answer
    }


# ============================================================================
# STEP 8: ROUTING LOGIC
# ============================================================================

def should_continue_executing(state: PlanExecuteState) -> str:
    """
    Router: Decide if we should execute another step or generate response
    
    Returns:
        "execute" if more steps remain, "respond" if done
    """
    current_step = state["current_step"]
    plan = state["plan"]
    
    if current_step < len(plan):
        return "execute"
    else:
        return "respond"


# ============================================================================
# STEP 9: BUILD THE GRAPH
# ============================================================================

def create_plan_execute_agent():
    """Build and return the Plan-and-Execute graph"""
    
    graph = StateGraph(PlanExecuteState)
    
    # Add nodes
    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("responder", response_generator_node)
    
    # Set entry point
    graph.set_entry_point("planner")
    
    # After planning, start executing
    graph.add_edge("planner", "executor")
    
    # After each execution, decide: more steps or respond?
    graph.add_conditional_edges(
        "executor",
        should_continue_executing,
        {
            "execute": "executor",  # Loop back for next step
            "respond": "responder"  # Done executing, generate response
        }
    )
    
    # After responding, we're done
    graph.add_edge("responder", END)
    
    return graph.compile()


# ============================================================================
# STEP 10: RUN THE AGENT
# ============================================================================

def main():
    """
    Main execution function with example CSV data
    """
    print("\n" + "🔷"*35)
    print("PLAN-AND-EXECUTE AGENT - CSV ANALYSIS DEMO")
    print("🔷"*35 + "\n")
    
    # ========================================================================
    # CREATE SAMPLE CSV DATA
    # ========================================================================
    # Option 1: Create sample data inline
    sample_data = {
        'product': ['Laptop', 'Phone', 'Tablet', 'Monitor', 'Keyboard', 'Mouse'],
        'category': ['Electronics', 'Electronics', 'Electronics', 'Electronics', 'Accessories', 'Accessories'],
        'sales': [1500, 2300, 890, 450, 120, 80],
        'revenue': [1200000, 1840000, 534000, 180000, 36000, 12000],
        'units_sold': [800, 800, 600, 400, 300, 150]
    }
    df = pd.DataFrame(sample_data)
    
    # Option 2: Load from CSV file (uncomment to use)
    # df = pd.read_csv('your_data.csv')
    
    print("📊 Loaded CSV Data:")
    print(df.to_string())
    print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns\n")
    
    # ========================================================================
    # CREATE THE AGENT
    # ========================================================================
    agent = create_plan_execute_agent()
    
    # ========================================================================
    # EXAMPLE QUESTIONS
    # ========================================================================
    
    questions = [
        #"What is the total revenue across all products?",
        "get me the revenue per sale of Electronics products",
        # "Which product has the highest sales?",
        # "What are the average units sold for Electronics category?",
    ]
    
    for i, question in enumerate(questions, 1):
        print("\n" + "🔷"*35)
        print(f"QUESTION {i}: {question}")
        print("🔷"*35)
        
        # Invoke the agent
        result = agent.invoke({
            "messages": [HumanMessage(content=question)],
            "plan": [],
            "current_step": 0,
            "step_results": {},
            "csv_data": df,
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
# TO USE WITH YOUR OWN CSV FILE
# ============================================================================
"""
To analyze your own CSV file:

1. Place your CSV file in the same directory
2. In the main() function, replace the sample data with:
   
   df = pd.read_csv('your_file.csv')

3. Update the questions list with your questions

4. Run the script:
   python plan_execute_agent.py

Example questions for different datasets:
- Sales data: "What is the total revenue by region?"
- Customer data: "How many customers signed up last month?"
- Transaction data: "What's the average transaction value?"
- Product data: "Which category has the most items?"
"""