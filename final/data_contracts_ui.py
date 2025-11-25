"""
DATA CONTRACT MANAGEMENT UI
Gradio interface for the LangGraph Data Contract Agent
"""

import gradio as gr
import os
import json
import queue
import threading
import shutil
import tempfile
from datetime import datetime
from typing import List, Dict, Any, Generator, Tuple, Optional
import pandas as pd

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated

# Import our tools
from consolidate_contract_tool import consolidate_contract
from compare_contracts_tool import compare_contracts
from merge_and_highlight_tool import merge_and_highlight
from read_file_tool import read_file
from find_file_tool import find_file
from display_csv_tool import display_csv


# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_FILES_DIR = "./input_files"
MAX_UPLOAD_FILES = 2
PREVIEW_ROWS = 3

# Ensure input_files directory exists
os.makedirs(INPUT_FILES_DIR, exist_ok=True)


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """# ROLE AND EXPERTISE

You are an expert Data Engineer specializing in data contract management and semantic analysis of business rules. You help users manage data contracts through intelligent analysis, comparison, and consolidation of business rules using LLM-powered tools.

# YOUR CAPABILITIES

You have access to these specialized tools:

## Contract Processing Tools:
1. **consolidate_contract**: Creates a master/golden contract from consumer data with duplicate detection
2. **compare_contracts**: Compares proposed vs master contract to identify new rules (delta)
3. **merge_and_highlight**: Merges master + delta with RED/YELLOW highlighting for review

## Utility Tools:
4. **find_file**: Search for files by name in the workspace
5. **read_file**: Preview CSV/Excel file contents
6. **display_csv**: Display file in the UI artifact panel for human review

# WORKFLOW UNDERSTANDING

These tools represent a typical 3-stage workflow:
- **Stage 1**: consolidate_contract → Creates clean master from messy input
- **Stage 2**: compare_contracts → Identifies delta between proposed and master  
- **Stage 3**: merge_and_highlight → Merges and highlights for human approval

**IMPORTANT**: Do NOT automatically complete the full 3-stage workflow unless explicitly requested.

# FILE MANAGEMENT

**File Roles:**
- When user uploads files, they may specify roles: "master" or "proposed"
- If role is "auto" or unspecified, ask the user to clarify
- Files are saved to ./input_files/ directory

**File Types:**
- **PRIMARY OUTPUT** files: Use as inputs to subsequent tools
- **AUDIT ONLY** files: For human review only, do NOT pass to processing tools

**Finding Files:**
- If user references a filename without path, use find_file tool
- Always use full paths from tool responses

# DISPLAYING RESULTS

**CRITICAL**: You MUST use display_csv to show results to the user after ANY tool creates an output file.
- After consolidate_contract: ALWAYS call display_csv on the master_contract file
- After compare_contracts: ALWAYS call display_csv on the new_rules_delta file (if new rules found)
- After merge_and_highlight: ALWAYS call display_csv on the merged contract file

Do NOT wait for user to ask - automatically display results so they can review immediately.

Example workflow:
1. User: "Consolidate this file"
2. You: Call consolidate_contract → get master_contract path
3. You: Call display_csv with that path → shows in UI artifact panel
4. You: Respond with summary

# INTERACTION GUIDELINES

**When Files Are Uploaded:**
- Acknowledge uploaded files and their assigned roles
- If roles are unclear, ask user to clarify which is master vs proposed
- Use read_file to preview if user wants to see contents

**Be Concise:**
- Summarize tool results in 2-3 sentences
- Use emoji indicators: ✅ success, 📊 stats, 🚨 conflicts, 📄 files
- Highlight critical information (conflicts, file paths)

**When to Ask Questions:**
- If file roles are ambiguous
- If user's intent is unclear
- Before running potentially long operations

# IMPORTANT CONSTRAINTS

- Maximum 5 tool calls per single user prompt
- STOP after merge_and_highlight - output requires human review
- Do NOT pass AUDIT files to processing tools
- ALWAYS use display_csv for files that need human review
- Alert immediately when conflicts are detected

# OUTPUT FORMAT

After tool execution:
```
✅ [Action completed]
📄 File: [path]
📊 [Key statistics]
🚨 [Conflicts/warnings if any]
```
"""


# =============================================================================
# AGENT STATE
# =============================================================================

class AgentState(TypedDict):
    """State for the data contract management agent"""
    messages: Annotated[List, add_messages]


# =============================================================================
# LANGGRAPH AGENT SETUP
# =============================================================================

# All available tools
ALL_TOOLS = [
    consolidate_contract,
    compare_contracts,
    merge_and_highlight,
    read_file,
    find_file,
    display_csv
]


def create_agent():
    """Create the LangGraph agent with all tools"""
    
    def agent_node(state: AgentState):
        """Main agent reasoning node"""
        messages = state["messages"]
        
        model = ChatBedrock(
            model_id="global.anthropic.claude-haiku-4-5-20251001-v1:0",
            region_name="us-east-1",
            model_kwargs={
                "temperature": 0,
                "max_tokens": 4096
            }
        ).bind_tools(ALL_TOOLS)
        
        # Add system message if not present
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)
        
        response = model.invoke(messages)
        return {"messages": [response]}
    
    def should_continue(state: AgentState):
        """Determine if agent should continue or end"""
        messages = state["messages"]
        last_message = messages[-1]
        
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        return END
    
    # Build the graph
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(ALL_TOOLS))
    workflow.set_entry_point("agent")
    
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", END: END}
    )
    workflow.add_edge("tools", "agent")
    
    # Compile with memory
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


# =============================================================================
# FILE UPLOAD HANDLING
# =============================================================================

def save_uploaded_file(file_obj, original_name: str) -> str:
    """Save uploaded file to input_files directory"""
    if file_obj is None:
        return None
    
    # Get the file path from Gradio's file object
    if hasattr(file_obj, 'name'):
        source_path = file_obj.name
    else:
        source_path = file_obj
    
    # Create destination path
    dest_path = os.path.join(INPUT_FILES_DIR, original_name)
    
    # Copy file
    shutil.copy2(source_path, dest_path)
    
    return os.path.abspath(dest_path)


def get_file_preview(file_path: str, max_rows: int = PREVIEW_ROWS) -> Dict[str, Any]:
    """Get a preview of the uploaded file"""
    try:
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.csv':
            df = pd.read_csv(file_path, nrows=max_rows)
            total_rows = sum(1 for _ in open(file_path)) - 1
        elif ext in ['.xlsx', '.xls', '.xlsm']:
            df = pd.read_excel(file_path, nrows=max_rows)
            df_full = pd.read_excel(file_path)
            total_rows = len(df_full)
        else:
            return {"error": f"Unsupported file type: {ext}"}
        
        file_size = os.path.getsize(file_path)
        
        return {
            "preview_html": df.to_html(index=False, classes="preview-table"),
            "total_rows": total_rows,
            "total_columns": len(df.columns),
            "columns": df.columns.tolist(),
            "file_size_kb": round(file_size / 1024, 1)
        }
    except Exception as e:
        return {"error": str(e)}


def format_file_info_for_prompt(files_info: List[Dict]) -> str:
    """Format file upload information for the agent prompt"""
    if not files_info:
        return ""
    
    lines = ["\n[Uploaded Files]"]
    for info in files_info:
        role_str = f", role: {info['role']}" if info['role'] != 'auto' else ", role: auto-detect"
        lines.append(f"- {info['filename']} (saved to: {info['path']}{role_str})")
    
    return "\n".join(lines)


# =============================================================================
# ARTIFACT HANDLING
# =============================================================================

def detect_artifact(tool_result: Dict) -> Optional[Dict]:
    """Detect if a tool result contains a displayable artifact"""
    if isinstance(tool_result, dict) and tool_result.get("display_type"):
        return {
            "type": tool_result["display_type"],
            "title": tool_result.get("title", "Untitled"),
            "data": tool_result,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "file_path": tool_result.get("file_path")
        }
    return None


def render_artifact_html(artifact: Dict) -> str:
    """Render artifact to HTML for display"""
    if not artifact:
        return "<p>No artifact selected</p>"
    
    data = artifact.get("data", {})
    
    # Check if we have HTML content (from display_csv)
    if "html_content" in data:
        return data["html_content"]
    
    # Fallback to raw data display
    return f"<pre>{json.dumps(data, indent=2)}</pre>"


# =============================================================================
# STREAMING RESPONSE HANDLER
# =============================================================================

def stream_agent_response(
    user_message: str,
    agent,
    thread_id: str,
    current_artifacts: List[Dict]
) -> Generator[Tuple[str, List[Dict], List[Dict]], None, None]:
    """
    Stream agent response with tool call visibility.
    
    Yields: (accumulated_response, artifacts, tool_calls_info)
    """
    config = {"configurable": {"thread_id": thread_id}}
    
    accumulated = ""
    artifacts = list(current_artifacts)
    tool_calls_info = []
    
    try:
        # Stream events from the agent
        for event in agent.stream(
            {"messages": [HumanMessage(content=user_message)]},
            config=config,
            stream_mode="values"
        ):
            messages = event.get("messages", [])
            if not messages:
                continue
            
            last_msg = messages[-1]
            
            # Handle AI message with potential tool calls
            if isinstance(last_msg, AIMessage):
                # Check for tool calls
                if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                    for tool_call in last_msg.tool_calls:
                        tool_info = {
                            "name": tool_call["name"],
                            "args": tool_call["args"],
                            "id": tool_call["id"],
                            "start_time": datetime.now(),
                            "status": "running"
                        }
                        tool_calls_info.append(tool_info)
                        
                        # Add tool call to response
                        accumulated += format_tool_call_start(tool_call["name"], tool_call["args"])
                        yield accumulated, artifacts, tool_calls_info
                
                # Handle text content
                elif last_msg.content and isinstance(last_msg.content, str):
                    accumulated += last_msg.content
                    yield accumulated, artifacts, tool_calls_info
            
            # Handle tool results
            elif isinstance(last_msg, ToolMessage):
                # Parse tool result
                try:
                    if isinstance(last_msg.content, str):
                        tool_result = json.loads(last_msg.content)
                    else:
                        tool_result = last_msg.content
                except:
                    tool_result = {"raw": last_msg.content}
                
                # Update tool call info
                for tc in tool_calls_info:
                    if tc["id"] == last_msg.tool_call_id:
                        tc["status"] = "complete"
                        tc["result"] = tool_result
                        tc["end_time"] = datetime.now()
                        tc["duration_ms"] = (tc["end_time"] - tc["start_time"]).total_seconds() * 1000
                
                # Check for artifact
                artifact = detect_artifact(tool_result)
                if artifact:
                    artifacts.append(artifact)
                
                # Add result to response
                accumulated += format_tool_result(tool_result, artifact is not None)
                yield accumulated, artifacts, tool_calls_info
    
    except Exception as e:
        accumulated += f"\n\n❌ **Error:** {str(e)}"
        yield accumulated, artifacts, tool_calls_info


def format_tool_call_start(name: str, args: Dict) -> str:
    """Format tool call for display in chat"""
    output = f"\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    output += f"🔧 **Tool:** `{name}`\n\n"
    output += "<details><summary>📥 View Parameters</summary>\n\n"
    output += "```json\n"
    output += json.dumps(args, indent=2)
    output += "\n```\n</details>\n"
    return output


def format_tool_result(result: Dict, has_artifact: bool = False) -> str:
    """Format tool result for display in chat"""
    output = ""
    
    # Show status summary
    status = result.get("status", "unknown")
    message = result.get("message", "")
    
    if status == "success":
        output += f"\n✅ {message}\n"
    elif status == "error":
        output += f"\n❌ {message}\n"
        if result.get("suggestion"):
            output += f"💡 {result['suggestion']}\n"
    else:
        output += f"\n📋 {message}\n"
    
    # Note if artifact was created
    if has_artifact:
        output += "\n📊 *Result displayed in artifact panel* →\n"
    
    # Collapsible raw output
    output += "\n<details><summary>📤 View Raw Output</summary>\n\n"
    output += "```json\n"
    output += json.dumps(result, indent=2, default=str)[:2000]  # Truncate long outputs
    if len(json.dumps(result, default=str)) > 2000:
        output += "\n... (truncated)"
    output += "\n```\n</details>\n"
    output += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    
    return output


# =============================================================================
# GRADIO UI
# =============================================================================

def create_ui():
    """Create the Gradio interface"""
    
    # Create agent
    agent = create_agent()
    
    # Generate unique thread ID for this session
    thread_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    with gr.Blocks(
        title="Data Contract Management",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
        ),
        css="""
        .container { max-width: 1400px; margin: auto; }
        .chatbot { height: 550px !important; }
        .preview-table { font-size: 11px; }
        .file-info { 
            background: #f8fafc; 
            border: 1px solid #e2e8f0; 
            border-radius: 8px; 
            padding: 12px;
            margin: 8px 0;
        }
        .file-header { 
            font-weight: 600; 
            color: #1e293b;
            margin-bottom: 8px;
        }
        .file-meta { 
            font-size: 12px; 
            color: #64748b;
        }
        .artifact-container {
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            overflow: hidden;
        }
        .artifact-header {
            background: #f1f5f9;
            padding: 10px 15px;
            border-bottom: 1px solid #e2e8f0;
            font-weight: 600;
        }
        .artifact-content {
            max-height: 500px;
            overflow-y: auto;
        }
        """
    ) as app:
        
        # Header
        gr.Markdown(
            """
            # 📋 Data Contract Management Assistant
            *Consolidate, compare, and manage data contracts with semantic analysis*
            """
        )
        
        with gr.Row():
            # ===== LEFT COLUMN: Chat Interface =====
            with gr.Column(scale=3):
                
                # Chat display
                chatbot = gr.Chatbot(
                    height=550,
                    show_label=False,
                    elem_classes="chatbot",
                    render_markdown=True,
                    type="messages"
                )
                
                # File upload section - collapsible
                with gr.Accordion("📎 Upload Files", open=False):
                    with gr.Row():
                        file_upload_1 = gr.File(
                            label="File 1",
                            file_types=[".csv", ".xlsx", ".xls"],
                            scale=2
                        )
                        role_select_1 = gr.Dropdown(
                            choices=["auto", "master", "proposed"],
                            value="auto",
                            label="Role",
                            scale=1
                        )
                    
                    with gr.Row():
                        file_upload_2 = gr.File(
                            label="File 2 (optional)",
                            file_types=[".csv", ".xlsx", ".xls"],
                            scale=2
                        )
                        role_select_2 = gr.Dropdown(
                            choices=["auto", "master", "proposed"],
                            value="auto",
                            label="Role",
                            scale=1
                        )
                    
                    # File preview area
                    file_preview = gr.HTML(
                        value="<p style='color: #64748b; font-size: 13px;'>Upload files to see preview...</p>",
                        label="File Preview"
                    )
                
                # Message input
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Ask me to consolidate, compare, or analyze contracts...",
                        show_label=False,
                        scale=9,
                        container=False
                    )
                    submit_btn = gr.Button("Send", scale=1, variant="primary")
                
                # Example prompts
                gr.Examples(
                    examples=[
                        "Consolidate the business rules in the uploaded file",
                        "Compare the master contract against the proposed contract",
                        "Show me a preview of the uploaded file",
                        "Find and display master_contract.csv",
                    ],
                    inputs=msg_input
                )
            
            # ===== RIGHT COLUMN: Artifact Display =====
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Data Display")
                
                # Artifact selector
                artifact_selector = gr.Dropdown(
                    label="Select Artifact",
                    choices=[],
                    value=None,
                    interactive=True,
                    visible=False
                )
                
                # Artifact display area
                artifact_display = gr.HTML(
                    value="<div style='padding: 40px; text-align: center; color: #64748b;'><p>📄 Artifacts will appear here</p><p style='font-size: 13px;'>Process a contract to see results</p></div>",
                    elem_classes="artifact-container"
                )
                
                # Download button
                download_btn = gr.Button(
                    "⬇️ Download Active Artifact",
                    visible=False
                )
                download_file = gr.File(
                    visible=False,
                    label="Download"
                )
        
        # ===== STATE MANAGEMENT =====
        artifacts_state = gr.State([])
        uploaded_files_state = gr.State([])
        active_artifact_index = gr.State(None)
        
        # ===== EVENT HANDLERS =====
        
        def handle_file_upload(file1, role1, file2, role2, current_files):
            """Handle file uploads and generate preview"""
            files_info = []
            preview_html_parts = []
            
            for file_obj, role, idx in [(file1, role1, 1), (file2, role2, 2)]:
                if file_obj is not None:
                    # Get original filename
                    if hasattr(file_obj, 'name'):
                        original_name = os.path.basename(file_obj.name)
                    else:
                        original_name = f"uploaded_file_{idx}.csv"
                    
                    # Save file
                    saved_path = save_uploaded_file(file_obj, original_name)
                    
                    # Get preview
                    preview_data = get_file_preview(saved_path)
                    
                    files_info.append({
                        "filename": original_name,
                        "path": saved_path,
                        "role": role,
                        "preview": preview_data
                    })
                    
                    # Build preview HTML
                    if "error" not in preview_data:
                        preview_html_parts.append(f"""
                        <div class="file-info">
                            <div class="file-header">📄 {original_name}</div>
                            <div class="file-meta">
                                {preview_data['total_rows']} rows × {preview_data['total_columns']} columns | 
                                {preview_data['file_size_kb']} KB | 
                                Role: <strong>{role}</strong>
                            </div>
                            <details>
                                <summary style="cursor: pointer; margin-top: 8px;">Preview ({PREVIEW_ROWS} rows)</summary>
                                <div style="overflow-x: auto; margin-top: 8px;">
                                    {preview_data['preview_html']}
                                </div>
                            </details>
                        </div>
                        """)
                    else:
                        preview_html_parts.append(f"""
                        <div class="file-info" style="border-color: #fca5a5;">
                            <div class="file-header">❌ {original_name}</div>
                            <div style="color: #dc2626;">Error: {preview_data['error']}</div>
                        </div>
                        """)
            
            if preview_html_parts:
                preview_html = "".join(preview_html_parts)
            else:
                preview_html = "<p style='color: #64748b; font-size: 13px;'>Upload files to see preview...</p>"
            
            return files_info, preview_html
        
        def user_submit(message, history, uploaded_files):
            """Handle user message submission"""
            if not message.strip() and not uploaded_files:
                return "", history, [], None, "auto", None, "auto", "<p style='color: #64748b; font-size: 13px;'>Upload files to see preview...</p>"
            
            # Build full message with file info
            full_message = message.strip()
            
            # Only append file info if files were just uploaded this turn
            if uploaded_files:
                file_info_str = format_file_info_for_prompt(uploaded_files)
                full_message = full_message + file_info_str
            
            # Add to history
            new_history = history + [{"role": "user", "content": full_message}]
            
            # Clear uploaded files state and UI components
            return (
                "",           # Clear message input
                new_history,  # Updated history
                [],           # Clear uploaded files state
                None,         # Clear file upload 1
                "auto",       # Reset role 1
                None,         # Clear file upload 2
                "auto",       # Reset role 2
                "<p style='color: #64748b; font-size: 13px;'>Upload files to see preview...</p>"  # Reset preview
            )
        
        def bot_response(history, artifacts_state, active_index):
            """Stream bot response"""
            if not history or history[-1]["role"] != "user":
                yield history, artifacts_state, active_index, gr.update(), gr.update(), gr.update(), gr.update()
                return
            
            user_message = history[-1]["content"]
            
            # Add placeholder for assistant
            history.append({"role": "assistant", "content": ""})
            
            # Stream response
            for response_text, new_artifacts, tool_info in stream_agent_response(
                user_message,
                agent,
                thread_id,
                artifacts_state
            ):
                history[-1]["content"] = response_text
                
                # Check for new artifacts
                if len(new_artifacts) > len(artifacts_state):
                    artifacts_state = new_artifacts
                    latest_idx = len(new_artifacts) - 1
                    latest_artifact = new_artifacts[latest_idx]
                    
                    # Update artifact display
                    artifact_html = render_artifact_html(latest_artifact)
                    
                    # Update dropdown choices
                    choices = [(f"{a['timestamp']} - {a['title']}", i) for i, a in enumerate(new_artifacts)]
                    
                    yield (
                        history,
                        artifacts_state,
                        latest_idx,
                        gr.update(value=artifact_html),
                        gr.update(choices=choices, value=latest_idx, visible=True),
                        gr.update(visible=True),  # Show download button
                        gr.update()
                    )
                else:
                    yield history, artifacts_state, active_index, gr.update(), gr.update(), gr.update(), gr.update()
        
        def select_artifact(artifacts, selection):
            """Handle artifact selection from dropdown"""
            if selection is None or not artifacts:
                return gr.update(), None, gr.update()
            
            try:
                idx = int(selection)
                artifact = artifacts[idx]
                artifact_html = render_artifact_html(artifact)
                return gr.update(value=artifact_html), idx, gr.update(visible=True)
            except:
                return gr.update(), None, gr.update()
        
        def download_artifact(artifacts, active_idx):
            """Prepare artifact for download"""
            if active_idx is None or not artifacts or active_idx >= len(artifacts):
                return gr.update(value=None, visible=False)
            
            artifact = artifacts[active_idx]
            file_path = artifact.get("file_path") or artifact.get("data", {}).get("file_path")
            
            if file_path and os.path.exists(file_path):
                return gr.update(value=file_path, visible=True)
            
            # If no file path, create temp file from csv_data
            csv_data = artifact.get("data", {}).get("csv_data")
            if csv_data:
                title = artifact.get("title", "download").replace(" ", "_")
                temp_path = os.path.join(tempfile.gettempdir(), f"{title}.csv")
                with open(temp_path, "w") as f:
                    f.write(csv_data)
                return gr.update(value=temp_path, visible=True)
            
            return gr.update(value=None, visible=False)
        
        # ===== WIRE UP EVENTS =====
        
        # File upload handling
        for file_input, role_input in [(file_upload_1, role_select_1), (file_upload_2, role_select_2)]:
            file_input.change(
                handle_file_upload,
                inputs=[file_upload_1, role_select_1, file_upload_2, role_select_2, uploaded_files_state],
                outputs=[uploaded_files_state, file_preview]
            )
            role_input.change(
                handle_file_upload,
                inputs=[file_upload_1, role_select_1, file_upload_2, role_select_2, uploaded_files_state],
                outputs=[uploaded_files_state, file_preview]
            )
        
        # Message submission
        msg_input.submit(
            user_submit,
            inputs=[msg_input, chatbot, uploaded_files_state],
            outputs=[msg_input, chatbot, uploaded_files_state, file_upload_1, role_select_1, file_upload_2, role_select_2, file_preview],
            queue=False
        ).then(
            bot_response,
            inputs=[chatbot, artifacts_state, active_artifact_index],
            outputs=[chatbot, artifacts_state, active_artifact_index, artifact_display, artifact_selector, download_btn, download_file]
        )
        
        submit_btn.click(
            user_submit,
            inputs=[msg_input, chatbot, uploaded_files_state],
            outputs=[msg_input, chatbot, uploaded_files_state, file_upload_1, role_select_1, file_upload_2, role_select_2, file_preview],
            queue=False
        ).then(
            bot_response,
            inputs=[chatbot, artifacts_state, active_artifact_index],
            outputs=[chatbot, artifacts_state, active_artifact_index, artifact_display, artifact_selector, download_btn, download_file]
        )
        
        # Artifact selection
        artifact_selector.change(
            select_artifact,
            inputs=[artifacts_state, artifact_selector],
            outputs=[artifact_display, active_artifact_index, download_btn]
        )
        
        # Download
        download_btn.click(
            download_artifact,
            inputs=[artifacts_state, active_artifact_index],
            outputs=[download_file]
        )
    
    return app


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Check AWS credentials
    if not os.getenv("AWS_ACCESS_KEY_ID") and not os.getenv("AWS_PROFILE"):
        print("\n⚠️  Warning: AWS credentials not detected")
        print("Set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY or AWS_PROFILE\n")
    
    # Create and launch
    app = create_ui()
    app.queue()
    
    print("\n🚀 Starting Data Contract Management UI...")
    print(f"📁 Input files directory: {os.path.abspath(INPUT_FILES_DIR)}")
    
    try:
        app.launch(server_port=7860, share=False)
    except OSError:
        app.launch(server_port=7861, share=False)