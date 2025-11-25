"""
LANGGRAPH UTILITY TOOL: Find File
Search for CSV/Excel files in workspace
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from langchain_core.tools import tool


# =============================================================================
# CONFIGURATION
# =============================================================================

MAX_SEARCH_DEPTH = 3
SUPPORTED_EXTENSIONS = ['.csv', '.xlsx', '.xls', '.xlsm']

# Input files directory - primary search location for uploaded files
INPUT_FILES_DIR = './input_files'

# Workspace root can be set via environment variable
# Falls back to current directory if not set
WORKSPACE_ROOT = os.getenv('DATA_CONTRACT_WORKSPACE', os.getcwd())


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class FileMatch(BaseModel):
    """Information about a matched file"""
    file_path: str = Field(description="Full absolute path to the file")
    file_name: str = Field(description="Just the filename")
    directory: str = Field(description="Directory containing the file")
    file_size_mb: float = Field(description="File size in megabytes")
    last_modified: str = Field(description="Last modified timestamp (ISO format)")
    last_modified_epoch: float = Field(description="Last modified as epoch timestamp for sorting")


class FindFileOutput(BaseModel):
    """Output from the find_file tool"""
    status: str = Field(description="'success' or 'error'")
    message: str = Field(description="Human-readable summary")
    search_term: str = Field(description="The filename that was searched for")
    matches_found: int = Field(description="Number of matching files found")
    searched_locations: List[str] = Field(
        default_factory=list,
        description="Directories that were searched"
    )
    files: Optional[List[FileMatch]] = Field(
        default=None,
        description="List of matched files, ordered by last modified (newest first)"
    )
    error_type: Optional[str] = Field(
        default=None,
        description="Type of error if status='error'"
    )
    suggestion: Optional[str] = Field(
        default=None,
        description="Suggestion for resolving error"
    )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_file_size_mb(file_path: str) -> float:
    """Get file size in megabytes"""
    try:
        size_bytes = os.path.getsize(file_path)
        return round(size_bytes / (1024 * 1024), 2)
    except:
        return 0.0


def get_last_modified(file_path: str) -> tuple[str, float]:
    """Get last modified time as ISO string and epoch"""
    try:
        mtime = os.path.getmtime(file_path)
        dt = datetime.fromtimestamp(mtime)
        iso_string = dt.isoformat()
        return iso_string, mtime
    except:
        return "unknown", 0.0


def is_supported_file(file_path: str) -> bool:
    """Check if file has a supported extension"""
    ext = os.path.splitext(file_path)[1].lower()
    return ext in SUPPORTED_EXTENSIONS


def search_directory(
    directory: str,
    filename: str,
    max_depth: int = MAX_SEARCH_DEPTH,
    current_depth: int = 0
) -> List[str]:
    """
    Recursively search directory for matching files
    
    Parameters:
        directory: Directory to search
        filename: Exact filename to match
        max_depth: Maximum recursion depth
        current_depth: Current recursion depth
        
    Returns:
        List of full paths to matching files
    """
    matches = []
    
    # Stop if we've reached max depth
    if current_depth > max_depth:
        return matches
    
    try:
        # List directory contents
        for entry in os.listdir(directory):
            full_path = os.path.join(directory, entry)
            
            # Skip if we can't access
            if not os.access(full_path, os.R_OK):
                continue
            
            # If it's a file, check if it matches
            if os.path.isfile(full_path):
                # Exact match (case-insensitive)
                if entry.lower() == filename.lower() and is_supported_file(full_path):
                    matches.append(os.path.abspath(full_path))
            
            # If it's a directory, recurse
            elif os.path.isdir(full_path):
                # Skip hidden directories and common excludes
                if entry.startswith('.') or entry in ['__pycache__', 'node_modules', 'venv', '.git']:
                    continue
                
                # Recurse
                sub_matches = search_directory(full_path, filename, max_depth, current_depth + 1)
                matches.extend(sub_matches)
    
    except (PermissionError, OSError):
        # Skip directories we can't access
        pass
    
    return matches


def create_file_match(file_path: str) -> Dict[str, Any]:
    """Create a FileMatch dict from a file path"""
    file_name = os.path.basename(file_path)
    directory = os.path.dirname(file_path)
    file_size_mb = get_file_size_mb(file_path)
    last_modified_iso, last_modified_epoch = get_last_modified(file_path)
    
    return {
        'file_path': file_path,
        'file_name': file_name,
        'directory': directory,
        'file_size_mb': file_size_mb,
        'last_modified': last_modified_iso,
        'last_modified_epoch': last_modified_epoch
    }


# =============================================================================
# MAIN TOOL FUNCTION
# =============================================================================

@tool
def find_file(
    filename: str,
    search_workspace: bool = True
) -> dict:
    """
    Find CSV or Excel files in the workspace by filename.
    
    Searches for files by exact filename match (case-insensitive). Useful when user
    references a file by name without providing the full path.
    
    SEARCH STRATEGY:
    1. Search ./input_files directory first (primary upload location)
    2. If not found, search current directory and subdirectories (max 3 levels deep)
    3. If not found and search_workspace=True, search workspace root
    4. Never searches beyond workspace boundaries
    
    MULTIPLE MATCHES:
    - If multiple files found, returns ALL matches
    - Files are ordered by last modified date (newest first)
    - Agent should ask user to clarify which file to use
    
    IMPORTANT:
    - Only searches for .csv, .xlsx, .xls, .xlsm files
    - Uses exact filename matching (case-insensitive)
    - Maximum search depth: 3 levels
    - Skips hidden directories and common excludes (.git, __pycache__, etc.)
    
    Args:
        filename: Exact filename to search for (e.g., "contract.csv", "master_contract.xlsx")
                 Case-insensitive matching.
        search_workspace: If True and file not found in current directory, search workspace root.
                         Defaults to True. Workspace root is set via DATA_CONTRACT_WORKSPACE
                         environment variable or defaults to current directory.
    
    Returns:
        Dictionary with:
        - status: 'success' or 'error'
        - message: Human-readable summary
        - search_term: The filename searched for
        - matches_found: Number of files found
        - searched_locations: List of directories searched
        - files: List of FileMatch objects (ordered by last modified, newest first)
        - error_type: (on error) Type of error
        - suggestion: (on error) How to fix it
    """
    
    try:
        matches = []
        searched_locations = []
        
        # Normalize filename for comparison
        search_filename = filename.strip()
        
        # Validate filename
        if not search_filename:
            output = FindFileOutput(
                status='error',
                message='Filename cannot be empty',
                search_term=search_filename,
                matches_found=0,
                error_type='InvalidFilenameError',
                suggestion='Provide a valid filename to search for.'
            )
            return output.model_dump()
        
        # Check if filename has supported extension
        ext = os.path.splitext(search_filename)[1].lower()
        if ext and ext not in SUPPORTED_EXTENSIONS:
            output = FindFileOutput(
                status='error',
                message=f'Unsupported file type: {ext}. Only CSV and Excel files are supported.',
                search_term=search_filename,
                matches_found=0,
                error_type='UnsupportedFileTypeError',
                suggestion=f'Supported extensions: {", ".join(SUPPORTED_EXTENSIONS)}'
            )
            return output.model_dump()
        
        # STEP 1: Search ./input_files directory first (primary upload location)
        input_files_path = os.path.abspath(INPUT_FILES_DIR)
        if os.path.exists(input_files_path) and os.path.isdir(input_files_path):
            searched_locations.append(input_files_path)
            matches = search_directory(input_files_path, search_filename, max_depth=MAX_SEARCH_DEPTH)
        
        # STEP 2: If not found, search current directory
        if not matches:
            current_dir = os.getcwd()
            current_dir_abs = os.path.abspath(current_dir)
            
            # Only search current directory if it's different from input_files
            if current_dir_abs != input_files_path:
                searched_locations.append(current_dir_abs)
                current_matches = search_directory(current_dir_abs, search_filename, max_depth=MAX_SEARCH_DEPTH)
                matches.extend(current_matches)
        
        # STEP 3: If not found and search_workspace enabled, search workspace root
        if not matches and search_workspace:
            workspace_root = os.path.abspath(WORKSPACE_ROOT)
            
            # Only search workspace if it's different from already searched locations
            if workspace_root not in [os.path.abspath(loc) for loc in searched_locations]:
                searched_locations.append(workspace_root)
                workspace_matches = search_directory(workspace_root, search_filename, max_depth=MAX_SEARCH_DEPTH)
                matches.extend(workspace_matches)
        
        # Remove duplicates (in case file is in both searches)
        matches = list(set(matches))
        
        # Create FileMatch objects
        file_matches = [create_file_match(path) for path in matches]
        
        # Sort by last modified (newest first)
        file_matches.sort(key=lambda x: x['last_modified_epoch'], reverse=True)
        
        # Convert to Pydantic objects
        file_match_objects = [FileMatch(**fm) for fm in file_matches]
        
        # Create message
        if len(matches) == 0:
            message = f"No files found matching '{search_filename}'"
        elif len(matches) == 1:
            message = f"Found 1 file: {file_matches[0]['file_path']}"
        else:
            message = (
                f"Found {len(matches)} files matching '{search_filename}'. "
                f"Files ordered by last modified (newest first). "
                f"Ask user to clarify which file to use."
            )
        
        output = FindFileOutput(
            status='success',
            message=message,
            search_term=search_filename,
            matches_found=len(matches),
            searched_locations=searched_locations,
            files=file_match_objects
        )
        
        return output.model_dump()
        
    except Exception as e:
        output = FindFileOutput(
            status='error',
            message=f'Error during file search: {str(e)}',
            search_term=filename,
            matches_found=0,
            error_type=type(e).__name__,
            suggestion='Check that you have read permissions for the directories being searched.'
        )
        return output.model_dump()


# =============================================================================
# END OF TOOL
# =============================================================================


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

"""
HOW TO USE THIS TOOL IN LANGGRAPH:

1. Import the tool:
   from find_file_tool import find_file

2. Add to your agent's tools list:
   tools = [find_file, read_file, ...]

3. The agent can invoke it:
   result = find_file.invoke({
       "filename": "contract.csv"
   })

4. Access structured results:
   if result['status'] == 'success':
       if result['matches_found'] == 1:
           # Single match - use it
           file_path = result['files'][0]['file_path']
       elif result['matches_found'] > 1:
           # Multiple matches - ask user to clarify
           for file in result['files']:
               print(f"- {file['file_path']} (modified: {file['last_modified']})")

AGENT BEHAVIOR NOTES:
- Use this when user references a file by name without full path
- If multiple matches found, ASK USER to clarify which one to use
- Files are ordered by last modified (newest first) to help user identify
- Only searches .csv, .xlsx, .xls, .xlsm files
- Search is case-insensitive
- Never searches beyond workspace boundaries
- Prioritizes ./input_files directory (where uploaded files are stored)

INPUT FILES DIRECTORY:
The tool searches ./input_files first, which is where the UI uploads files.
Make sure this directory exists in your application root:
  mkdir input_files

SETTING WORKSPACE ROOT:
Set via environment variable before running agent:
  export DATA_CONTRACT_WORKSPACE=/path/to/your/workspace
  
Or in code:
  os.environ['DATA_CONTRACT_WORKSPACE'] = '/path/to/workspace'
"""


# =============================================================================
# FOR STANDALONE TESTING
# =============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python find_file_tool.py <filename>")
        print("\nExample: python find_file_tool.py contract.csv")
        sys.exit(1)
    
    filename = sys.argv[1]
    
    result = find_file.invoke({
        "filename": filename
    })
    
    print(f"\nStatus: {result['status']}")
    print(f"Message: {result['message']}")
    print(f"Searched: {result['searched_locations']}")
    
    if result['status'] == 'success' and result['matches_found'] > 0:
        print(f"\nFound {result['matches_found']} file(s):")
        for i, file in enumerate(result['files'], 1):
            print(f"\n{i}. {file['file_name']}")
            print(f"   Path: {file['file_path']}")
            print(f"   Directory: {file['directory']}")
            print(f"   Size: {file['file_size_mb']} MB")
            print(f"   Modified: {file['last_modified']}")
    
    elif result['status'] == 'error':
        print(f"\nError: {result['error_type']}")
        print(f"Suggestion: {result['suggestion']}")