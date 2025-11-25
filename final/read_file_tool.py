"""
LANGGRAPH UTILITY TOOL: Read File
Read and preview CSV/Excel files for agent inspection
"""

import pandas as pd
import os
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool


# =============================================================================
# CONFIGURATION
# =============================================================================

MAX_FILE_SIZE_MB = 100
DEFAULT_PREVIEW_ROWS = 5
MAX_FULL_ROWS = 100


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class SheetPreview(BaseModel):
    """Preview of a single Excel sheet"""
    sheet_name: str = Field(description="Name of the Excel sheet")
    row_count: int = Field(description="Total number of rows in sheet")
    column_count: int = Field(description="Number of columns in sheet")
    columns: List[str] = Field(description="List of column names")
    preview: str = Field(description="Formatted preview of first N rows")
    preview_rows: int = Field(description="Number of rows in preview")


class FileInfo(BaseModel):
    """Information about the file"""
    file_path: str = Field(description="Full absolute path to the file")
    file_name: str = Field(description="Just the filename")
    file_size_mb: float = Field(description="File size in megabytes")
    file_type: str = Field(description="File type: csv or excel")


class ReadFileOutput(BaseModel):
    """Output from the read_file tool"""
    status: str = Field(description="'success' or 'error'")
    message: str = Field(description="Human-readable summary")
    file_info: Optional[FileInfo] = Field(
        default=None,
        description="Information about the file"
    )
    # For CSV files
    row_count: Optional[int] = Field(default=None, description="Total rows (CSV only)")
    column_count: Optional[int] = Field(default=None, description="Total columns (CSV only)")
    columns: Optional[List[str]] = Field(default=None, description="Column names (CSV only)")
    preview: Optional[str] = Field(default=None, description="Formatted preview (CSV only)")
    preview_rows: Optional[int] = Field(default=None, description="Rows shown in preview (CSV only)")
    # For Excel files
    sheet_count: Optional[int] = Field(default=None, description="Number of sheets (Excel only)")
    sheets: Optional[List[SheetPreview]] = Field(default=None, description="Preview of all sheets (Excel only)")
    # Error handling
    error_type: Optional[str] = Field(default=None, description="Type of error if status='error'")
    suggestion: Optional[str] = Field(default=None, description="Suggestion for resolving error")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_file_size_mb(file_path: str) -> float:
    """Get file size in megabytes"""
    size_bytes = os.path.getsize(file_path)
    return round(size_bytes / (1024 * 1024), 2)


def format_dataframe_preview(df: pd.DataFrame, max_rows: int) -> str:
    """Format a dataframe as a readable string preview"""
    preview_df = df.head(max_rows)
    return preview_df.to_string(index=False)


def get_file_type(file_path: str) -> str:
    """Determine if file is CSV or Excel"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.csv':
        return 'csv'
    elif ext in ['.xlsx', '.xls', '.xlsm']:
        return 'excel'
    else:
        raise ValueError(f"Unsupported file type: {ext}")


# =============================================================================
# CSV READING
# =============================================================================

def read_csv_file(file_path: str, mode: str, max_rows: int) -> Dict[str, Any]:
    """
    Read a CSV file
    
    Parameters:
        file_path: Path to CSV file
        mode: 'preview', 'full', 'head', or 'tail'
        max_rows: Number of rows to show
        
    Returns:
        Dict with file contents and metadata
    """
    # Determine how many rows to read
    if mode == 'preview':
        nrows = max_rows
    elif mode == 'full':
        nrows = None  # Read all, but we'll cap later
    elif mode == 'head':
        nrows = max_rows
    elif mode == 'tail':
        nrows = None  # Need to read all to get tail
    else:
        nrows = max_rows
    
    # Read CSV
    df = pd.read_csv(file_path, nrows=nrows)
    
    # Handle tail mode
    if mode == 'tail':
        df = df.tail(max_rows)
    
    # Cap at MAX_FULL_ROWS if in full mode
    if mode == 'full' and len(df) > MAX_FULL_ROWS:
        df = df.head(MAX_FULL_ROWS)
    
    # Get total row count (need to read file again for accurate count if we limited rows)
    if mode in ['preview', 'head', 'tail']:
        # Read just to count rows (more efficient than reading all data)
        total_rows = sum(1 for _ in open(file_path)) - 1  # Subtract header
    else:
        total_rows = len(df)
    
    return {
        'row_count': total_rows,
        'column_count': len(df.columns),
        'columns': df.columns.tolist(),
        'preview': format_dataframe_preview(df, len(df)),
        'preview_rows': len(df)
    }


# =============================================================================
# EXCEL READING
# =============================================================================

def read_excel_file(file_path: str, mode: str, max_rows: int, sheet_name: Optional[str]) -> Dict[str, Any]:
    """
    Read an Excel file
    
    Parameters:
        file_path: Path to Excel file
        mode: 'preview', 'full', 'head', or 'tail'
        max_rows: Number of rows to show per sheet
        sheet_name: Specific sheet to read (None = all sheets)
        
    Returns:
        Dict with file contents and metadata
    """
    # Get all sheet names
    excel_file = pd.ExcelFile(file_path)
    all_sheets = excel_file.sheet_names
    
    # Determine which sheets to read
    if sheet_name:
        if sheet_name not in all_sheets:
            raise ValueError(f"Sheet '{sheet_name}' not found. Available sheets: {all_sheets}")
        sheets_to_read = [sheet_name]
    else:
        sheets_to_read = all_sheets
    
    # Read each sheet
    sheet_previews = []
    for sheet in sheets_to_read:
        # Determine rows to read
        if mode == 'preview':
            nrows = max_rows
        elif mode == 'full':
            nrows = None
        elif mode == 'head':
            nrows = max_rows
        elif mode == 'tail':
            nrows = None
        else:
            nrows = max_rows
        
        # Read sheet
        df = pd.read_excel(file_path, sheet_name=sheet, nrows=nrows)
        
        # Handle tail mode
        if mode == 'tail':
            df = df.tail(max_rows)
        
        # Cap at MAX_FULL_ROWS
        if mode == 'full' and len(df) > MAX_FULL_ROWS:
            df = df.head(MAX_FULL_ROWS)
        
        # Get total row count for this sheet
        if mode in ['preview', 'head', 'tail']:
            df_full = pd.read_excel(file_path, sheet_name=sheet)
            total_rows = len(df_full)
        else:
            total_rows = len(df)
        
        sheet_previews.append({
            'sheet_name': sheet,
            'row_count': total_rows,
            'column_count': len(df.columns),
            'columns': df.columns.tolist(),
            'preview': format_dataframe_preview(df, len(df)),
            'preview_rows': len(df)
        })
    
    return {
        'sheet_count': len(all_sheets),
        'sheets': sheet_previews
    }


# =============================================================================
# MAIN TOOL FUNCTION
# =============================================================================

@tool
def read_file(
    file_path: str,
    mode: str = "preview",
    max_rows: int = DEFAULT_PREVIEW_ROWS,
    sheet_name: Optional[str] = None
) -> dict:
    """
    Read and preview CSV or Excel files.
    
    Use this tool to inspect the contents of data contract files. Perfect for understanding
    file structure, column names, and sample data before processing.
    
    IMPORTANT:
    - Files larger than 100MB are rejected
    - Preview mode shows first 5 rows by default
    - Full mode is capped at 100 rows maximum
    - For Excel files, previews ALL sheets (or specify sheet_name for just one)
    
    Args:
        file_path: Full path to the CSV or Excel file to read
        mode: Read mode - 'preview' (first N rows, default), 'full' (all rows, capped at 100),
              'head' (first N rows), or 'tail' (last N rows). Defaults to 'preview'.
        max_rows: Number of rows to show in preview/head/tail modes. Defaults to 5.
        sheet_name: For Excel files, specify a sheet name to read only that sheet.
                   If None (default), previews all sheets.
    
    Returns:
        Dictionary with:
        - status: 'success' or 'error'
        - message: Human-readable summary
        - file_info: File metadata (path, size, type)
        - For CSV: row_count, column_count, columns, preview, preview_rows
        - For Excel: sheet_count, sheets (list of SheetPreview objects)
        - error_type: (on error) Type of error
        - suggestion: (on error) How to fix it
    """
    
    try:
        # Validate file exists
        if not os.path.exists(file_path):
            output = ReadFileOutput(
                status='error',
                message=f"File not found: {file_path}",
                error_type='FileNotFoundError',
                suggestion='Verify the file path is correct. Use find_file tool to search for the file.'
            )
            return output.model_dump()
        
        # Get absolute path
        abs_path = os.path.abspath(file_path)
        file_name = os.path.basename(abs_path)
        
        # Check file size
        file_size_mb = get_file_size_mb(abs_path)
        if file_size_mb > MAX_FILE_SIZE_MB:
            output = ReadFileOutput(
                status='error',
                message=f"File too large: {file_size_mb}MB (max: {MAX_FILE_SIZE_MB}MB)",
                error_type='FileTooLargeError',
                suggestion=f'File exceeds {MAX_FILE_SIZE_MB}MB limit. Consider splitting the file or processing it outside this tool.'
            )
            return output.model_dump()
        
        # Determine file type
        try:
            file_type = get_file_type(abs_path)
        except ValueError as e:
            output = ReadFileOutput(
                status='error',
                message=str(e),
                error_type='UnsupportedFileTypeError',
                suggestion='Only CSV and Excel files (.csv, .xlsx, .xls, .xlsm) are supported.'
            )
            return output.model_dump()
        
        # Create file info
        file_info = FileInfo(
            file_path=abs_path,
            file_name=file_name,
            file_size_mb=file_size_mb,
            file_type=file_type
        )
        
        # Read file based on type
        if file_type == 'csv':
            result = read_csv_file(abs_path, mode, max_rows)
            
            message = (
                f"Read CSV file: {file_name} "
                f"({result['row_count']} rows, {result['column_count']} columns). "
                f"Showing {result['preview_rows']} rows."
            )
            
            output = ReadFileOutput(
                status='success',
                message=message,
                file_info=file_info,
                row_count=result['row_count'],
                column_count=result['column_count'],
                columns=result['columns'],
                preview=result['preview'],
                preview_rows=result['preview_rows']
            )
        
        else:  # Excel
            result = read_excel_file(abs_path, mode, max_rows, sheet_name)
            
            sheet_previews = [SheetPreview(**sheet) for sheet in result['sheets']]
            
            if sheet_name:
                message = f"Read Excel file: {file_name} (sheet: {sheet_name})"
            else:
                message = f"Read Excel file: {file_name} ({result['sheet_count']} sheets)"
            
            output = ReadFileOutput(
                status='success',
                message=message,
                file_info=file_info,
                sheet_count=result['sheet_count'],
                sheets=sheet_previews
            )
        
        return output.model_dump()
        
    except pd.errors.EmptyDataError:
        output = ReadFileOutput(
            status='error',
            message='File is empty',
            error_type='EmptyFileError',
            suggestion='The file contains no data. Verify the file is not corrupted.'
        )
        return output.model_dump()
        
    except pd.errors.ParserError as e:
        output = ReadFileOutput(
            status='error',
            message=f'Failed to parse file: {str(e)}',
            error_type='ParserError',
            suggestion='File may be corrupted or have formatting issues. Check the file can be opened in Excel/text editor.'
        )
        return output.model_dump()
        
    except Exception as e:
        output = ReadFileOutput(
            status='error',
            message=f'Unexpected error reading file: {str(e)}',
            error_type=type(e).__name__,
            suggestion='Review the error message. The file may be corrupted or inaccessible.'
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
   from read_file_tool import read_file

2. Add to your agent's tools list:
   tools = [read_file, consolidate_contract, ...]

3. The agent can invoke it:
   result = read_file.invoke({
       "file_path": "/path/to/contract.csv",
       "mode": "preview"
   })

4. Access structured results:
   if result['status'] == 'success':
       print(f"File: {result['file_info']['file_name']}")
       print(f"Columns: {result['columns']}")
       print(f"Preview:\n{result['preview']}")

AGENT BEHAVIOR NOTES:
- Use this tool to inspect files before processing
- Always check status before using results
- For Excel files, all sheets are previewed by default
- File size is limited to 100MB
- Preview mode (default) shows first 5 rows

INPUT FILES DIRECTORY:
In production, uploaded files are typically stored in ./input_files.
Use find_file tool if user references a filename without full path.
"""


# =============================================================================
# FOR STANDALONE TESTING
# =============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python read_file_tool.py <file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    result = read_file.invoke({
        "file_path": file_path,
        "mode": "preview"
    })
    
    print(f"\nStatus: {result['status']}")
    print(f"Message: {result['message']}")
    
    if result['status'] == 'success':
        if result.get('columns'):  # CSV
            print(f"\nColumns: {result['columns']}")
            print(f"\nPreview:\n{result['preview']}")
        elif result.get('sheets'):  # Excel
            print(f"\nSheets: {result['sheet_count']}")
            for sheet in result['sheets']:
                print(f"\n--- {sheet['sheet_name']} ---")
                print(f"Columns: {sheet['columns']}")
                print(f"\nPreview:\n{sheet['preview']}")
    else:
        print(f"\nError: {result['error_type']}")
        print(f"Suggestion: {result['suggestion']}")