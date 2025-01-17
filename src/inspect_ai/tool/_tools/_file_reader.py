from inspect_ai.tool import Tool, ToolError, tool
from inspect_ai.util import sandbox


@tool
def read_file_chunk():
    async def execute(file: str, start_line: int = 1, max_lines: int = 50) -> str:
        """Read a chunk of lines from a file.

        Args:
            file (str): Path to the file to read
            start_line (int): Line number to start reading from (1-indexed)
            max_lines (int): Maximum number of lines to read (default: 50, max: 100)

        Returns:
            str: The requested lines from the file

        Raises:
            ToolError: If the file cannot be read or if invalid line numbers are provided
        """
        if start_line < 1:
            raise ToolError("start_line must be >= 1")
        
        if max_lines < 1:
            raise ToolError("max_lines must be >= 1")
            
        if max_lines > 100:
            raise ToolError("max_lines cannot exceed 100")

        try:
            # Read the file
            content = await sandbox().read_file(file)
            
            # Split into lines
            lines = content.splitlines()
            
            # Calculate end line
            end_line = min(start_line + max_lines - 1, len(lines))
            
            # Get the requested chunk
            chunk = lines[start_line - 1:end_line]
            
            # Add line numbers and join
            numbered_lines = [f"{i+start_line}: {line}" for i, line in enumerate(chunk)]
            
            # Add summary info
            total_lines = len(lines)
            summary = f"File has {total_lines} total lines. Showing lines {start_line} to {end_line}.\n\n"
            
            return summary + "\n".join(numbered_lines)

        except FileNotFoundError:
            raise ToolError(f"File '{file}' not found")
        except Exception as e:
            raise ToolError(f"Error reading file: {str(e)}")

    return execute


