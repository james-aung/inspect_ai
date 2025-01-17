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


@tool
def search_file():
    async def execute(file: str, query: str, context_lines: int = 2) -> str:
        """Search for a keyword or phrase in a file and return matching lines with context.

        Args:
            file (str): Path to the file to search
            query (str): Text to search for (case-insensitive)
            context_lines (int): Number of lines of context to show before and after each match (default: 2)

        Returns:
            str: Matching lines with their line numbers and context

        Raises:
            ToolError: If the file cannot be read or if invalid parameters are provided
        """
        if context_lines < 0:
            raise ToolError("context_lines must be >= 0")

        try:
            # Read the file
            content = await sandbox().read_file(file)
            
            # Split into lines
            lines = content.splitlines()
            
            # Find matches (case-insensitive)
            matches = []
            query = query.lower()
            
            for i, line in enumerate(lines):
                if query in line.lower():
                    # Calculate context range
                    start = max(0, i - context_lines)
                    end = min(len(lines), i + context_lines + 1)
                    
                    # Get context lines
                    context = []
                    for j in range(start, end):
                        prefix = ">>> " if j == i else "    "  # Highlight matching line
                        context.append(f"{prefix}{j+1}: {lines[j]}")
                    
                    matches.append("\n".join(context))
            
            if not matches:
                return f"No matches found for '{query}' in {file}"
            
            summary = f"Found {len(matches)} matches for '{query}' in {file}:\n\n"
            return summary + "\n\n".join(matches)

        except FileNotFoundError:
            raise ToolError(f"File '{file}' not found")
        except Exception as e:
            raise ToolError(f"Error searching file: {str(e)}")

    return execute 