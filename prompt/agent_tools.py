# Agent tools for Pydantic AI - functions that the agent can call
from typing import Any, Dict, List


def search_documents_tool(query: str, index: Any, num_results: int = 5) -> str:
    """
    Search tool for the agent - searches documents and returns formatted results.

    Args:
        query: The search query string
        index: The search index to query (Minsearch Index or VectorIndex)
        num_results: Maximum number of results to return

    Returns:
        Formatted string with search results
    """
    try:
        # Use your existing search logic from search_utils.py
        if hasattr(index, "__class__") and "VectorIndex" in str(index.__class__):
            # Vector search (query, num_results)
            results = index.search(query, num_results=num_results)
        else:
            # Text search (minsearch)
            results = index.search(
                query,
                boost_dict={},
                filter_dict={},
                num_results=num_results,
            )

        if not results:
            return "No relevant documents found for this query."

        # Format results for the agent
        formatted_results = f"Found {len(results)} relevant documents:\n\n"
        for i, result in enumerate(results, 1):
            formatted_results += f"{i}. **{result.get('filename', 'Unknown')}**\n"
            if result.get("title"):
                formatted_results += f"   Title: {result['title']}\n"
            formatted_results += f"   Content: {result.get('content', '')[:300]}...\n"
            if result.get("similarity_score"):
                formatted_results += (
                    f"   Similarity: {result['similarity_score']:.3f}\n"
                )
            formatted_results += "\n"

        return formatted_results

    except Exception as e:
        return f"Search failed: {str(e)}"


def read_file_tool(filename: str, file_index: Dict[str, str]) -> str:
    """
    Read file tool for the agent - retrieves full file content.

    Args:
        filename: The name of the file to read
        file_index: Dictionary mapping filenames to their content

    Returns:
        The file's content or an error message
    """
    if filename in file_index:
        content = file_index[filename]
        return f"**File: {filename}**\n\n{content}"
    else:
        return f"File '{filename}' not found in the repository."


def list_files_tool(file_index: Dict[str, str], max_files: int = 20) -> str:
    """
    List files tool for the agent - shows available files in the repository.

    Args:
        file_index: Dictionary mapping filenames to their content
        max_files: Maximum number of files to list

    Returns:
        Formatted list of available files
    """
    files = list(file_index.keys())

    if not files:
        return "No files available in the repository."

    formatted_list = f"Available files ({len(files)} total):\n\n"

    # Show first max_files files
    for i, filename in enumerate(files[:max_files], 1):
        formatted_list += f"{i}. {filename}\n"

    if len(files) > max_files:
        formatted_list += f"\n... and {len(files) - max_files} more files"

    return formatted_list


def analyze_search_results_tool(results: List[Dict[str, Any]]) -> str:
    """
    Analyze search results tool for the agent - provides summary and insights.

    Args:
        results: List of search results from the index

    Returns:
        Formatted analysis of the search results
    """
    if not results:
        return "No results to analyze."

    analysis = "**Search Results Analysis**\n\n"
    analysis += f"Total results: {len(results)}\n\n"

    # Group by file type
    file_types = {}
    for result in results:
        filename = result.get("filename", "unknown")
        ext = filename.split(".")[-1] if "." in filename else "no_extension"
        file_types[ext] = file_types.get(ext, 0) + 1

    analysis += "**File types found:**\n"
    for ext, count in file_types.items():
        analysis += f"- {ext}: {count} files\n"

    analysis += "\n**Top results by relevance:**\n"
    for i, result in enumerate(results[:3], 1):
        filename = result.get("filename", "Unknown")
        similarity = result.get("similarity_score", "N/A")
        analysis += f"{i}. {filename} (similarity: {similarity})\n"

    return analysis


def get_file_info_tool(filename: str, file_index: Dict[str, str]) -> str:
    """
    Get file information tool for the agent - provides metadata about a specific file.

    Args:
        filename: The name of the file to get info about
        file_index: Dictionary mapping filenames to their content

    Returns:
        Formatted file information
    """
    if filename not in file_index:
        return f"File '{filename}' not found in the repository."

    content = file_index[filename]

    info = f"**File Information: {filename}**\n\n"
    info += f"Size: {len(content)} characters\n"
    info += f"Lines: {content.count(chr(10)) + 1}\n"

    # Basic content analysis
    if content.strip().startswith("#"):
        info += "Type: Likely a markdown file\n"
    elif filename.endswith(".md"):
        info += "Type: Markdown documentation\n"
    elif filename.endswith(".mdx"):
        info += "Type: MDX documentation\n"
    elif filename.endswith(".txt"):
        info += "Type: Text file\n"
    else:
        info += "Type: Unknown\n"

    # Show first few lines as preview
    lines = content.split("\n")[:5]
    info += "\n**Preview (first 5 lines):**\n"
    for line in lines:
        info += f"  {line}\n"

    content_lines = content.split("\n")
    if len(content_lines) > 5:
        info += f"  ... and {len(content_lines) - 5} more lines\n"

    return info
