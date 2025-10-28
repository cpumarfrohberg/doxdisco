# Document search utilities
from typing import Any

NUM_RESULTS = 5


class RAGError(Exception):
    """Base exception for RAG-related errors"""

    pass


def search_documents(question: str, index: Any | None = None) -> list[dict[str, Any]]:
    """
    Search for relevant documents using the provided search index.

    Supports both text-based search (Minsearch) and vector-based search (SentenceTransformers).
    Automatically detects the index type and uses the appropriate search method.

    Args:
        question: The search query/question to find relevant documents for
        index: The search index to query (Minsearch Index or VectorIndex)

    Returns:
        List of relevant document dictionaries with content, filename, and metadata

    Raises:
        RAGError: If question is empty, index is None, or search fails
    """
    try:
        # Basic validation
        if not question or not question.strip():
            raise RAGError("Question cannot be empty")

        if index is None:
            raise RAGError("Search index is required")

        # Define boost strategy for functions and tools
        boost_dict = {
            # Core programming concepts (highest priority)
            "def ": 2.5,  # Function definitions
            "function": 2.0,  # Function mentions
            "tool": 2.0,  # Tool mentions
            "tools": 2.0,  # Tools plural
            # Implementation details (medium priority)
            "return": 1.4,  # Return statements
            "yield": 1.4,  # Yield statements
            "async": 1.3,  # Async functions
            "await": 1.3,  # Await statements
            # Documentation (lower priority but still useful)
            "example": 1.2,  # Examples
            "usage": 1.2,  # Usage patterns
            "note": 1.1,  # Notes
        }

        # Check if it's a vector index by checking the class name
        if hasattr(index, "__class__") and "VectorIndex" in str(index.__class__):
            # Vector search (query, num_results)
            results = index.search(question, num_results=NUM_RESULTS)
        else:
            # Text search (minsearch) with boosting
            results = index.search(
                question,
                boost_dict=boost_dict,
                filter_dict={},
                num_results=NUM_RESULTS,
            )

        if not results:
            raise RAGError("No documents found for the given question")

        return results

    except RAGError:
        raise
    except Exception as e:
        raise RAGError(f"Search failed: {str(e)}") from e
