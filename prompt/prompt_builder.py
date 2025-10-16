# Prompt construction utilities
import json
from typing import Any, Dict, List

from config import InstructionType

from .models import SearchResult


def build_prompt(
    question: str,
    search_results: List[Dict[str, Any]],
    instruction_type: str = InstructionType.PYDANTIC_AI_EXPERT.value,
) -> str:
    """
    Build a structured prompt for the LLM using search results.

    Converts raw search results into a structured format and creates a prompt
    that instructs the LLM to respond in the RAGAnswer format.

    Args:
        question: The user's question to be answered
        search_results: List of relevant document dictionaries from search
        instruction_type: Type of instruction/prompt template to use

    Returns:
        Formatted prompt string ready for LLM processing
    """

    # Convert search results to structured format
    structured_results = []
    for result in search_results:
        structured_results.append(
            SearchResult(
                content=result.get("content", ""),
                filename=result.get("filename", "unknown"),
                title=result.get("title"),
                similarity_score=result.get("similarity_score"),
            )
        )

    # Get the appropriate prompt template based on instruction type
    if instruction_type == InstructionType.PYDANTIC_AI_EXPERT.value:
        prompt = _build_pydantic_ai_expert_prompt(question, structured_results)
    else:
        prompt = _build_faq_assistant_prompt(question, structured_results)

    return prompt


def _build_faq_assistant_prompt(
    question: str, structured_results: List[SearchResult]
) -> str:
    """Build prompt for FAQ assistant role"""
    return f"""
Answer the following question based on the provided context documents.

Question: {question}

Context Documents:
{json.dumps([result.dict() for result in structured_results], indent=2)}

Please provide a structured response with:
1. A clear answer to the question
2. Your confidence level (0.0 to 1.0)
3. List of source filenames you used
4. Brief reasoning for your answer (optional)

Respond in the exact format specified by the RAGAnswer model.
"""


def _build_pydantic_ai_expert_prompt(
    question: str, structured_results: List[SearchResult]
) -> str:
    """Build prompt for PydanticAI documentation expert role"""
    return f"""
You are a PydanticAI documentation expert. Answer the following question based on the provided context documents from the PydanticAI codebase.

Question: {question}

Context Documents:
{json.dumps([result.dict() for result in structured_results], indent=2)}

As a PydanticAI expert, focus on:
1. Providing accurate technical information about PydanticAI features, APIs, and usage patterns
2. Explaining concepts clearly with practical examples when possible
3. Highlighting best practices and common patterns
4. Mentioning relevant function names, class names, and import statements from the codebase
5. Being specific about PydanticAI capabilities and limitations

Please provide a structured response with:
1. A comprehensive answer to the question with PydanticAI-specific details
2. Your confidence level (0.0 to 1.0)
3. List of source filenames you used
4. Brief reasoning for your answer, including any PydanticAI-specific considerations

Respond in the exact format specified by the RAGAnswer model.
"""
