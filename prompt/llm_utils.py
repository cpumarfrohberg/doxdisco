# LLM interaction utilities
import re
from typing import Any

from openai import OpenAI

from config import InstructionType, ModelType
from prompt.models import RAGAnswer
from prompt.prompt_builder import build_prompt
from prompt.search_utils import RAGError, search_documents

CONFIDENCE = 0.5


def query_with_context(
    question: str,
    index: Any = None,
    openai_client: OpenAI | None = None,
    instruction_type: str = InstructionType.PYDANTIC_AI_EXPERT,
) -> RAGAnswer:
    """
    Answer a question using RAG with structured Pydantic output.

    Args:
        question: The question to answer
        index: The search index to retrieve relevant documents
        openai_client: OpenAI client for generating responses
        instruction_type: Type of instruction/prompt template to use

    Returns:
        Structured RAGAnswer with validated data

    Raises:
        RAGError: If any error occurs during processing
    """
    if openai_client is None:
        raise RAGError("OpenAI client is required")

    try:
        search_results: list[dict[str, Any]] = search_documents(question, index)

        user_prompt: str = build_prompt(question, search_results, instruction_type)
        messages = [{"role": "user", "content": user_prompt}]

        try:
            response = openai_client.responses.parse(
                model=ModelType.GPT_4O_MINI, input=messages, text_format=RAGAnswer
            )

            rag_answer = response.output_parsed
            if not rag_answer.answer.strip():
                raise RAGError("Empty answer received from LLM")

            return rag_answer

        except Exception as e:
            print("⚠️  Structured output failed, trying fallback...")
            try:
                fallback_response = _generate_fallback_response(
                    question, search_results, openai_client
                )
                return fallback_response
            except Exception:
                raise RAGError(
                    f"Both structured and fallback responses failed: {str(e)}"
                ) from e

    except RAGError:
        raise
    except Exception as e:
        raise RAGError(f"Unexpected error in RAG processing: {str(e)}") from e


def _generate_fallback_response(
    question: str, search_results: list[dict[str, Any]], openai_client: OpenAI
) -> RAGAnswer:
    """
    Generate a fallback response when structured output parsing fails.

    Creates a simple prompt and manually parses the response to extract
    answer, confidence, and sources. Used as a backup when the primary
    structured output method fails.

    Args:
        question: The user's question to be answered
        search_results: List of relevant document dictionaries from search
        openai_client: OpenAI client for API calls

    Returns:
        RAGAnswer object with parsed response data

    Raises:
        RAGError: If fallback response generation fails
    """
    try:
        # Create a simple prompt for fallback
        context_text = "\n\n".join(
            [
                f"Source: {result.get('filename', 'unknown')}\n{result.get('content', '')[:500]}"
                for result in search_results[:3]
            ]
        )

        fallback_prompt = f"""
Answer this question based on the provided context:

Question: {question}

Context:
{context_text}

Provide a clear answer and rate your confidence from 0.0 to 1.0.
"""

        messages = [{"role": "user", "content": fallback_prompt}]

        response = openai_client.responses.create(
            model=ModelType.GPT_4O_MINI, input=messages
        )

        answer_text = response.output_text.strip()

        confidence = CONFIDENCE
        if "confidence" in answer_text.lower():
            try:
                confidence_match = re.search(
                    r"confidence[:\s]+([0-9.]+)", answer_text.lower()
                )
                if confidence_match:
                    confidence = float(confidence_match.group(1))
            except (ValueError, AttributeError):
                pass

        sources_used = [
            result.get("filename", "unknown") for result in search_results[:3]
        ]

        return RAGAnswer(
            answer=answer_text,
            confidence=confidence,
            sources_used=sources_used,
            reasoning="Generated using fallback method due to structured output failure",
        )

    except Exception as e:
        raise RAGError(f"Fallback response generation failed: {str(e)}") from e
