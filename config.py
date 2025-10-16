import os
from enum import Enum
from typing import Dict

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class InstructionType(Enum):
    FAQ_ASSISTANT = "faq_assistant"
    TECHNICAL_SUPPORT = "technical_support"


class PromptTemplate(Enum):
    FAQ_ASSISTANT = """
<QUESTION>
{question}
</QUESTION>

<CONTEXT>
{context}
</CONTEXT>
""".strip()

    TECHNICAL_SUPPORT = """
<QUESTION>
{question}
</QUESTION>

<TECHNICAL_CONTEXT>
{context}
</TECHNICAL_CONTEXT>

Please provide a clear, step-by-step solution based on the technical context above.
""".strip()


class InstructionsConfig:
    INSTRUCTIONS: Dict[InstructionType, str] = {
        InstructionType.FAQ_ASSISTANT: """
You're a helpful FAQ assistant. Answer questions based on the provided CONTEXT.
Be concise, accurate, and helpful. If you don't know the answer based on the context, say so.
""".strip(),
        InstructionType.TECHNICAL_SUPPORT: """
You're a technical support assistant. Help users with technical questions.
Provide clear, step-by-step solutions when possible.
Use the CONTEXT to provide accurate technical information.
""".strip(),
    }


class ModelType(Enum):
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4O = "gpt-4o"
    GPT_3_5_TURBO = "gpt-3.5-turbo"


class ChunkingConfig(Enum):
    DEFAULT_SIZE = 2000
    DEFAULT_OVERLAP = 0.5
    DEFAULT_CONTENT_FIELD = "content"


class RepositoryConfig(Enum):
    DEFAULT_OWNER = "pydantic"
    DEFAULT_NAME = "pydantic-ai"
    DEFAULT_EXTENSIONS = {"md", "mdx"}
    GITHUB_CODELOAD_URL = "https://codeload.github.com"


class SearchType(Enum):
    TEXT = "text"
    VECTOR_MINSEARCH = "vector_minsearch"
    VECTOR_SENTENCE_TRANSFORMERS = "vector_sentence_transformers"


class SentenceTransformerModel(Enum):
    # Small models (fast, good for testing)
    ALL_MINILM_L6_V2 = "all-MiniLM-L6-v2"
    ALL_MPNET_BASE_V2 = "all-mpnet-base-v2"
    PARAPHRASE_MULTILINGUAL_MINI = "paraphrase-multilingual-MiniLM-L12-v2"

    # Tiny models (very fast, basic quality)
    ALL_MINILM_L12_V2 = "all-MiniLM-L12-v2"
    DISTILBERT_BASE_NLI_STSB_MEAN_TOKENS = "distilbert-base-nli-stsb-mean-tokens"

    # Specialized models
    MSMARCO_DISTILBERT_BASE_V4 = "msmarco-distilbert-base-v4"
    MULTI_QA_MINILM_L6_COS_V1 = "multi-qa-MiniLM-L6-cos-v1"


class GitHubConfig(Enum):
    BASE_URL = "https://codeload.github.com"
    TIMEOUT = 30
    MAX_FILE_SIZE = 10_000_000
    MAX_FILES = 1000
    MAX_TOTAL_SIZE = 50_000_000


class FileProcessingConfig(Enum):
    MAX_FILE_SIZE = 10_000_000  # 10MB per file
    MAX_CONTENT_SIZE = 5_000_000  # 5MB content
    ALLOWED_EXTENSIONS = {"md", "mdx", "txt", "rst", "adoc"}
    BLOCKED_EXTENSIONS = {"exe", "bat", "sh", "py", "js", "jar", "dll", "so"}


class AgentConfig(Enum):
    """Configuration for Pydantic AI agents"""

    # Default agent models
    DEFAULT_MODEL = "openai:gpt-4o-mini"
    GPT_4O_MODEL = "openai:gpt-4o"
    GPT_3_5_TURBO_MODEL = "openai:gpt-3.5-turbo"

    # Agent behavior settings
    DEFAULT_MAX_TOOL_CALLS = 10
    DEFAULT_TOOL_TIMEOUT = 30
    DEFAULT_MAX_SEARCH_RESULTS = 5
    DEFAULT_MAX_FILE_PREVIEW_LINES = 5

    # Agent instruction templates
    BASE_INSTRUCTIONS = """
You are an intelligent documentation assistant with access to a GitHub repository's content.

You have access to the following tools:
- search_tool: Search for relevant information in the repository documents
- read_file_tool_wrapper: Read the complete content of any file
- list_files_tool_wrapper: See what files are available in the repository
- analyze_results_tool: Analyze search results to provide insights
- get_file_info_tool_wrapper: Get detailed information about a specific file

Guidelines:
1. Always search for relevant information before answering questions
2. Use multiple search queries if needed to gather comprehensive information
3. When you find relevant files, use read_file_tool_wrapper to get complete context
4. Provide specific file references and code examples when possible
5. If information is missing or unclear, state this explicitly
6. Be concise but thorough in your responses
7. Use analyze_results_tool to provide insights about search results when helpful

When citing sources, use this format: [filename](https://github.com/{repo_owner}/{repo_name}/blob/main/{filename})
""".strip()


class AgentInstructionsConfig:
    """Agent-specific instruction configurations"""

    INSTRUCTIONS: Dict[InstructionType, str] = {
        InstructionType.FAQ_ASSISTANT: AgentConfig.BASE_INSTRUCTIONS.value
        + """

As an FAQ assistant, focus on:
- Providing clear, direct answers
- Addressing common questions comprehensively
- Including practical examples
- Being helpful and user-friendly
- Citing specific documentation sections when relevant
""".strip(),
        InstructionType.TECHNICAL_SUPPORT: AgentConfig.BASE_INSTRUCTIONS.value
        + """

As a technical support assistant, focus on:
- Providing step-by-step solutions
- Explaining technical concepts clearly
- Including relevant code examples
- Troubleshooting common issues
- Citing specific documentation sections
- Breaking down complex procedures into manageable steps
""".strip(),
    }

    # Tool-specific instructions
    TOOL_INSTRUCTIONS = {
        "search_tool": "Use this to find relevant information. Make multiple searches with different keywords if needed.",
        "read_file_tool_wrapper": "Use this to get complete file content when you need full context or specific details.",
        "list_files_tool_wrapper": "Use this to see what files are available when you need to explore the repository structure.",
        "analyze_results_tool": "Use this to provide insights about search results, especially when dealing with multiple sources.",
        "get_file_info_tool_wrapper": "Use this to get metadata about files before deciding whether to read their full content.",
    }


# OpenAI configuration
API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=API_KEY)
