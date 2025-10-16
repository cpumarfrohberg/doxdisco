# Agent-based RAG implementation using Pydantic AI
import asyncio
from typing import Any, Callable, Dict, List

from minsearch import Index
from pydantic_ai import Agent
from sentence_transformers import SentenceTransformer

from config import (
    InstructionType,
    SearchType,
    SentenceTransformerModel,
)
from fetch_prep_data.parser import parse_data
from fetch_prep_data.reader import read_github_data
from prompt.agent_tools import (
    analyze_search_results_tool,
    get_file_info_tool,
    list_files_tool,
    read_file_tool,
    search_documents_tool,
)
from prompt.chunking_utils import chunk_documents
from prompt.vector_search import create_vector_index


class AgentRAG:
    """Agent-based RAG implementation using Pydantic AI with tool access"""

    def __init__(
        self,
        search_type: str = SearchType.TEXT.value,
        model_name: str = SentenceTransformerModel.ALL_MINILM_L6_V2.value,
        instruction_type: str = InstructionType.FAQ_ASSISTANT.value,
        agent_model: str = "openai:gpt-4o-mini",
    ):
        self.search_type = search_type
        self.model_name = model_name
        self.instruction_type = instruction_type
        self.agent_model = agent_model
        self.documents = []
        self.chunks = []
        self.index = None
        self.file_index = {}
        self.agent = None
        self.embedder = None

        # Initialize embedder if using SentenceTransformers
        if search_type == SearchType.VECTOR_SENTENCE_TRANSFORMERS.value:
            try:
                self.embedder = SentenceTransformer(model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for vector_sentence_transformers search type. "
                    "Install it with: pip install sentence-transformers"
                )

    def load_repository(
        self,
        repo_owner: str,
        repo_name: str,
        allowed_extensions: set | None = None,
        filename_filter: Callable | None = None,
        chunk_size: int | None = None,
        chunk_step: int | None = None,
    ):
        """Load and process GitHub repository data"""

        # Step 1: Fetch GitHub data
        github_data = read_github_data(
            repo_owner=repo_owner,
            repo_name=repo_name,
            allowed_extensions=allowed_extensions,
            filename_filter=filename_filter,
        )

        # Step 2: Parse the data
        parsed_data = parse_data(github_data)

        # Step 3: Create file index for read_file tool
        for item in parsed_data:
            self.file_index[item["filename"]] = item["content"]

        # Step 4: Chunk the documents
        self.chunks = chunk_documents(parsed_data, size=chunk_size, step=chunk_step)
        print(f"📝 Created {len(self.chunks)} document chunks")

        # Step 5: Create appropriate index based on search type
        if self.search_type == SearchType.TEXT.value:
            # Standard text search with minsearch
            self.index = Index(
                text_fields=["content", "filename", "title", "description"]
            )
            self.index.fit(self.chunks)
        elif self.search_type == SearchType.VECTOR_MINSEARCH.value:
            # Minsearch with embeddings (placeholder for future implementation)
            self.index = Index(
                text_fields=["content", "filename", "title", "description"]
            )
            self.index.fit(self.chunks)
            print("⚠️  Vector minsearch not yet implemented, using text search")
        elif self.search_type == SearchType.VECTOR_SENTENCE_TRANSFORMERS.value:
            # SentenceTransformers vector search
            if self.embedder is None:
                raise ValueError("SentenceTransformer model not initialized")
            self.index = create_vector_index(self.chunks, self.embedder)
        else:
            raise ValueError(f"Unsupported search type: {self.search_type}")

        self.documents = parsed_data

        # Step 6: Create the agent with tools
        self._create_agent()

    def _create_agent(self):
        """Create the Pydantic AI agent with tools"""

        # Define agent instructions based on instruction type
        instructions = self._get_agent_instructions()

        # Create tools that have access to the loaded data
        def search_tool(query: str, num_results: int = 5) -> str:
            """Search the repository documents for relevant information."""
            return search_documents_tool(query, self.index, num_results)

        def read_file_tool_wrapper(filename: str) -> str:
            """Read the full content of a specific file."""
            return read_file_tool(filename, self.file_index)

        def list_files_tool_wrapper() -> str:
            """List all available files in the repository."""
            return list_files_tool(self.file_index)

        def analyze_results_tool(results: List[Dict[str, Any]]) -> str:
            """Analyze search results to provide insights."""
            return analyze_search_results_tool(results)

        def get_file_info_tool_wrapper(filename: str) -> str:
            """Get detailed information about a specific file."""
            return get_file_info_tool(filename, self.file_index)

        # Create the agent
        self.agent = Agent(
            name="documentation_agent",
            instructions=instructions,
            tools=[
                search_tool,
                read_file_tool_wrapper,
                list_files_tool_wrapper,
                analyze_results_tool,
                get_file_info_tool_wrapper,
            ],
            model=self.agent_model,
        )

    def _get_agent_instructions(self) -> str:
        """Get agent instructions based on instruction type"""

        base_instructions = f"""
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

When citing sources, use this format: [filename](https://github.com/{{repo_owner}}/{{repo_name}}/blob/main/{{filename}})

Repository: {self.search_type} search with {self.model_name} model
"""

        # Add specific instructions based on type
        if self.instruction_type == InstructionType.TECHNICAL_SUPPORT.value:
            base_instructions += """

As a technical support assistant, focus on:
- Providing step-by-step solutions
- Explaining technical concepts clearly
- Including relevant code examples
- Troubleshooting common issues
- Citing specific documentation sections
"""
        elif self.instruction_type == InstructionType.FAQ_ASSISTANT.value:
            base_instructions += """

As an FAQ assistant, focus on:
- Providing clear, direct answers
- Addressing common questions comprehensively
- Including practical examples
- Being helpful and user-friendly
"""

        return base_instructions.strip()

    async def query(self, question: str, show_decisions: bool = False) -> str:
        """Query the repository using the agent"""
        if not self.agent:
            raise ValueError("No repository loaded. Call load_repository() first.")

        try:
            result = await self.agent.run(user_prompt=question)

            if show_decisions:
                self._print_agent_decisions(result)

            return result.output
        except Exception as e:
            raise ValueError(f"Agent query failed: {str(e)}") from e

    def _print_agent_decisions(self, result):
        """Print detailed information about agent's decision-making process"""
        print("\n🤖 === AGENT DECISION ANALYSIS ===")

        messages = result.all_messages()
        tool_calls = 0
        tool_returns = 0

        for i, message in enumerate(messages):
            print(f"\n📝 Message {i+1}: {message.kind}")

            if message.kind == "request":
                if hasattr(message, "parts"):
                    for part in message.parts:
                        if part.part_kind == "user-prompt":
                            # Handle different attribute names
                            content = getattr(
                                part,
                                "data",
                                getattr(
                                    part, "text", getattr(part, "content", str(part))
                                ),
                            )
                            print(f"   👤 User Question: {content}")
                        elif part.part_kind == "tool-call":
                            tool_calls += 1
                            content = getattr(
                                part,
                                "data",
                                getattr(
                                    part, "text", getattr(part, "content", str(part))
                                ),
                            )
                            print(f"   🔧 Tool Call #{tool_calls}: {content}")
                        elif part.part_kind == "tool-return":
                            tool_returns += 1
                            content = getattr(
                                part,
                                "data",
                                getattr(
                                    part, "text", getattr(part, "content", str(part))
                                ),
                            )
                            print(
                                f"   📤 Tool Return #{tool_returns}: {str(content)[:200]}..."
                            )

            elif message.kind == "response":
                if hasattr(message, "text"):
                    print(f"   🤖 Agent Response: {message.text}")
                elif hasattr(message, "parts"):
                    for part in message.parts:
                        if part.part_kind == "text":
                            content = getattr(
                                part,
                                "data",
                                getattr(
                                    part, "text", getattr(part, "content", str(part))
                                ),
                            )
                            print(f"   🤖 Agent Response: {content}")
                        elif part.part_kind == "tool-call":
                            tool_calls += 1
                            content = getattr(
                                part,
                                "data",
                                getattr(
                                    part, "text", getattr(part, "content", str(part))
                                ),
                            )
                            print(f"   🔧 Tool Call #{tool_calls}: {content}")

        print("\n📊 Summary:")
        print(f"   🔧 Total Tool Calls: {tool_calls}")
        print(f"   📤 Total Tool Returns: {tool_returns}")
        print(f"   💬 Total Messages: {len(messages)}")
        print("=" * 50)

    def query_sync(self, question: str) -> str:
        """Synchronous wrapper for the async query method"""
        return asyncio.run(self.query(question))

    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about the current agent configuration"""
        if not self.agent:
            return {"status": "No agent created"}

        return {
            "status": "Agent ready",
            "search_type": self.search_type,
            "model_name": self.model_name,
            "instruction_type": self.instruction_type,
            "agent_model": self.agent_model,
            "documents_loaded": len(self.documents),
            "chunks_created": len(self.chunks),
            "files_indexed": len(self.file_index),
            "tools_available": len(self.list_available_tools()) if self.agent else 0,
        }

    def list_available_tools(self) -> List[str]:
        """List the names of available tools"""
        if not self.agent:
            return []
        # Handle different Pydantic AI versions
        if hasattr(self.agent, "tools"):
            return [tool.__name__ for tool in self.agent.tools]
        elif hasattr(self.agent, "_tools"):
            return [tool.__name__ for tool in self.agent._tools]
        else:
            return ["tools_not_accessible"]

    def reload_agent(self):
        """Reload the agent with current configuration"""
        if self.index and self.file_index:
            self._create_agent()
            print("🔄 Agent reloaded with current data")
        else:
            raise ValueError("No data loaded. Call load_repository() first.")
