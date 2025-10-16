<h1 align="center">DoxDisco - GitHub Documentation Discovery🔍</h1>
<p align="center">CLI for asking questions about GitHub repository documentation using RAG (Retrieval-Augmented Generation).</p>

## Features
* Ask questions about GitHub repository documentation
* Support for text and vector search methods
* Multiple search types (text, vector with Minsearch, vector with SentenceTransformers)

## Prerequisites
* Python 3.11+
* OpenAI API key (set as OPENAI_API_KEY environment variable)
* uv (recommended) or pip

## Quick Start

```bash
# Ask a basic question about repository documentation
$ disco ask "How do I install this package?"

# Use vector search for better semantic understanding
$ disco ask "How to configure authentication?" --search-type vector_sentence_transformers

# Customize chunking parameters (chunk_size, overlap)
$ disco ask "What are the main features?" 1500 0.3

# Ask about a different repository
$ disco ask "What is this project about?" --owner pydantic --repo pydantic-ai
```

## Examples

### Search Types
```bash
# Text search (fast, keyword-based)
$ disco ask "installation guide" --search-type text

# Vector search (better semantic understanding)
$ disco ask "authentication setup" --search-type vector_sentence_transformers
```

### Repository Options
```bash
# Different repository
$ disco ask "What are the main features?" --owner facebook --repo react

# Different file types
$ disco ask "API documentation" --extensions md,rst,txt
```

## Configuration

### Environment Variables

Create a `.env` file in your project root:

```
OPENAI_API_KEY=your_openai_api_key_here
```

### Default Settings

- **Default Repository**: pydantic/pydantic-ai
- **Default File Extensions**: md, mdx
- **Default Search Type**: text
- **Default Chunk Size**: 2000 characters
- **Default Overlap**: 0.5 (50%)

## Notes

- Repository content is downloaded and processed on first use
- Vector search models are downloaded automatically when needed
- Use `--verbose` flag to see detailed processing information
- Use `--help` for complete command reference


## Future Updates
* [ ] Add support for private repositories with authentication
* [ ] Implement caching for better performance
* [ ] Add support for more file formats (PDF, Word docs)
* [ ] Implement conversation memory for follow-up questions
