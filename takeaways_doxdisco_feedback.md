# Doxdisco Code Review Takeaways

## None Check Best Practices

### Problem with `or` operator for default values

**❌ Avoid:**
```python
self.text_fields = text_fields or ["content", "filename", "title", "description"]
```

**Problem:** The `or` operator treats ANY falsy value (`None`, `[]`, `""`, `0`, `False`) the same. If someone intentionally passes an empty list `[]`, it would be replaced with the default value, overriding the caller's intent.

**✅ Preferred:**
```python
self.text_fields = text_fields if text_fields is not None else ["content", "filename", "title", "description"]
```

**Why:** This only applies the default when the argument is actually `None`. Empty lists, empty strings, and other falsy values are preserved as the caller intended.

### Concrete Example

```python
# BAD: Using 'or'
class TextRAG:
    def __init__(self, text_fields=None):
        self.text_fields = text_fields or ["content", "filename"]

# User wants to disable text fields
rag = TextRAG(text_fields=[])
# Result: Gets ["content", "filename"] - WRONG! User's intent ignored ❌

# GOOD: Using 'is not None'
class TextRAG:
    def __init__(self, text_fields=None):
        self.text_fields = text_fields if text_fields is not None else ["content", "filename"]

# User wants to disable text fields
rag = TextRAG(text_fields=[])
# Result: Gets [] - CORRECT! User's intent preserved ✅

# User doesn't pass anything
rag = TextRAG()
# Result: Gets ["content", "filename"] - Default applied correctly ✅
```

**Key point:** With `or`, you can't tell the difference between "not provided" and "intentionally empty". With `is not None`, only `None` triggers the default, all other falsy values are respected.

### Apply this pattern to any optional argument that can have meaningful falsy values:
- `allowed_extensions` (if it can be an empty set)
- `filename_filter` (if it can be a callable)
- `chunk_size` (if `0` has meaning)
- Any list/collection parameters

## Import Practices

### Prefer absolute imports over relative imports
- Absolute imports are clearer and more maintainable
- Example: `from prompt.models import RAGAnswer` instead of `from .models import RAGAnswer`
- Avoids confusion about module location

### Lazy imports only when necessary
- If there's no circular import, move imports to the top of the file
- Lazy imports (inside functions) are only needed to break actual circular dependencies
- Don't add "avoid circular imports" comments if there are no circular imports

## Package Structure

### Root-level published modules (`cli.py`, `config.py`)
- Use `py-modules` in `pyproject.toml` to include root-level modules
- Standard practice for packages that have both:
  - Subpackages (like `core/`, `prompt/`)
  - Root-level modules (like `cli.py`, `config.py`)

### Alternatives considered
- Moving `cli.py` into a package would work but is more complex
- Require: creating new package directory, updating imports, updating entry points
- The `py-modules` approach is simpler and cleaner for this structure

## When to use what:
- **`or` operator**: Only when `None` and falsy values should be treated the same (rare)
- **`is not None` check**: When `None` vs empty/falsy values have different meanings (common)
- **Lazy imports**: Only when you have actual circular dependencies
- **Absolute imports**: Default choice, always preferred
