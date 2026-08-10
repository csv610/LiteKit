# LiteLLM Examples

This directory contains examples demonstrating various features of the LiteLLM library.

## 📁 Directory Structure

```
examples/
├── basic/                    # Simple usage examples
├── structured_output/        # JSON/Pydantic structured responses
├── web_search/              # Web search and URL processing
├── advanced/                # Advanced features and patterns
└── simple_example.py        # Basic litellm usage
```

---

## 🚀 Basic Examples

### simple_example.py
**Purpose**: Demonstrates direct usage of litellm with different providers.

**Usage**:
```bash
python examples/simple_example.py "What is AI?"
```

**Features**:
- Switch between models (Claude, Llama, Perplexity, Gemini)
- Direct litellm.completion() API usage
- Minimal configuration

---

### basic/gemini_chat.py
**Purpose**: Simple Gemini chat with reasoning efforts parameter.

**Usage**:
```bash
python examples/basic/gemini_chat.py "Explain quantum computing"
```

**Features**:
- Gemini 2.5 Flash model
- Reasoning efforts configuration
- Full response + extracted content

---

## 📋 Structured Output Examples

These examples demonstrate using Pydantic models for structured JSON responses.

### structured_output/medicine_info.py
**Purpose**: Get structured medicine information using Pydantic models.

**Usage**:
```bash
python examples/structured_output/medicine_info.py "aspirin"
```

**Features**:
- Pydantic BaseModel for response validation
- Structured fields: name, brand, description, history, ingredients, uses, side effects
- JSON output formatting

---

### structured_output/nobel_prize_info.py
**Purpose**: Retrieve structured Nobel Prize information.

**Usage**:
```bash
python examples/structured_output/nobel_prize_info.py "2024 Physics"
```

---

### structured_output/drugbank_medicine.py
**Purpose**: Comprehensive drug information with DrugBank-style schema.

**Usage**:
```bash
python examples/structured_output/drugbank_medicine.py "metformin"
```

---

### structured_output/jsonout.py
**Purpose**: Simple JSON output formatting example.

**Usage**:
```bash
python examples/structured_output/jsonout.py
```

---

## 🌐 Web Search Examples

### web_search/google_search.py
**Purpose**: Integrate Google Search with Gemini model.

**Usage**:
```bash
python examples/web_search/google_search.py "latest AI developments"
```

---

### web_search/websearch.py
**Purpose**: Search latest news with configurable context.

**Usage**:
```bash
python examples/web_search/websearch.py -i "climate change" -m summary -c high
```

**Arguments**:
- `-i, --topic`: Topic to search for
- `-m, --mode`: Search mode (summary or list)
- `-c, --context_size`: Search context (low, medium, high)

---

### web_search/url_explain.py
**Purpose**: Explain or summarize content from URLs.

---

## 🔬 Advanced Examples

### advanced/perplx_chat.py
**Purpose**: Advanced Perplexity chat with structured response parsing.

**Features**:
- Perplexity API integration
- Structured response with citations
- Image and source extraction

---

## 📚 For Complete CLI Usage

For the full-featured unified CLI:

```bash
python scripts/liteclient_cli.py -q "What is AI?"
python scripts/liteclient_cli.py -i photo.jpg
streamlit run scripts/streamlit_liteclient.py
```

See [../README.md](../README.md) for details.
