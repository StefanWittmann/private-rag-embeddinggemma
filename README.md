# Private RAG System with EmbeddingGemma

A 100% private, local Retrieval-Augmented Generation (RAG) stack using:
- **EmbeddingGemma-300m** for embeddings  
- **SQLite-vec** for vector storage
- **Crawl4AI** for async web scraping
- **gpt-oss:20b** for language generation
- **100% Private & Offline Capable**

## 🎯 What This Project Does

Build a completely private, offline RAG application right on your laptop. This system combines Google's new EmbeddingGemma model for best-in-class local embeddings, SQLite-vec for a dead-simple vector database, Crawl4AI for intelligent web scraping, and Ollama for a powerful, local LLM. No API keys, no costs, no data sent to the cloud.

## 🔧 How It Works

1. **Web Scraping (Crawl4AI)**: Uses Crawl4AI's async web crawler to fetch documentation pages and automatically convert them to clean markdown format
2. **Chunking**: Splits documents into overlapping token-based chunks for optimal retrieval
3. **Embedding (EmbeddingGemma)**: Generates dense vector representations using Google's mobile-optimized embedding model
4. **Storage (SQLite-vec)**: Stores vectors in a local SQLite database with vector search extension
5. **Retrieval**: Finds the most semantically similar chunks to your query via cosine similarity
6. **Generation (gpt-oss)**: Passes retrieved context to a local LLM to generate accurate, grounded responses

## 📋 Prerequisites

- **Python 3.13+** recommended (for Crawl4AI compatibility)
- Modern laptop with at least 8GB RAM
- Internet connection for initial model downloads

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone <your-repo>
cd embeddinggemma
```

### 2. Install UV (if not already installed)

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv
```

### 3. Install Dependencies

```bash
# Install all project dependencies
uv sync
```

### 4. Install Playwright for Crawl4AI

Crawl4AI uses Playwright for browser automation:

```bash
# Install Chromium browser for Crawl4AI
playwright install chromium
```

### 5. Setup Ollama

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve &

# Pull the gpt-oss model
ollama pull gpt-oss:20b
```

### 6. Hugging Face Authentication

EmbeddingGemma requires Hugging Face access:

1. Request access at: https://huggingface.co/google/embeddinggemma-300m
2. Wait for approval (usually within 24 hours)
3. Login via CLI:

```bash
# Login to Hugging Face
uv run huggingface-cli login
```

### 7. Run the Demo

```bash
# Run the RAG system
uv run python rag_demo.py
```

## 📔 Jupyter Notebook Setup

To use this project with Jupyter notebooks in a standalone virtual environment:

### Step 1: Add Jupyter Dependencies

```bash
# Add Jupyter packages to your project
uv add jupyter notebook ipykernel
```

### Step 2: Register Jupyter Kernel

```bash
# Register your virtual environment as a Jupyter kernel
uv run python -m ipykernel install --user --name embeddinggemma --display-name "EmbeddingGemma RAG"
```

### Step 3: Launch Jupyter

```bash
# Start Jupyter
uv run jupyter notebook

# Or use Jupyter Lab
uv run jupyter lab
```

### Step 4: Use the Correct Kernel

1. Open your notebook
2. Go to **Kernel** → **Change kernel** → **EmbeddingGemma RAG**
3. Now all your project dependencies are available!

## 🏗️ Project Structure

```
embeddinggemma/
├── .venv/                  # Virtual environment
├── docs/                   # Scraped documentation output (markdown files)
├── rag_demo.py            # Main RAG demonstration script
├── rag_demo.ipynb         # Complete tutorial notebook  
├── pyproject.toml         # Project dependencies (uv format)
├── requirements.txt       # Alternative pip format
└── vectors_docs.db        # SQLite vector database (generated)
```

## 🔧 Configuration

Key parameters you can modify:

```python
EMBEDDING_MODEL = "google/embeddinggemma-300m"
EMBEDDING_DIMS = 256  # 256 for 3x speed, 768 for max quality
LLM_MODEL = "gpt-oss:20b"  # Try: llama3:8b, mistral:7b
DRY_RUN = False  # Set True to test without LLM
```

## 🧪 Usage Examples

### Command Line
```bash
uv run python rag_demo.py
```

### In Python/Jupyter
```python
from rag_demo import *

# Query the system
response = semantic_search_and_query("How do I use SQLite-vec with Python?")
```

## 🔍 Troubleshooting

### Common Issues

#### "pip not found" in Jupyter
**Solution**: Make sure you're using the correct kernel
1. Register kernel: `uv run python -m ipykernel install --user --name embeddinggemma --display-name "EmbeddingGemma RAG"`
2. Switch kernel in Jupyter to "EmbeddingGemma RAG"

#### "Command not found: jupyter"
**Solution**: Install Jupyter in your environment
```bash
uv add jupyter notebook ipykernel
uv sync
```

#### EmbeddingGemma Access Denied
**Solution**: Request access and wait for approval
1. Visit: https://huggingface.co/google/embeddinggemma-300m
2. Click "Request access to this repo"
3. Wait 24 hours for approval
4. Run `uv run huggingface-cli login`

#### Ollama Connection Error  
**Solution**: Ensure Ollama is running
```bash
# Check if running
ps aux | grep ollama

# Start if not running
ollama serve &

# Pull model if needed
ollama pull gpt-oss:20b
```

#### Crawl4AI / Playwright Issues
**Solution**: Ensure Playwright browsers are installed
```bash
# Install Chromium for Crawl4AI
playwright install chromium

# If you get permission issues, try:
playwright install --with-deps chromium
```

#### Out of Memory Errors
**Solutions**:
- Reduce `EMBEDDING_DIMS` to 256
- Use smaller batch sizes
- Try a smaller LLM model
- Close other applications

### Verification Commands

Check your setup:
```bash
# Verify environment is activated
which python  # Should show .venv path

# Test imports
uv run python -c "import sqlite_vec, ollama, sentence_transformers; print('All imports working!')"

# Check Ollama
ollama list  # Should show gpt-oss:20b

# Test Jupyter kernel
jupyter kernelspec list  # Should show embeddinggemma kernel

# Test Crawl4AI
uv run python -c "from crawl4ai import AsyncWebCrawler; print('Crawl4AI ready!')"
```

## 📊 System Requirements

- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: ~3GB for models + data
- **Models Downloaded**:
  - EmbeddingGemma-300m: ~600MB
  - gpt-oss:20b: model size varies

## 🛠️ Advanced Customization

### Add Custom Documentation
Edit `DOCUMENTATION_URLS` in the script to scrape your own docs. Crawl4AI handles JavaScript-rendered pages and automatically extracts clean markdown.

### Different Models
- **Embeddings**: Try `google/embeddinggemma-768` for higher quality
- **LLM**: Try `llama3:8b`, or `mistral:7b`

### Chunking Strategy
Modify token-based chunking parameters:
```python
max_tokens = 2048      # Chunk size
overlap_tokens = 100   # Overlap between chunks
```

## 🎯 Benefits

✅ **100% Private**: All processing happens locally  
✅ **Zero Cost**: No API fees after initial setup  
✅ **Mobile-Optimized**: EmbeddingGemma designed for mobile deployment  
✅ **Fast**: SQLite-vec provides sub-millisecond vector search  
✅ **Smart Scraping**: Crawl4AI handles JavaScript-rendered pages and returns clean markdown  
✅ **Standalone**: Complete isolation in virtual environment  

## 📜 License

This project is open source. See individual model licenses:
- EmbeddingGemma: Gemma License
- SQLite-vec: Apache 2.0
- Crawl4AI: Apache 2.0

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 🔗 Links

- [EmbeddingGemma](https://huggingface.co/google/embeddinggemma-300m)
- [SQLite-vec](https://github.com/asg017/sqlite-vec)
- [Ollama](https://ollama.ai/)
- [Crawl4AI Documentation](https://docs.crawl4ai.com/)
- [UV Package Manager](https://github.com/astral-sh/uv)