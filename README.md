# mychatbot

A Google Docs RAG (Retrieval-Augmented Generation) chatbot powered by LlamaIndex and OpenAI.

## Setup

### Option 1: pip (recommended for Windows)

Requires **Python 3.10** (Python 3.14 is not compatible with several dependencies).

```bash
pip install -r requirements.txt
```

### Option 2: Conda

```bash
conda env create -f environment.yml
conda activate mychatbot
```

## Configuration

1. Set your OpenAI API key in `main.py`:
   ```python
   os.environ['OPENAI_API_KEY'] = 'your-key-here'
   ```
2. Place your `credentials.json` (Google OAuth) in the project root.

## Run

```bash
python main.py
```