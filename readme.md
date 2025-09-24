# Mini RAG System (Movie Plots)

> This repository contains a minimal Retrieval-Augmented Generation (RAG) pipeline for answering questions about movie plots.
> 
> **Pipeline:** data → chunking → embeddings → FAISS retrieval → LLM generation → structured JSON output.

---

## 📁 Files

- `main.py` — CLI entrypoint
- `utils.py` — utility functions (chunking, embeddings, FAISS, generation)
- `requirements.txt`

---

## ⚙️ Setup

### 1. Create a Python 3.11 virtual environment and install dependencies

#### Windows
```powershell
py -3.11 -m venv venv3.11
.\venv3.11\Scripts\activate
pip install -r requirements.txt
```

#### macOS/Linux
```bash
python3.11 -m venv venv3.11
source venv3.11/bin/activate
pip install -r requirements.txt
```

### 2. Prepare the dataset

Download a subset (200–500 rows) of the [Wikipedia Movie Plots dataset from Kaggle](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots).

Save it as `movie_plots_subset.csv` (the script auto-detects title and plot columns).

Or use your own CSV with columns containing 'title' and 'plot' in their names.

---

### 3. (Optional) OpenAI Setup
Currently this option is not working due to version mismatch. Tried to resolve. It is taking some more time

If you want OpenAI embeddings and generation, set the following environment variables:

```bash
# Example environment variables
CSV_PATH=wiki_movie_plots_deduped.csv
SUBSET=200
CHUNK_WORDS=300
TOP_K=3

OPENAI_API_KEY=sk-...
OPENAI_EMBED_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-4o-mini

OUTPUT_FILE=output.json
USE_OPENAI=true
```

---

## 🚀 Run

### Basic local run (no OpenAI)

```bash
python main.py --csv movie_plots_subset.csv --subset 300 --chunk_words 300
```

### Use OpenAI for embeddings + generation

```powershell
# On Windows (PowerShell):
$env:OPENAI_API_KEY="..."
python main.py --csv movie_plots_subset.csv --subset 300 --chunk_words 300 --use_openai
```

```bash
# On macOS/Linux:
export OPENAI_API_KEY="..."
python main.py --csv movie_plots_subset.csv --subset 300 --chunk_words 300 --use_openai
```

When running, the script builds an in-memory FAISS index, then opens an interactive prompt. Type a question (e.g. `Which movies feature an AI?`) and it will return a JSON with:

- `answer` — the generated answer
- `contexts` — top-k retrieved snippets (title + snippet)
- `reasoning` — short explanation of how the answer was formed
- `topk_scores` — similarity scores

Output is saved to `output.json` by default.

---

## 📝 Example

```json
> Which movie has an AI antagonist?
{
  "answer": "The movie *2001: A Space Odyssey* features the AI HAL 9000, which becomes antagonistic toward the crew.",
  "contexts": [
    "2001: A Space Odyssey: The HAL 9000 computer becomes antagonistic..."
  ],
  "reasoning": "Found the 'HAL 9000' mention in the retrieved plot chunk and used it to answer.",
  "topk_scores": [0.92]
}
```