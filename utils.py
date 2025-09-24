import os
import math
from typing import List, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

try:
    import openai
except Exception:
    openai = None

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
except Exception:
    pipeline = None
    AutoTokenizer = None
    AutoModelForSeq2SeqLM = None


_EMBED_MODEL_CACHE = {}


def chunk_text(text: str, words_per_chunk: int = 300) -> List[str]:
    words = text.split()
    if len(words) <= words_per_chunk:
        return [" ".join(words)]
    chunks = []
    for i in range(0, len(words), words_per_chunk):
        chunk = " ".join(words[i : i + words_per_chunk])
        chunks.append(chunk)
    return chunks


def embed_texts(texts: List[str], use_openai: bool = False):
    """
    Returns (list_of_embeddings, model_info)
    If use_openai=True, uses OPENAI_API_KEY environment variable (must be set).
    Otherwise uses sentence-transformers 'all-MiniLM-L6-v2'.
    """
    if use_openai:
        if openai is None:
            raise ImportError("openai package not installed. See requirements.")
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in environment.")
        openai.api_key = api_key
        # Using 'text-embedding-3-small' or 'text-embedding-3-large' could be used.
        model = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        batch_size = 32
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = openai.Embeddings.create(input=batch, model=model)
            for r in resp["data"]:
                embeddings.append(r["embedding"])
        return embeddings, {"provider": "openai", "model": model}
    else:
        model_name = "all-MiniLM-L6-v2"
        if model_name not in _EMBED_MODEL_CACHE:
            _EMBED_MODEL_CACHE[model_name] = SentenceTransformer(model_name)
        model = _EMBED_MODEL_CACHE[model_name]
        embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return embeddings.tolist(), {"provider": "sentence-transformers", "model": model_name}


def build_faiss_index(embeddings: List[List[float]]):
    """
    Build a simple FAISS IndexFlatIP index. We normalize vectors to use inner product as cosine similarity.
    """
    xb = np.array(embeddings).astype("float32")
    # normalize
    faiss.normalize_L2(xb)
    d = xb.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(xb)
    return index


def retrieve_topk(index, query_emb, top_k=3):
    """
    query_emb shape: (1, dim) numpy array, already dtype float32
    returns (indices, scores) where each is shape (1, top_k)
    """
    faiss.normalize_L2(query_emb)
    distances, indices = index.search(query_emb, top_k)
    # distances are inner product (cosine) scores
    return indices, distances


# -----------------
# Generation helpers
# -----------------
def generate_answer_openai(query: str, contexts: List[str]) -> Tuple[str, str]:
    if openai is None:
        raise ImportError("openai package not installed.")
    system_prompt = (
        "You are an assistant that answers questions about movie plots. "
        "Use the provided context snippets and cite that context if useful. "
        "If no relevant context is present, say you don't know."
    )
    # prepare message
    context_text = "\n\n".join([f"Context {i+1}: {c}" for i, c in enumerate(contexts)])
    user_prompt = f"Question: {query}\n\nContexts:\n{context_text}\n\nAnswer concisely and then provide a short 'Reasoning:' explanation (1-2 sentences)."
    model = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")  # or gpt-4o-mini/gpt-4o
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        max_tokens=300,
        temperature=0.0,
    )
    content = resp["choices"][0]["message"]["content"].strip()
    # Split into answer and reasoning heuristically
    if "Reasoning:" in content:
        answer, reasoning = content.split("Reasoning:", 1)
        return answer.strip(), reasoning.strip()
    else:
        # fallback: short reasoning
        return content, "I used the retrieved contexts to form the answer."


def generate_answer_local(query: str, contexts: List[str], model_name="google/flan-t5-small", hf_model=None) -> Tuple[str, str]:
    """
    Use a local HF seq2seq for generation. If HF transformers not installed, raise.
    Returns (answer_text, reasoning)
    """
    if pipeline is None:
        raise ImportError("transformers not installed. See requirements.")
    prefix_ctx = "\n\n".join([f"Context {i+1}: {c}" for i, c in enumerate(contexts)])
    prompt = f"Answer the question using only the contexts below. If not found, say 'I don't know'.\n\n{prefix_ctx}\n\nQuestion: {query}\n\nAnswer and then add a short 'Reasoning:' section."
    # instantiate pipeline (cached by HF)
    gen = pipeline("text2text-generation", model=model_name, device=0 if (os.environ.get("CUDA_VISIBLE_DEVICES")) else -1)
    out = gen(prompt, max_length=256, do_sample=False)[0]["generated_text"]
    if "Reasoning:" in out:
        answer, reasoning = out.split("Reasoning:", 1)
        return answer.strip(), reasoning.strip()
    else:
        return out.strip(), "Used the retrieved contexts."
