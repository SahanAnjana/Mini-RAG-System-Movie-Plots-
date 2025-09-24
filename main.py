import argparse
import json
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from utils import (
    chunk_text,
    build_faiss_index,
    embed_texts,
    retrieve_topk,
    generate_answer_openai,
    generate_answer_local,
)

load_dotenv()

DEFAULT_CSV = os.getenv("CSV_PATH", "wiki_movie_plots_deduped.csv")
DEFAULT_SUBSET = int(os.getenv("SUBSET", 200))
DEFAULT_CHUNK_WORDS = int(os.getenv("CHUNK_WORDS", 300))
DEFAULT_TOP_K = int(os.getenv("TOP_K", 3))
DEFAULT_USE_OPENAI = os.getenv("USE_OPENAI", "false").lower() == "true"
DEFAULT_OUTPUT = os.getenv("OUTPUT_FILE", "output.json")


def load_dataset(csv_path: str, subset: int = DEFAULT_SUBSET) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    title_col, plot_col = None, None
    for c in df.columns:
        cl = c.lower()
        if "title" in cl and title_col is None:
            title_col = c
        if "plot" in cl and plot_col is None:
            plot_col = c
    df = df[[title_col, plot_col]].dropna().rename(columns={title_col: "title", plot_col: "plot"})
    if subset and subset < len(df):
        df = df.sample(n=subset, random_state=42).reset_index(drop=True)
    return df


def prepare_chunks(df: pd.DataFrame, words_per_chunk: int = DEFAULT_CHUNK_WORDS):
    chunks = []
    idx = 0
    for _, row in df.iterrows():
        title, plot = row["title"], str(row["plot"])
        for c in chunk_text(plot, words_per_chunk):
            chunks.append({"id": idx, "title": title, "text": c})
            idx += 1
    return chunks


def run_query(query: str, chunks, index, top_k, use_openai):
    q_emb, _ = embed_texts([query], use_openai=use_openai)
    q_emb = np.array(q_emb).astype("float32")
    topk_idxs, _ = retrieve_topk(index, q_emb, top_k=top_k)
    topk_idxs = topk_idxs[0]

    contexts = []
    for idx in topk_idxs:
        if idx < 0 or idx >= len(chunks):
            continue
        snippet = chunks[idx]["text"]
        contexts.append(f"{chunks[idx]['title']}: {snippet[:500]}")

    if use_openai:
        answer_text, reasoning = generate_answer_openai(query, contexts)
    else:
        answer_text, reasoning = generate_answer_local(
            query, contexts, model_name="google/flan-t5-small", hf_model=None
        )

    return {
        "answer": answer_text,
        "contexts": contexts,
        "reasoning": reasoning,
    }


def main():
    parser = argparse.ArgumentParser(description="Mini RAG pipeline for movie plots")
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV, help="Path to movie plots CSV")
    parser.add_argument("--subset", type=int, default=DEFAULT_SUBSET, help="Number of rows to use")
    parser.add_argument("--chunk_words", type=int, default=DEFAULT_CHUNK_WORDS, help="Words per chunk")
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K, help="Top-k retrieved chunks")
    parser.add_argument("--use_openai", action="store_true", default=DEFAULT_USE_OPENAI, help="Use OpenAI for embeddings+generation")
    parser.add_argument("--question", type=str, help="Run in one-shot mode with this question")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Path to save JSON output")
    args = parser.parse_args()

    print("📂 Loading dataset...")
    df = load_dataset(args.csv, subset=args.subset)
    print(f"✅ Loaded {len(df)} rows")

    print("✂️  Splitting plots into chunks...")
    chunks = prepare_chunks(df, words_per_chunk=args.chunk_words)
    print(f"✅ Created {len(chunks)} chunks")

    print("🔎 Embedding corpus...")
    if args.use_openai and "OPENAI_API_KEY" not in os.environ:
        raise ValueError("Set OPENAI_API_KEY in your .env or environment variables.")
    emb, _ = embed_texts([c["text"] for c in chunks], use_openai=args.use_openai)
    emb = np.array(emb).astype("float32")

    print("📚 Building FAISS index...")
    index = build_faiss_index(emb)

    if args.question:
        result = run_query(args.question, chunks, index, args.top_k, args.use_openai)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved to {args.output}")
    else:
        print("\n💬 Interactive mode. Type a question (or 'quit' to exit).")
        while True:
            query = input("\n> ").strip()
            if query.lower() in ("quit", "exit"):
                print("👋 Exiting.")
                break
            result = run_query(query, chunks, index, args.top_k, args.use_openai)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"💾 Saved to {args.output}")


if __name__ == "__main__":
    main()
