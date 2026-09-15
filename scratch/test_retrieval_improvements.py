"""
Test potential general retrieval improvements:
1. Pure Dense (all-MiniLM-L6-v2)
2. BM25 (Lexical / Sparse)
3. Hybrid BM25 + Dense RRF (Reciprocal Rank Fusion)
4. Higher Top-K + Reranker
"""
import sys
from pathlib import Path
import re
import math
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import pymupdf as fitz
except ImportError:
    import fitz

from backend.services.rag_pipeline import RAGPipeline


# Simple pure-Python BM25 implementation for experimentation
class SimpleBM25:
    def __init__(self, corpus_texts, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus = [self._tokenize(t) for t in corpus_texts]
        self.doc_len = [len(d) for d in self.corpus]
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 1
        self.doc_count = len(self.corpus)
        self.df = Counter()
        for doc in self.corpus:
            self.df.update(set(doc))
        self.idf = {}
        for word, freq in self.df.items():
            self.idf[word] = math.log((self.doc_count - freq + 0.5) / (freq + 0.5) + 1.0)

    def _tokenize(self, text):
        return re.findall(r'\w+', text.lower())

    def score(self, query):
        q_tokens = self._tokenize(query)
        scores = []
        for i, doc in enumerate(self.corpus):
            doc_tf = Counter(doc)
            score = 0.0
            dl = self.doc_len[i]
            for token in q_tokens:
                if token in doc_tf:
                    tf = doc_tf[token]
                    idf = self.idf.get(token, 0.0)
                    denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl))
                    score += idf * (tf * (self.k1 + 1.0)) / denom
            scores.append(score)
        return scores


def run_comparison():
    doc = fitz.open()

    p1 = doc.new_page()
    p1_text = (
        "Neural Document Verification: An Empirical Study\n"
        "Authors: Research Team\n\n"
        "Abstract: In this paper, we present our methodology for verifying document citations and detecting hallucinations. "
        "We discuss our research methodology, system design, comparative evaluation methodology, and performance benchmarks. "
        "Our methodology achieves state-of-the-art results across diverse datasets.\n\n"
        "1. Introduction: As generative language models become prevalent, factual consistency is critical. "
        "In Section 2 we review background literature. In Section 3 we detail our data structures. "
        "In Section 4 we present our methodology and experimental pipeline. In Section 5 we analyze findings."
    )
    p1.insert_text(fitz.Point(50, 72), p1_text)

    p2 = doc.new_page()
    p2_text = (
        "2. Related Work and Prior Research\n"
        "Previous research in retrieval-augmented generation focused on dense passage retrieval and BM25 indexing. "
        "Standard retrieval methodologies often suffer from semantic drift when handling technical documentation. "
        "Traditional verification methods relied on rule-based heuristic parsers and entity extraction pipelines."
    )
    p2.insert_text(fitz.Point(50, 72), p2_text)

    p3 = doc.new_page()
    p3_text = (
        "3. System Architecture\n"
        "The overall system pipeline consists of a document ingestion layer, a sentence-boundary chunker, "
        "a dense embedding generator using SentenceTransformer models, and a FAISS vector index using IndexFlatIP. "
        "Incoming queries are vectorized and matched against stored chunk embeddings in sub-milliseconds."
    )
    p3.insert_text(fitz.Point(50, 72), p3_text)

    p4 = doc.new_page()
    p4_text = (
        "4. Experimental Protocol and Implementation Details\n"
        "The proposed approach was implemented using Python 3.12 and PyTorch. We trained a dual-encoder transformer "
        "on 120,000 annotated legal and scientific PDF passages. For data preprocessing, text was extracted using PyMuPDF, "
        "tokenized with Byte-Pair Encoding (BPE), and segmented into 512-token sequences with 10% overlap. "
        "Optimization used AdamW with weight decay 0.01, cosine learning rate schedule starting at 3e-5, and batch size 64 "
        "across 4 NVIDIA H100 GPUs for 15 epochs. Factual support checking was evaluated using NLI cross-entropy loss."
    )
    p4.insert_text(fitz.Point(50, 72), p4_text)

    p5 = doc.new_page()
    p5_text = (
        "5. Experimental Results and Discussion\n"
        "Our model achieved 94.2% precision and 91.8% recall on citation attribution benchmarks, outperforming baseline BM25 by 18.4%. "
        "Latency benchmarks showed an average retrieval time of 1.4 milliseconds per query on CPU. "
        "Error analysis indicated that failures were primarily caused by complex tabular data and multi-column document layouts."
    )
    p5.insert_text(fitz.Point(50, 72), p5_text)

    pdf_path = PROJECT_ROOT / "data" / "documents" / "test_comparison.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(pdf_path))
    doc.close()

    pipeline = RAGPipeline()
    pipeline.ingest_pdf(pdf_path, chunk_size=1000, chunk_overlap=200)

    bm25 = SimpleBM25([c.text for c in pipeline.chunks])

    queries = [
        "What is the methodology used in this study?",
        "What methodology and experimental protocol were used?",
        "How was the model trained and what were the implementation details?",
        "What were the experimental results and precision recall scores?",
    ]

    for q in queries:
        print("\n" + "=" * 80)
        print(f"QUERY: '{q}'")
        print("=" * 80)

        # 1. Dense (all-MiniLM-L6-v2)
        dense_results = pipeline.retrieve(q, top_k=5)
        dense_rank_map = {res.chunk.chunk_id: (rank, res.score, res.chunk.page_number) for rank, res in enumerate(dense_results, 1)}

        # 2. BM25
        bm25_scores = bm25.score(q)
        bm25_ranked = sorted(enumerate(bm25_scores, 1), key=lambda x: x[1], reverse=True)
        bm25_rank_map = {chunk_id: (rank, score, pipeline.chunks[chunk_id - 1].page_number) for rank, (chunk_id, score) in enumerate(bm25_ranked, 1)}

        # 3. Hybrid Reciprocal Rank Fusion (RRF): RRF_score = 1 / (60 + dense_rank) + 1 / (60 + bm25_rank)
        rrf_scores = {}
        for c in pipeline.chunks:
            cid = c.chunk_id
            d_rank = dense_rank_map[cid][0]
            b_rank = bm25_rank_map[cid][0]
            rrf_score = (1.0 / (60.0 + d_rank)) + (1.0 / (60.0 + b_rank))
            rrf_scores[cid] = (rrf_score, c.page_number, c.text[:80])

        rrf_ranked = sorted(rrf_scores.items(), key=lambda x: x[1][0], reverse=True)

        print("\n[Dense Retrieval - FAISS all-MiniLM-L6-v2]:")
        for rank, res in enumerate(dense_results, 1):
            print(f"  Rank {rank}: Chunk #{res.chunk.chunk_id} (Page {res.chunk.page_number}) | Score: {res.score:.4f} | {res.chunk.text[:70]}...")

        print("\n[Sparse Retrieval - BM25]:")
        for rank, (cid, score) in enumerate(bm25_ranked, 1):
            print(f"  Rank {rank}: Chunk #{cid} (Page {bm25_rank_map[cid][2]}) | Score: {score:.4f} | {pipeline.chunks[cid-1].text[:70]}...")

        print("\n[Hybrid Retrieval - Dense + BM25 RRF]:")
        for rank, (cid, (score, page, text)) in enumerate(rrf_ranked, 1):
            print(f"  Rank {rank}: Chunk #{cid} (Page {page}) | RRF Score: {score:.6f} | {text}...")

    if pdf_path.exists():
        pdf_path.unlink()


if __name__ == "__main__":
    run_comparison()
