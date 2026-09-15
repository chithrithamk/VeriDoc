"""
Detailed Scenario Test: Investigating semantic retrieval discrepancy
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import pymupdf as fitz
except ImportError:
    import fitz

from backend.services.pdf_processor import extract_text_from_pdf
from backend.services.rag_pipeline import RAGPipeline
from backend.services.embeddings import load_embedding_model


def run_realistic_academic_paper_test():
    doc = fitz.open()

    # Page 1: Abstract & Intro mentioning high-level terms
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

    # Page 2: Related Work / Background
    p2 = doc.new_page()
    p2_text = (
        "2. Related Work and Prior Research\n"
        "Previous research in retrieval-augmented generation focused on dense passage retrieval and BM25 indexing. "
        "Standard retrieval methodologies often suffer from semantic drift when handling technical documentation. "
        "Traditional verification methods relied on rule-based heuristic parsers and entity extraction pipelines."
    )
    p2.insert_text(fitz.Point(50, 72), p2_text)

    # Page 3: System Architecture
    p3 = doc.new_page()
    p3_text = (
        "3. System Architecture\n"
        "The overall system pipeline consists of a document ingestion layer, a sentence-boundary chunker, "
        "a dense embedding generator using SentenceTransformer models, and a FAISS vector index using IndexFlatIP. "
        "Incoming queries are vectorized and matched against stored chunk embeddings in sub-milliseconds."
    )
    p3.insert_text(fitz.Point(50, 72), p3_text)

    # Page 4: Detailed Methodology (The actual ground-truth facts!)
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

    # Page 5: Results & Discussion
    p5 = doc.new_page()
    p5_text = (
        "5. Experimental Results and Discussion\n"
        "Our model achieved 94.2% precision and 91.8% recall on citation attribution benchmarks, outperforming baseline BM25 by 18.4%. "
        "Latency benchmarks showed an average retrieval time of 1.4 milliseconds per query on CPU. "
        "Error analysis indicated that failures were primarily caused by complex tabular data and multi-column document layouts."
    )
    p5.insert_text(fitz.Point(50, 72), p5_text)

    pdf_path = PROJECT_ROOT / "data" / "documents" / "academic_paper_test.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(pdf_path))
    doc.close()

    pipeline = RAGPipeline()
    pipeline.ingest_pdf(pdf_path, chunk_size=1000, chunk_overlap=200)

    print("=" * 90)
    print("ANALYSIS OF CHUNKS AND MODEL METRICS")
    print("=" * 90)
    model = load_embedding_model()
    print(f"Model: {pipeline.embedding_model}")
    print(f"Max Seq Length: {model.max_seq_length}")
    print(f"Embedding Dimension: {model.get_sentence_embedding_dimension()}")
    print(f"Total Chunks: {len(pipeline.chunks)}")
    for c in pipeline.chunks:
        print(f"  Chunk #{c.chunk_id} | Page {c.page_number} | Chars: {len(c.text)} | First 100 chars: {repr(c.text[:100])}")

    test_queries = [
        "What is the methodology used in this study?",
        "What methodology and experimental protocol were used?",
        "How was the model trained and what were the implementation details?",
        "What were the experimental results and precision recall scores?",
        "What are the system architecture components?",
    ]

    for q in test_queries:
        print("\n" + "=" * 90)
        print(f"QUERY: '{q}'")
        print("=" * 90)
        results = pipeline.retrieve(q, top_k=5)
        for rank, res in enumerate(results, start=1):
            c = res.chunk
            print(f"Rank {rank} | Chunk #{c.chunk_id} | Page {c.page_number} | Score: {res.score:.4f}")
            print(f"   Excerpt: {c.text[:200]}...\n")

    if pdf_path.exists():
        pdf_path.unlink()


if __name__ == "__main__":
    run_realistic_academic_paper_test()
