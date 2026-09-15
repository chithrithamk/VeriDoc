"""
Diagnostic script for VeriDoc Semantic Retrieval
"""
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import pymupdf as fitz
except ImportError:
    import fitz

from backend.services.pdf_processor import ExtractedDocument, PageData, extract_text_from_pdf
from backend.services.chunker import chunk_document, chunk_text
from backend.services.embeddings import embed_chunks, embed_text, load_embedding_model, get_embedding_dimension
from backend.services.vector_store import FAISSVectorStore
from backend.services.retrieval import RetrievalService
from backend.services.rag_pipeline import RAGPipeline


def create_sample_multipage_pdf(file_path: Path) -> Path:
    doc = fitz.open()

    # Page 1: Title, Abstract, Introduction
    p1 = doc.new_page()
    p1_text = (
        "Report: Enterprise Security and Infrastructure Audit 2026\n"
        "Executive Summary & Introduction\n"
        "This comprehensive report evaluates organizational infrastructure, access control mechanisms, "
        "incident response frameworks, and cloud security postures across all global business units. "
        "The objective of this annual audit is to identify vulnerabilities, assess compliance with ISO 27001, "
        "and formulate strategic recommendations for the board of directors. Subsequent sections outline the "
        "audit background, system architecture, methodology, experimental penetration testing results, and conclusions."
    )
    p1.insert_text(fitz.Point(50, 72), p1_text)

    # Page 2: Background & Compliance Scope
    p2 = doc.new_page()
    p2_text = (
        "Section 2: Background and Regulatory Compliance Scope\n"
        "Over the past fiscal year, regulatory requirements under GDPR, HIPAA, and SOC2 Type II have tightened. "
        "Our assessment scope encompassed thirty-two hybrid cloud data centers across North America, Europe, and APAC. "
        "Previous audits identified minor discrepancies in access logging and cryptographic key rotation cycles. "
        "This review benchmarks current operational controls against international cybersecurity standards."
    )
    p2.insert_text(fitz.Point(50, 72), p2_text)

    # Page 3: System Architecture & Cloud Infrastructure
    p3 = doc.new_page()
    p3_text = (
        "Section 3: Cloud Infrastructure and Network Architecture\n"
        "The corporate network is designed as a Zero-Trust Architecture (ZTA) deployed across AWS and Google Cloud Platform. "
        "Core microservices communicate via mutual TLS (mTLS) within Kubernetes clusters orchestrated by Istio service mesh. "
        "Edge routing is managed by Cloudflare Web Application Firewalls (WAF) with automated DDoS mitigation. "
        "Identity verification relies on Okta Single Sign-On (SSO) integrated with FIDO2 hardware security keys."
    )
    p3.insert_text(fitz.Point(50, 72), p3_text)

    # Page 4: Audit Methodology & Data Collection Techniques
    p4 = doc.new_page()
    p4_text = (
        "Section 4: Audit Methodology and Data Collection Procedures\n"
        "The auditing methodology employed a hybrid three-tier assessment framework consisting of automated vulnerability scanning, "
        "manual penetration testing, and static source code analysis. Automated vulnerability discovery utilized Nessus Professional "
        "and Qualys VMDR with custom credentialed scanning scripts. Penetration testers executed simulated adversarial attack vectors "
        "modeled after the MITRE ATT&CK enterprise matrix (TA0001 through TA0011). Code analysis was conducted with SonarQube and Semgrep "
        "covering over 1.2 million lines of Python, Go, and TypeScript repository code. Sampling intervals spanned 45 continuous days "
        "from January 15 to February 28, collecting 85,000 endpoint telemetry logs."
    )
    p4.insert_text(fitz.Point(50, 72), p4_text)

    # Page 5: Findings, Vulnerabilities & Empirical Results
    p5 = doc.new_page()
    p5_text = (
        "Section 5: Empirical Findings and Vulnerability Analysis\n"
        "Testing revealed a total of 14 vulnerabilities: 2 Critical, 5 High, 4 Medium, and 3 Low severity issues. "
        "The critical vulnerabilities involved an unpatched remote code execution (RCE) flaw in an legacy internal billing portal "
        "(CVE-2025-4921) and an over-permissive IAM role allowing privilege escalation in the AWS staging environment. "
        "High-severity findings included missing rate limits on the public customer authentication API and unencrypted Redis cache volumes. "
        "Average mean time to detect (MTTD) during red team exercises was 14.2 minutes, with a mean time to remediate (MTTR) of 3.8 hours."
    )
    p5.insert_text(fitz.Point(50, 72), p5_text)

    # Page 6: Recommendations, Budget & Remediation Roadmap
    p6 = doc.new_page()
    p6_text = (
        "Section 6: Remediation Roadmap and Investment Recommendations\n"
        "To address identified security gaps, we recommend a phased 90-day remediation roadmap with an estimated budget of $450,000. "
        "Phase 1 requires immediate patching of CVE-2025-4921 and strict IAM least-privilege policy enforcement within 7 days. "
        "Phase 2 mandates deploying hardware security tokens for all database administrators and enabling KMS envelope encryption across all S3 buckets. "
        "Phase 3 establishes an ongoing 24/7 Managed Detection and Response (MDR) SOC contract to monitor anomalous telemetry."
    )
    p6.insert_text(fitz.Point(50, 72), p6_text)

    file_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(file_path))
    doc.close()
    return file_path


def run_diagnostics():
    pdf_path = PROJECT_ROOT / "data" / "documents" / "diagnostic_audit_report.pdf"
    create_sample_multipage_pdf(pdf_path)

    pipeline = RAGPipeline()
    stats = pipeline.ingest_pdf(pdf_path, chunk_size=1000, chunk_overlap=200)

    print("=" * 80)
    print("INGESTION STATS:")
    print(stats)
    print(f"Total Chunks: {len(pipeline.chunks)}")
    for c in pipeline.chunks:
        print(f"  Chunk ID={c.chunk_id} | Page={c.page_number} | Chars={c.char_count} | Preview: {c.text[:80]}...")
    print("=" * 80)

    # Test Queries targeting different pages
    test_questions = [
        ("What methodology and tools were used for data collection and testing?", 4),
        ("What vulnerabilities and empirical findings were discovered?", 5),
        ("How is the cloud network architecture designed?", 3),
        ("What are the remediation recommendations and estimated budget?", 6),
        ("What is the regulatory compliance scope and background?", 2),
        ("What is the executive summary and purpose of the report?", 1),
    ]

    print("\nRETRIEVAL RESULTS (TOP 10 PER QUESTION):\n")

    for q_text, expected_page in test_questions:
        print("-" * 80)
        print(f"QUERY: '{q_text}' (Target: Page {expected_page})")
        print("-" * 80)
        results = pipeline.retrieve(q_text, top_k=10)
        for rank, res in enumerate(results, start=1):
            c = res.chunk
            print(f"Rank {rank:2d} | Chunk #{c.chunk_id:2d} | Page {c.page_number} | Score: {res.score:.4f} | Text: {c.text[:140]}...")
        print()

    # Clean up test pdf
    if pdf_path.exists():
        pdf_path.unlink()

if __name__ == "__main__":
    run_diagnostics()
