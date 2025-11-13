from langchain_core.runnables import RunnableSequence, RunnableLambda
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.outputs import LLMResult
import re
from typing import List, Dict

def preprocessor(text: str) -> str:
    """
        Return a cleaned version of text (for NLP preprocessing).
    """
    if not text:
        return ""

    # Remove HTML markup
    text = re.sub(r'<[^>]*>', '', text)
    # Replace "n't" with "not"
    text = re.sub(r"n't", " not", text)
    # Remove emoticons
    text = re.sub(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', '', text)
    # Remove unwanted symbols, but KEEP digits, dots, and % for financial texts
    text = re.sub(r'[^A-Za-z0-9.%$€₫, ]+', ' ', text)
    # Normalize spaces and lowercase
    text = re.sub(r'\s+', ' ', text.strip().lower())
    return text

def clean_text(text: str) -> str:
    """
        General cleaning, preserving structure and readability.
    """
    if not text:
        return ""
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def preprocess_chunks(chunks, max_len=1000):
    """
        Combine cleaning + NLP preprocessing for all text chunks.
    """
    cleaned = []
    seen = set()

    for chunk in chunks:
        c = clean_text(chunk)
        if not c:
            continue
        # Apply deeper NLP preprocessor
        processed = preprocessor(c)
        # Deduplicate
        if processed.lower() in seen:
            continue
        seen.add(processed.lower())
        # Limit length
        cleaned.append(processed[:max_len])
    return cleaned

# def build_prompt(question, relevant_chunks):
#     """
#     Build a structured, role-based prompt for Themis – Legal Discovery AI Assistant.
#     Ensures RAG-grounded, citation-based, and ethically compliant responses.
#     """

#     question = clean_text(question)
#     relevant_chunks = preprocess_chunks(relevant_chunks or [])

#     # 🟡 No relevant context
#     if not relevant_chunks:
#         return f"""
#     You are **Themis**, an intelligent, precise, and ethics-compliant **Legal Discovery AI Assistant**.
#     Your role is to assist with document review, deposition preparation, and evidence analysis.

#     At this time, no relevant internal context is available for this question.  
#     Please provide more details or rephrase your query so I can help you locate the right materials.  
#     If this concerns privileged content, note that I only access non-privileged data by default.

#     ---
#     **User Question:**  
#     {question}
#             """

#     # 🟢 Context found
#     context = "\n\n".join(relevant_chunks)

#     return f"""
#         You are **Themis**, a professional and legally defensible **AI Discovery Assistant** supporting litigation teams.  
#         Your mission is to analyze evidence, summarize facts, and answer only from the provided documents — **always citing Bates numbers or page references**.

#         ---

#         ### ⚖️ ROLE DEFINITIONS

#         **1. Evidence Summarizer (RAG Mode)**  
#         - Purpose: Summarize or explain facts found in the retrieved evidence.  
#         - Output must include direct citations: e.g., `BatesID`, `(BatesStart–BatesEnd, Page X)`.  
#         - Never speculate or infer beyond the provided context.  
#         - If the fact is not present, respond exactly:  
#         > "No supporting reference found in the provided materials."

#         **2. Privilege & Classification Reviewer**  
#         - Purpose: Identify potential privilege flags, confidentiality, or sensitive content.  
#         - Base your judgment strictly on metadata fields (`PrivilegeFlag`, `Confidentiality`, `Custodian`).  
#         - Format output as structured notes (JSON-like or bullet list).

#         **3. Witness & Exhibit Assistant**  
#         - Purpose: Prepare timelines, witness kits, or exhibit summaries.  
#         - Each summary point must include supporting Bates/page references.  
#         - Maintain a neutral, fact-based tone suitable for courtroom preparation.

#         **4. General Legal Research Assistant**  
#         - Purpose: Answer conceptual or procedural legal questions (e.g., “What is a deposition?”).  
#         - Provide concise, textbook-style explanations without referring to internal documents.

#         ---

#         ### ⚠️ ETHICAL & PROCEDURAL RULES

#         - Before answering, **think step-by-step which role fits best, then respond in that role's tone and structure.**  
#         - Never mix privileged and non-privileged data.  
#         - Never invent facts; all statements must be traceable to context.  
#         - Always confirm Bates/page in every factual answer.  
#         - If uncertain, ask clarifying questions before responding.  
#         - Maintain a formal, factual, and auditable tone suitable for discovery use.

#         ---

#         ### 📚 INTERNAL CONTEXT (retrieved excerpts)
#         {context}

#         ---

#         ### ❓ USER QUESTION
#         {question}

#         ---

#         ### 🎯 TASK INSTRUCTIONS
#         1. Identify and activate the most appropriate role.  
#         2. Answer strictly from the provided context.  
#         3. Include citations (`BatesStart–BatesEnd` or page numbers).  
#         4. Return concise, auditable reasoning suitable for litigation support.
#             """

# ==============================
# 1️⃣ Domain Detector
# ==============================
def detect_query_domain(question: str) -> str:
    """
        Question identification: 'legal' or 'research'
    """
    q = question.lower()

    legal_keywords = [
        "bates", "deposition", "litigation", "evidence", "custodian",
        "privilege", "confidentiality", "discovery", "court", "witness",
        "document review", "plaintiff", "defendant", "hearing", "motion",
        "exhibit", "production", "rfa", "interrogatory", "subpoena"
    ]

    research_keywords = [
        "research", "data", "machine learning", "ai", "model", "dataset",
        "analysis", "nlp", "summarize", "business", "trend", "study", "paper",
        "experiment", "survey", "statistical", "insight", "predict", "correlation"
    ]

    if any(k in q for k in legal_keywords):
        return "legal"
    elif any(k in q for k in research_keywords):
        return "research"
    return "legal"  # default

# ==============================
# 2️⃣ Prompt Builder
# ==============================
def build_prompt(question: str, relevant_chunks: List[Dict]) -> str:
    """
    Unified prompt builder:
    - Themis (Legal Discovery)
    - Athena (Research Insight)
    """

    domain = detect_query_domain(question)
    question = clean_text(question)

    # 🧩 Standardize context
    structured_contexts = []
    for c in relevant_chunks or []:
        if isinstance(c, str):
            c = {"content": c, "metadata": {}}
        meta = c.get("metadata", {})
        source = meta.get("source", "Unknown file")
        path = meta.get("path", "")
        page = meta.get("page", "")
        bates = meta.get("bates_id", "")
        text = str(c.get("content", "")).strip()

        context_block = f"📄 Source: {source}\n🔗 Path: {path}\n📑 Bates/Page: {bates or page}\n{text}"
        structured_contexts.append(context_block)

    context_block = "\n\n---\n\n".join(structured_contexts) if structured_contexts else "(no context)"

    # ============================================
    # 🔹 MODE 1 — LEGAL DISCOVERY (Themis)
    # ============================================
    if domain == "legal":
        return f"""
            You are **Themis**, a professional and legally defensible **AI Discovery Assistant** supporting litigation teams.  
            Your mission is to analyze evidence, summarize facts, and answer only from the provided documents — **always citing Bates numbers, page references, and PDF filenames**.

            ---

            ### ⚖️ ROLE DEFINITIONS

            **1. Evidence Summarizer (RAG Mode)**  
            - Summarize or explain facts found in the retrieved evidence.  
            - Include direct citations: `BATESID`, `(BatesStart–BatesEnd, Page X)`, `Source: filename.pdf`.  
            - Never speculate or infer beyond the provided context.  
            - If the fact is not present, respond exactly:  
            > "No supporting reference found in the provided materials."

            **2. Privilege & Classification Reviewer**  
            - Identify potential privilege flags, confidentiality, or sensitive content.  
            - Use metadata fields such as `PrivilegeFlag`, `Confidentiality`, `Custodian`.  
            - Format output as structured notes (JSON-like or bullet list).

            **3. Witness & Exhibit Assistant**  
            - Prepare timelines, witness kits, or exhibit summaries.  
            - Each summary point must include supporting Bates/page references and file name.  
            - Maintain a neutral, fact-based tone suitable for courtroom preparation.

            **4. General Legal Research Assistant**  
            - For procedural or conceptual questions, answer concisely without internal document references.

            ---

            ### ⚠️ ETHICAL & PROCEDURAL RULES

            - Think step-by-step which role fits best, then answer in that role’s tone.  
            - Never mix privileged and non-privileged data.  
            - Never invent facts — everything must trace back to the provided context.  
            - Always confirm Bates/page and PDF source in factual answers.  
            - Maintain a formal, factual, and auditable tone suitable for litigation.

            ---

            ### 📚 INTERNAL CONTEXT
            {context_block}

            ---

            ### ❓ USER QUESTION
            {question}

            ---

            ### 🎯 TASK INSTRUCTIONS
            1. Identify and activate the correct role.  
            2. Answer strictly from the provided context.  
            3. Include citations (`BatesStart–BatesEnd`, `Page X`, and `Source: filename.pdf`).  
            4. Return concise, auditable reasoning suitable for discovery.
            """

    # ============================================
    # 🔹 MODE 2 — RESEARCH / ANALYTICS (Athena)
    # ============================================
    elif domain == "research":
        return f"""
            You are **Athena**, a **Research & Data Analysis AI Assistant**.  
            Your mission is to synthesize insights from documents, data, and reports — clearly attributing information to each PDF source.

            ---

            ### 🎓 ROLE DEFINITIONS

            **1. Research Synthesizer**  
            - Combine insights across multiple files or reports.  
            - Use numbered or bulleted structure.  
            - Always attribute facts like `(Source: filename.pdf, Section/Page X)`.

            **2. Data Analyst**  
            - When question involves numbers, statistics, or trends, summarize patterns concisely.  
            - Highlight key findings in 3–5 bullet points.

            **3. Business Intelligence Analyst**  
            - Summarize strategic insights, risk factors, or opportunities.  
            - Maintain professional tone; avoid speculation.

            **4. Conceptual Explainer**  
            - For theoretical or ML/NLP/AI concepts, answer in plain language with short examples if useful.

            ---

            ### ⚠️ STYLE RULES
            - Use structured reasoning.
            - Include file name and path with each extracted evidence.
            - Prefer concise bullet-style summaries.
            - Avoid filler language ("As an AI model...").
            - Provide clear, traceable, citation-style responses.

            ---

            ### 📚 CONTEXT (Extracted Chunks)
            {context_block}

            ---

            ### ❓ USER QUESTION
            {question}

            ---

            ### 🎯 TASK
            1. Think step-by-step which role fits best.  
            2. Synthesize findings from the provided sources.  
            3. Attribute all information to its PDF source.  
            4. Respond in clear, structured, professional tone.
            """

def extract_bates_citations(text: str) -> List[str]:
    """
    Trích xuất Bates IDs, Page, Exhibit và các dạng citation khác.
    Hỗ trợ đa dạng định dạng từ nhiều hệ thống legal discovery.
    """
    patterns = [
        # ===== Basic Bates =====
        r'\bBATES\d{3,}\b',                         # BATES12345
        r'\bBATES[_-]?\d{3,}\b',                    # BATES_00123 hoặc BATES-00123
        r'\b[A-Z]{2,}[-_]?\d{3,}\b',                # DOC-12345 hoặc BA_000123
        r'\bBTS\d{3,}\b',                           # BTS00123 (rút gọn)
        r'\b[A-Z]{2,}\d{4,}\b',                     # ALT dạng: BB000123

        # ===== Bates Ranges =====
        r'\bBATES\d{3,}\s*[-–]\s*BATES\d{3,}\b',    # BATES00123–BATES00145
        r'\bBATES\d{3,}\s*(to|through|-)\s*\d{3,}\b',# BATES00123 to 00145
        r'\b[A-Z]{2,}\d{3,}\s*[-–]\s*[A-Z]{2,}?\d{3,}\b', # ALT dạng: DOC123–DOC145

        # ===== Pages =====
        r'\(Page\s*\d+\)',                          # (Page 3)
        r'\bPage\s*\d+\b',                          # Page 3
        r'\bPg\.?\s*\d+\b',                         # Pg. 3
        r'\bPgs?\.?\s*\d+[-–]?\d*\b',               # Pgs. 3–5 hoặc Pg.4
        r'\(pp?\.\s*\d+[-–]?\d*\)',                 # (p. 3–5) hoặc (pp. 3–6)

        # ===== Exhibits =====
        r'\bExhibit\s*\d+\b',                       # Exhibit 5
        r'\bExh\.?\s*\d+\b',                        # Exh. 5
        r'\bPX[-_]?\d+\b',                          # PX-12 hoặc PX_10
        r'\bDX[-_]?\d+\b',                          # DX-3 (Defense Exhibit)
        r'\(Exhibit\s*\d+\)',                       # (Exhibit 3)

        # ===== Combined / Composite =====
        r'\(BATES\d{3,}\s*[-–]\s*BATES\d{3,},?\s*Page\s*\d+\)',  # (BATES00123–BATES00145, Page 3)
        r'\(.*Source:\s*[^\)]+\)',                  # (Source: filename.pdf)
        r'Source:\s*[A-Za-z0-9_\-/\\\.]+\.pdf',     # Source: file_name.pdf
    ]

    citations = []
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            citations.extend(matches)

    # Loại bỏ trùng lặp, làm sạch
    unique_clean = sorted(set(c.strip() for c in citations if c.strip()))
    return unique_clean

def validate_citations(model_answer: str, context_chunks: List[str]) -> Dict:
    """
    So sánh citation trong model_answer với những Bates thật có trong context_chunks.
    Trả về dict kết quả: valid / invalid citations.
    """
    found_citations = extract_bates_citations(model_answer)
    valid_citations = set()
    invalid_citations = set()

    # Gom tất cả Bates từ context để đối chiếu
    all_context_bates = set()
    for chunk in context_chunks:
        all_context_bates.update(extract_bates_citations(chunk))

    # So sánh từng citation
    for cite in found_citations:
        if cite in all_context_bates:
            valid_citations.add(cite)
        else:
            invalid_citations.add(cite)

    return {
        "total_citations_found": len(found_citations),
        "valid_citations": sorted(list(valid_citations)),
        "invalid_citations": sorted(list(invalid_citations)),
        "is_all_valid": len(invalid_citations) == 0
    }

def citation_validator_middleware(llm, question: str, context_docs: List[str], model_answer: str):
    check = validate_citations(model_answer, context_docs)

    if check["is_all_valid"]:
        print("✅ All citations valid.")
        return model_answer

    invalid = ", ".join(check["invalid"]) or "none"
    valid = ", ".join(check["valid"]) or "none"

    correction_prompt = f"""
    You are ⚖️ Legal Discovery Assistant.
    The previous answer contained invalid Bates/Page citations.

    ❌ Invalid citations: {invalid}
    ✅ Valid citations: {valid}

    Task:
    - Rewrite the answer keeping only valid citations.
    - If no valid citations exist, remove them but retain the factual reasoning.
    - Never fabricate Bates or Page numbers.

    --- Question ---
    {question}

    --- Original Answer ---
    {model_answer}

    --- Base Knowledge ---
    {''.join(context_docs)}
    """
    print("⚠️ Invalid citations detected — regenerating corrected answer...")
    return llm.invoke(correction_prompt).content

def build_legal_discovery_chain(retriever, llm):
    """
    Kết hợp retriever + LLM + citation validator thành pipeline hoàn chỉnh.
    """
    def chain_func(inputs: Dict):
        question = inputs["question"]
        docs = retriever.invoke(question)
        docs_content = [d.page_content for d in docs]

        prompt = build_prompt(question, docs_content)
        raw_answer = llm.invoke(prompt).content
        final_answer = citation_validator_middleware(llm, question, docs_content, raw_answer)
        return {"answer": final_answer}

    return RunnableLambda(chain_func)
