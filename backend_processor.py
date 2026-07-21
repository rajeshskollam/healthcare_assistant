import os
import asyncio
import sys

# Suppress Windows ProactorEventLoop connection reset warnings
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama

FAISS_INDEX_PATH = "faiss_index"
DOCUMENT_PATH = r"documents\diabetes-in-pregnancy-management-from-preconception-to-the-postnatal-period.pdf"

def get_embeddings():
    return OllamaEmbeddings(model="nomic-embed-text")
 
def get_index():
    if os.path.exists(FAISS_INDEX_PATH):
        print("✅ Loading existing FAISS index from disk...")
        vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH,
            get_embeddings(),
            allow_dangerous_deserialization=True
        )
        return vectorstore

    data_load = PyPDFLoader(DOCUMENT_PATH)
    pages = data_load.load()
    text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=1500,
            chunk_overlap=200
        )
    docs = text_splitter.split_documents(pages)
    data_embeddings = get_embeddings()

    vectorstore = FAISS.from_documents(docs, data_embeddings)
    vectorstore.save_local(FAISS_INDEX_PATH)
    return vectorstore

def hr_llm():
    llm = ChatOllama(
        model="llama3.1:8b",
        temperature=0.0,
        num_ctx=4096,
        top_p=0.9,
    )
    return llm

def _build_prompt(index, question):
    """Retrieve context and build the prompt (shared by streaming and non-streaming)."""
    search_queries = [
        question,
        "gestational diabetes management metformin insulin diet blood glucose targets monitoring",
        "self-monitoring blood glucose target fasting pre-meal 1 hour post-meal pregnant women",
    ]

    seen_contents = set()
    all_docs = []
    for query in search_queries:
        docs = index.similarity_search(query, k=4)
        for doc in docs:
            if doc.page_content not in seen_contents:
                seen_contents.add(doc.page_content)
                all_docs.append(doc)

    all_docs = all_docs[:8]
    context = "\n\n---\n\n".join([doc.page_content for doc in all_docs])

    prompt = f"""You are a clinical decision-support assistant strictly based on the NICE guideline NG3: "Diabetes in pregnancy: management from preconception to the postnatal period."

STRICT RULES - YOU MUST FOLLOW THESE:
1. ONLY use information from the Guideline Context provided below. Do NOT add any external knowledge.
2. ONLY use clinical details explicitly stated in the query. Do NOT assume or infer patient history, BMI, age, or any data not provided.
3. Do NOT use phrases like "likely", "probably", or "it's possible" when making a diagnosis. If the values meet diagnostic criteria per the guideline, state the diagnosis definitively.
4. Do NOT reference ADA, WHO, or any other guideline. ONLY NICE NG3.
5. When quoting blood glucose targets or thresholds, use the EXACT values from the guideline context. Do not paraphrase or approximate.
6. Metformin CAN be used in pregnancy per this guideline — as an adjunct or alternative to insulin. Do NOT say it is contraindicated unless the guideline says so.
7. Be specific with time values (e.g., "2-hour value" not "1-2 hour").
8. If a specific detail is truly not in the guideline context, say "Not specified in the available guideline sections" for ONLY that detail — but still answer all other sections fully.

RESPONSE FORMAT:
- Diagnosis: (based strictly on OGTT values vs guideline thresholds)
- Dietary & Lifestyle Advice: (from guideline)
- Blood Glucose Targets: (exact NICE targets with units)
- Pharmacological Management: (metformin and/or insulin as per guideline)
- Monitoring Schedule: (from guideline)
- Referral/Follow-up: (from guideline)

Guideline Context:
{context}

Clinical Query: {question}

Clinical Recommendation:"""
    return prompt

def hr_rag_response(index, question):
    """Non-streaming response (kept for compatibility)."""
    rag_llm = hr_llm()
    prompt = _build_prompt(index, question)
    response = rag_llm.invoke(prompt)
    return response.content

def hr_rag_response_stream(index, question):
    """Streaming response - yields chunks of text as they are generated."""
    rag_llm = hr_llm()
    prompt = _build_prompt(index, question)
    for chunk in rag_llm.stream(prompt):
        yield chunk.content