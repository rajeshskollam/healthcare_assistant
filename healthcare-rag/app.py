import os
from typing import TypedDict, List

import psycopg
from pgvector.psycopg import register_vector
from pgvector import Vector
from dotenv import load_dotenv

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langgraph.graph import StateGraph, START, END

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:1b")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

llm = ChatOllama(model=CHAT_MODEL, temperature=0)
embeddings = OllamaEmbeddings(model=EMBED_MODEL)

class RAGState(TypedDict):
    patient_id: str
    question: str
    context: str
    sources: List[str]
    found: bool
    answer: str

def retrieve_documents(state: RAGState):
    query_vector = Vector(embeddings.embed_query(state["question"]))

    with psycopg.connect(DATABASE_URL) as conn:
        register_vector(conn)

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT source, content, embedding <=> %s AS distance
                FROM document_chunks
                WHERE patient_id = %s
                ORDER BY embedding <=> %s
                LIMIT 3
                """,
                (query_vector, state["patient_id"], query_vector),
            )
            rows = cur.fetchall()

        context = "\n\n".join(
            f"Source: {source}\n{content}"
            for source, content, distance in rows
        )

        sources = list(
            dict.fromkeys(source for source, _, _ in rows)
        )

        print(f"Retrieved {len(rows)} documents")

        return {
            "context": context,
            "sources": sources,
            "found": len(rows) > 0,
        }

def generate_answer(state: RAGState):
    prompt = ChatPromptTemplate.from_template(
        """
        You are a healthcare documentation assistant.
        Patient ID: {patient_id}

        Patient history:
        {context}

        User question:
        {question}

        Instructions:
        - Use only the retrieved patient records.
        - Summarize documented medical conditions.
        - Include medications when documented.
        - Include relevant visit dates and observations.
        - Do not invent missing information.
        - Do not provide diagnosis or treatment advice.
        - Mention when information is unavailable.
        """
    )

    chain = prompt | llm | StrOutputParser()

    response = chain.invoke(
        {
            "patient_id": state["patient_id"],
            "context": state["context"],
            "question": state["question"],
        }
    )

    return {"answer": response}

def no_context(state: RAGState):
    return {
        "answer": (
            "No relevant records were found for patient "
            + state["patient_id"]
            + ". Please verify the patient ID or records."
        )
    }

def route_after_retrieval(state: RAGState):
    return "generate" if state["found"] else "no_context"

workflow = StateGraph(RAGState)

workflow.add_node("retrieve", retrieve_documents)
workflow.add_node("generate", generate_answer)
workflow.add_node("no_context", no_context)

workflow.add_edge(START, "retrieve")
workflow.add_conditional_edges(
    "retrieve",
    route_after_retrieval,
    {
        "generate": "generate",
        "no_context": "no_context"
    }
)
workflow.add_edge("generate", END)
workflow.add_edge("no_context", END)

app = workflow.compile()

if __name__ == "__main__":
    patient_id = input("Enter patient ID: ").strip()
    question = input("Enter your question: ").strip()

    result = app.invoke(
        {
            "patient_id": patient_id,
            "question": question,
            "context": "",
            "sources": [],
            "found": False,
            "answer": "",
        }
    )

    print("\n==============================")
    print("PATIENT HISTORY SNAPSHOT")
    print("==============================")
    print(result["answer"])

    print("\nSOURCE DOCUMENTS:")
    for source in result["sources"]:
        print("-", source)