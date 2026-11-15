import os
from pathlib import Path

import psycopg
from dotenv import  load_dotenv
from pgvector.psycopg import register_vector

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

embeddings = OllamaEmbeddings(model="nomic-embed-text")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

def ingest_documents():
    documents = []
    for file in Path("data").glob("*.txt"):
        content = file.read_text(encoding="utf-8")
        patient_id = content.splitlines()[0].split(":")[1].strip()

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "patient_id": patient_id,
                    "source": file.name
                }
            )
        )

    chunks = splitter.split_documents(documents)
    print("Total chunks:", len(chunks))

    vectors = embeddings.embed_documents(
        [chunk.page_content for chunk in chunks]
    )

    with psycopg.connect(DATABASE_URL) as conn:
        register_vector(conn)

        with conn.cursor() as cur:
            for chunk, vector in zip(chunks, vectors):
                patient_id = chunk.metadata["patient_id"]
                source = chunk.metadata["source"]

                cur.execute(
                    """
                    INSERT INTO patients (patient_id)
                    VALUES (%s)
                    ON CONFLICT (patient_id) DO NOTHING
                    """,
                    (patient_id,),
                )

                cur.execute(
                    """
                    INSERT INTO document_chunks
                        (patient_id, source, content, embedding)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (patient_id, source, chunk.page_content, vector),
                )

                conn.commit()

            print("Documents successfully stored in PostgreSQL!")

if __name__ == "__main__":
    ingest_documents()








