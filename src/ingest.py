from pathlib import Path
import hashlib
import uuid
import argparse
import time

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from embeddings import embeddings


SOURCE_DIR = Path("/home/thibaut/Desktop/Jdr")

QDRANT_PATH = "./qdrant_db"
COLLECTION_NAME = "documents"

MAX_FILE_SIZE = 500 * 1024

ALLOWED_EXTENSIONS = {
    ".txt",
    ".md",
    ".html",
    ".htm",
    ".pdf",
    ".csv",
    ".json",
    ".xml",
}

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

BATCH_SIZE = 8


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)


def get_file_id(path: Path):
    return hashlib.sha256(
        str(path.resolve()).encode("utf-8")
    ).hexdigest()


def load_files():

    for path in SOURCE_DIR.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() not in ALLOWED_EXTENSIONS:
            continue

        if path.stat().st_size >= MAX_FILE_SIZE:
            continue

        extension = path.suffix.lower()
        file_id = get_file_id(path)

        try:

            if extension == ".pdf":

                loader = PyMuPDF4LLMLoader(
                    file_path=str(path),
                    mode="page",
                )

                documents = loader.load()

                for document in documents:

                    document.metadata.update({
                        "source": str(path),
                        "filename": path.name,
                        "extension": extension,
                        "size": path.stat().st_size,
                        "file_id": file_id,
                    })

                    yield document

            else:

                text = path.read_text(
                    encoding="utf-8",
                    errors="ignore",
                )

                if not text.strip():
                    continue

                yield Document(
                    page_content=text,
                    metadata={
                        "source": str(path),
                        "filename": path.name,
                        "extension": extension,
                        "size": path.stat().st_size,
                        "file_id": file_id,
                    },
                )

        except Exception as e:

            print(
                f"  ! Error reading {path}: {e}"
            )


def make_chunk_id(document, index):

    raw = (
        f"{document.metadata['file_id']}:"
        f"{index}:"
        f"{document.page_content}"
    )

    digest = hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()

    return str(uuid.UUID(digest[:32]))


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    start_time = time.time()

    client = QdrantClient(
        path=QDRANT_PATH
    )

    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=1024,
                distance=Distance.COSINE,
            ),
        )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )

    total_files = 0
    total_chunks = 0
    batch_documents = []
    batch_ids = []

    for document in load_files():

        if args.limit is not None and time.time() - start_time >= args.limit:
            print(f"\nTime limit reached ({args.limit}s).")
            break

        total_files += 1

        chunks = text_splitter.split_documents(
            [document]
        )

        print(
            f"[{total_files}] Loading: "
            f"{document.metadata['source']}"
        )

        print(
            f"  → {len(chunks)} chunks"
        )

        for index, chunk in enumerate(chunks):

            chunk_id = make_chunk_id(
                chunk,
                index,
            )

            chunk.metadata["chunk_id"] = chunk_id

            batch_documents.append(chunk)
            batch_ids.append(chunk_id)

            if len(batch_documents) >= BATCH_SIZE:

                vector_store.add_documents(
                    documents=batch_documents,
                    ids=batch_ids,
                )

                total_chunks += len(batch_documents)

                print(
                    f"  → stored | "
                    f"total chunks: {total_chunks}"
                )

                batch_documents = []
                batch_ids = []

    if batch_documents:

        vector_store.add_documents(
            documents=batch_documents,
            ids=batch_ids,
        )

        total_chunks += len(batch_documents)

        print(
            f"  → stored | "
            f"total chunks: {total_chunks}"
        )

    print()
    print("Done.")
    print(f"Files processed: {total_files}")
    print(f"Chunks processed: {total_chunks}")


if __name__ == "__main__":
    main()