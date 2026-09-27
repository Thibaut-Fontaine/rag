from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from embeddings import embeddings

QDRANT_PATH = "./qdrant_db"
COLLECTION_NAME = "documents"


def main():
    query = input("Query: ")

    client = QdrantClient(path=QDRANT_PATH)

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )

    results = vector_store.similarity_search(query, k=5)

    for i, document in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(f"Source: {document.metadata['source']}")
        print(document.page_content)


if __name__ == "__main__":
    main()