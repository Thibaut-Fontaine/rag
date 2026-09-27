from langchain_huggingface import HuggingFaceEmbeddings

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

if __name__ == "__main__":
    vector = embeddings.embed_query("This is a test.")
    print(f"Embedding dimension: {len(vector)}")