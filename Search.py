from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
import sys

class Search:

    collection = "document_collection"
    milvus_host = "http://127.0.0.1:19530"
    vector_field = "embedding"

    def __init__(self, text):
        self.client = MilvusClient(Search.milvus_host)
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.find(text)

    def find(self, text):
        print(f"Searching for: {text}")

        # Generate text embedding
        vector = self.model.encode(text).tolist()

        # Perform vector search
        results = self.client.search(
            collection_name=Search.collection,
            data=[vector],
            limit=5,
            output_fields=["id", "filename", "content"]
        )

        # Display results
        print("\nTop Matches:\n")
        for hit in results[0]:
            print(f"ID: {hit['id']}")
            print(f"Filename: {hit['filename']}")
            print(f"Content Snippet: {hit['content'][:200]}...")
            print(f"Score: {hit.score:.4f}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python Search.py <text query>")
    else:
        app = Search(sys.argv[1])