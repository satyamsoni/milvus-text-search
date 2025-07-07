import os
import sys
import time
import hashlib
import struct
import pdfplumber
from tqdm import tqdm
from docx import Document
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient, DataType

class DocumentIngest:

	collection = "document_collection"
	milvus_host = "http://127.0.0.1:19530"
	vector_dimension = 384  # Typical for Sentence-BERT models like all-MiniLM

	def __init__(self, doc_folder):
		print("Starting Document Ingestion into Milvus")
		self.client = MilvusClient(DocumentIngest.milvus_host)
		self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
		self.doc_folder = doc_folder
		self.prep_collection()
		self.ingest()

	def prep_collection(self):
		# To Reset 
		#self.client.drop_collection(collection_name=DocumentIngest.collection)
		if not self.client.has_collection(DocumentIngest.collection):
			print(f"Creating collection '{DocumentIngest.collection}'...")

			schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
			schema.add_field(
				"id", 
				DataType.VARCHAR, 
				is_primary=True, 
				max_length=36
			)
			schema.add_field(
				"filename", 
				DataType.VARCHAR, 
				max_length=256
			)
			schema.add_field(
				"content", 
				DataType.VARCHAR, 
				max_length=10000
			)
			schema.add_field(
				"embedding", 
				DataType.FLOAT_VECTOR, 
				dim=DocumentIngest.vector_dimension
			)
			schema.add_field(
				"timestamp", 
				DataType.INT64
			)

			self.client.create_collection(
				collection_name=DocumentIngest.collection, 
				schema=schema
			)

			index_params = self.client.prepare_index_params()
			index_params.add_index(
				field_name="embedding",
				index_type="IVF_FLAT",
				index_name="emb_index",
				metric_type="L2",
				params={"nlist": 128}
			)
			self.client.create_index(
				collection_name=DocumentIngest.collection, 
				index_params=index_params
			)

			print("Collection created.")
		else:
			print(f"Collection '{DocumentIngest.collection}' already exists.")

	def ingest(self):
		self.client.load_collection(collection_name=DocumentIngest.collection)

		all_files = [f for f in os.listdir(self.doc_folder) if f.lower().endswith((".pdf", ".docx", ".txt"))]
		preIngested=0
		ingested=0
		for filename in tqdm(all_files, desc="Ingesting Documents", unit="file"):
			file_path = os.path.join(self.doc_folder, filename)

			text = self.extract_text(file_path)
			if not text:
				continue

			vector = self.model.encode(text).tolist()
			record_id = self.vector_md5(vector)

			existing = self.client.query(
				collection_name=DocumentIngest.collection,
				filter=f'id == "{record_id}"',
				output_fields=["id"]
			)
			if existing:
				preIngested+=1
				continue

			self.client.insert(
				collection_name=DocumentIngest.collection,
				data=[{
					"id": record_id,
					"filename": filename,
					"content": text[:10000],
					"embedding": vector,
					"timestamp": int(time.time())
				}]
			)
			ingested+=1

		print(f"Ingested {ingested}, found duplicate {preIngested}")

	def extract_text(self, file_path):
		try:
			if file_path.lower().endswith(".pdf"):
				with pdfplumber.open(file_path) as pdf:
					return " ".join(page.extract_text() or "" for page in pdf.pages)
			elif file_path.lower().endswith(".docx"):
				doc = Document(file_path)
				return " ".join(para.text for para in doc.paragraphs)
			elif file_path.lower().endswith(".txt"):
				with open(file_path, encoding='utf-8') as f:
					return f.read()
		except Exception as e:
			print(f"Failed to extract text from {file_path}: {e}")
		return None

	def vector_md5(self, vector):
		vector_bytes = bytearray()
		for num in vector:
			vector_bytes.extend(struct.pack("f", float(num)))
		return hashlib.md5(vector_bytes).hexdigest()

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print("Usage: python DocumentIngest.py <documents_path>")
	else:
		app = DocumentIngest(sys.argv[1])