# Milvus Vector Text Document Search
Vector Database Setup, Text Document ingestion and Search - Small Sample

## Purpose : 
A small sample repo to learn 
1. How to setup Milvus standalone in local machine
2. Create collection and ingest images as embeddings in Vector Database using pretrained AI models.
3. Search near match by meaning/context from documents.

## How to setup Milvus
1. Clone the repo "git clone git@github.com:satyamsoni/vector_text_search.git"
2. Create virtual environment "python3 -m venv venv"
3. Activate the virtual environment "source venv/bin/activate"
4. Install python lib pymilvus, transformers, torch, numpy, pdfplumber, python-docx, tqdm by "pip install <package_name>"
5. Install docker if not not already installed ref : https://docs.docker.com/engine/install/ubuntu
6. run milvus standalone "docker-compose up -d"
7. You can verify if it is running by "docker ps"


## How to Ingest Photos
run "python DocumentIngest.py ./docs"

## How to Search Photos
run "python Search.py <text>"