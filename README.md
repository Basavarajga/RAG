# Finance RAG System

This project implements a Retrieval-Augmented Generation (RAG) pipeline
for answering financial questions using a knowledge corpus.

## Features

- Wikipedia finance corpus builder
- Sentence-transformer embeddings
- FAISS vector database
- Hybrid retrieval (dense + BM25)
- Reranking with cross-encoder
- Evaluation framework

## Architecture
Query → Embedding → Hybrid Retrieval → Reranking → Context → Answer

## How to Run
1. Install dependencies
pip install -r requirements.txt

2. Build corpus
python src/build_corpus.py

3. Build embeddings
python src/embeddings.py

4. Run retrieval
python src/rag_pipeline.py
