# AI_Assistant is 
## RAG System Implementation (In Progress)
A Retrieval-Augmented Generation (RAG) system implementation that processes Wikipedia data, transforms it into vector embeddings, and stores them in Qdrant vector database for efficient retrieval.

## Project Status
### Completed Features:

Data extraction from JSON

Data normalization and validation

Vector embedding generation

Qdrant vector database integration

Comprehensive test coverage

## 🚧 In Progress:

API interface development

Ollama LLM

Full RAG pipeline completion

## Data Analysis
The system processes JSON data with the following structure (array of dictionaries):

### Example entry:
{
"uid": "unique_id_1", (Unique identifier for the data entry)
"ru_wiki_pageid": 12345, (Wikipedia page ID (may repeat))
"text": "Paragraph text..." (Text content from Wikipedia)
}

### Key characteristics:

uid: Unique identifier for each entry

ru_wiki_pageid: Wikipedia page ID (can be used to visit the page at https://ru.wikipedia.org/?curid={ru_wiki_pageid})

text: Paragraph text extracted from the associated Wikipedia page

Sample data source: https://github.com/vladislavneon/RuBQ/blob/master/RuBQ_2.0/RuBQ_2.0_paragraphs.json

## Key Features
### Data Processing Pipeline:

Batch processing of raw data

Data normalization and validation

Duplicate handling (both in DB and within batches)

Vector Database Integration:

Qdrant database initialization

Collection creation/deletion

Record insertion and validation

Embedding Generation:

Configurable embedding models

Text-to-vector transformation

## Technologies Used
Python 3.11

Docker + Docker Compose

Qdrant (vector database)

Sentence Transformers (embeddings)

Dependencies
Main dependencies:
qdrant-client==1.15.0
sentence-transformers==5.0.0
transformers==4.53.2
torch==2.7.1

Additional:
numpy==2.3.1
tqdm==4.67.1
pydantic==2.11.7
pyyaml==6.0.2
grpcio==1.73.1
protobuf==6.31.1
loguru

Testing:
pytest==8.4.1
pytest-cov==6.2.1
coverage==7.9.2

##Getting Started
### Configuration:

Modify configs/embedding_docker.yaml for your needs

### Running the System:
docker-compose up

### Loading Data:

The main pipeline is executed through scripts/load_vector_db.py

## Main Script Overview
### `DataBatchProcessor`
- Processes data batches into vector DB records
- Handles text normalization and embedding generation
- Extracts additional record fields from input data

### `VectorDBLoader` (Main Class)
- Orchestrates the complete loading pipeline:
  1. Configuration loading
  2. Database initialization (supports Qdrant)
  3. Data loading and validation
  4. Batch processing with duplicate handling
- Works with any `BaseRecord` implementation

### Key Features:
- Configurable batch processing
- Duplicate detection (both in-batch and in-DB)
- Docker/Test mode support
- Detailed logging

## Future Improvements
API interface development

Enhanced documentation

Full RAG pipeline completion