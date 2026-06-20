# LinearRag-RAG Engine

Multi-hop context retrieval via linear sparse graph walking.

## Setup & Installation

1. **Install dependencies**:
   ```bash
   pip install -e .
   python -m spacy download en_core_web_sm
   ```

2. **Setup database**:
   ```bash
   psql -f scripts/schema.sql <database_url>
   ```

3. **Copy and configure env**:
   ```bash
   cp .env.example .env
   # Edit .env with your database URL and API keys
   ```

4. **Run server**:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload
   ```

## Running with Docker

Build and run the Docker image:
```bash
docker build -t linear-rag .
docker run -p 8002:8002 --env-file .env linear-rag
```
