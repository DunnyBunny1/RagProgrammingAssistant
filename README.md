# RagProgrammingAssistant
AI programming assistant that uses Retrieval-Augmented Generation (RAG) to answer technical programming questions, using LLM-enhanced responses from [Stack Overflow](https://stackoverflow.com/)	

## Quick Start

### Running with Docker (Recommended)
```bash
# 1. Clone the repository
git clone https://github.com/DunnyBunny1/RagProgrammingAssistant
cd ragoverflow

# 2. Set up environment variables
cp .env.template .env
# Edit .env with your Pinecone credentials

# 3. Run the application
docker-compose up --build
```

The application will be available at:
- **Frontend:** http://localhost:8501
- **Backend API:** http://localhost:8000

### Local Development Setup

For local development without Docker: (using Python 3.12)
```bash
# Install dependencies (from project root)
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt

# Run backend
cd backend
python server.py 

# Run frontend (separate terminal)
cd frontend
python app.py
```

> **Note:** All `pip install` commands must be run from the project root directory, as dependencies reference the `shared` package with a relative path.

#### Data Pipeline & Notebooks

To explore data processing or run Jupyter notebooks:
```bash
# Install data pipeline dependencies
pip install -r data_pipeline/requirements.txt

# Start JupyterLab
jupyter lab

# Run the scripts to process data  
python data_pipeline/scripts/01_load_xml_dump_data_to_duck_db.py
python data_pipeline/scripts/02_generate_embeddings_to_parquet.py
python data_pipeline/scripts/03_upload_embeddings_to_cloud_vectordb.py
```

Notebooks are located in `data_pipeline/notebooks/`.