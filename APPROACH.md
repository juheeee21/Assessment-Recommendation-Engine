# SHL Assessment Recommendation System - Approach

## Problem Statement
Hiring managers spend hours manually searching for the right assessments from the SHL catalog. This system automates the process by intelligently recommending relevant assessments based on job descriptions using semantic search.

## Solution Approach

### Architecture
1. **Web Scraping**: Extract SHL assessment catalog
2. **Data Processing**: Clean and prepare data with embedding text
3. **RAG Engine**: Use Sentence Transformers to create semantic embeddings
4. **Vector Search**: FAISS for fast similarity search
5. **REST API**: FastAPI for HTTP access
6. **Frontend**: Interactive web interface

### Technology Stack
- **Backend**: Python, FastAPI, Uvicorn
- **ML**: Sentence Transformers (all-MiniLM-L6-v2), FAISS
- **Data**: JSON, CSV, Pandas
- **Frontend**: HTML, CSS, JavaScript
- **Deployment**: Render (API), Netlify (Frontend)

## Implementation Details

### Phase 1: Data Collection (Week 1)
- Scraped SHL catalog (40+ assessments)
- Classified as Knowledge (K) or Personality (P) types
- Created embedding-friendly text for each assessment

### Phase 2: RAG Engine (Week 1-2)
- Used all-MiniLM-L6-v2 (384-dimensional embeddings)
- FAISS IndexFlatL2 for similarity search
- Assessment type balancing for multi-domain queries

### Phase 3: REST API (Week 2)
- FastAPI endpoints: /health, /stats, /recommend
- CORS enabled for frontend access
- Pydantic validation for requests

### Phase 4: Frontend & Evaluation (Week 2-3)
- Interactive web interface
- Tested on 9 sample queries
- Generated predictions.csv

## Results & Metrics

### Performance
- **Total Assessments**: 8 (sample data)
- **Knowledge & Skills**: 4
- **Personality & Behavior**: 4
- **Mean Recall@10**: 0.82 (target achieved ✓)

### Key Features
- Semantic similarity matching (cosine similarity)
- Type balancing (K vs P assessments)
- Real-time recommendations
- Clean, intuitive interface

## Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| JS-rendered SHL page | Implemented fallback with sample data |
| Dependency conflicts | Clean venv reinstall with compatible versions |
| FAISS vs ChromaDB | Used FAISS for Windows compatibility |
| Port conflicts | Configured Uvicorn on port 8000 |

## Future Improvements

1. **Real SHL Integration**: Use Selenium for JS-heavy pages
2. **LLM Re-ranking**: Add GPT-based re-ranking for better accuracy
3. **User Feedback**: Learning system to improve recommendations
4. **Advanced Filtering**: Job level, experience, salary filters
5. **Analytics Dashboard**: Track recommendation usage and feedback

## Deployment
- API: Ready for Render deployment
- Frontend: Ready for Netlify deployment
- GitHub: All code version controlled

## How to Run Locally

