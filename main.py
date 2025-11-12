"""
FastAPI Application - REST API Server
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import logging

try:
    from recommendation_system import RecommendationSystem
except ImportError:
    import sys
    sys.path.insert(0, '.')
    from recommendation_system import RecommendationSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SHL Assessment Recommendation API",
    description="Intelligent recommendation system for SHL assessments",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

recommendation_system = None


def get_system() -> RecommendationSystem:
    """Get or initialize the recommendation system"""
    global recommendation_system
    if recommendation_system is None:
        logger.info("Initializing recommendation system...")
        recommendation_system = RecommendationSystem()
    return recommendation_system


class QueryRequest(BaseModel):
    query: str
    max_results: Optional[int] = 10


class RecommendationItem(BaseModel):
    assessment_name: str
    assessment_url: str
    relevance_score: float
    test_type: Optional[str] = None


class RecommendationResponse(BaseModel):
    recommendations: List[RecommendationItem]
    total_results: int
    query: str


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.get("/stats")
async def get_system_stats():
    """Get system statistics"""
    try:
        system = get_system()
        stats = system.get_stats()
        return {
            "status": "ok",
            "total_assessments": stats['total_assessments'],
            "knowledge_skills_count": stats['knowledge_skills'],
            "personality_behavior_count": stats['personality_behavior'],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommend", response_model=RecommendationResponse)
async def recommend_assessments(request: QueryRequest):
    """Get assessment recommendations"""
    try:
        if not request.query or len(request.query.strip()) == 0:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        if request.max_results < 5 or request.max_results > 10:
            request.max_results = 10
        
        system = get_system()
        recommendations = system.recommend(request.query, n_results=request.max_results)
        
        if len(recommendations) < 5:
            raise HTTPException(
                status_code=500,
                detail="Could not generate enough recommendations"
            )
        
        formatted_recs = [
            RecommendationItem(
                assessment_name=rec['assessment_name'],
                assessment_url=rec['assessment_url'],
                relevance_score=rec['relevance_score'],
                test_type=rec.get('test_type')
            )
            for rec in recommendations
        ]
        
        return RecommendationResponse(
            recommendations=formatted_recs,
            total_results=len(formatted_recs),
            query=request.query
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.on_event("startup")
async def startup_event():
    logger.info("Starting up API server...")
    try:
        system = get_system()
        logger.info("✓ Recommendation system initialized")
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
