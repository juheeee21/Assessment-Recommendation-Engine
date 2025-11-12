"""
RAG System - Recommendation Engine
Core semantic search and recommendation logic using FAISS
"""

import json
import logging
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecommendationSystem:
    """Semantic search and recommendation engine using FAISS"""
    
    def __init__(self, assessments_file: str = 'data/shl_assessments_processed.json', 
                 model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the system"""
        logger.info("Initializing Recommendation System...")
        
        # Load embedding model
        self.encoder = SentenceTransformer(model_name)
        logger.info(f"✓ Loaded embedding model: {model_name}")
        
        # Initialize FAISS index
        self.embeddings = []
        self.metadata = []
        self.index = None
        self.assessments = []
        
        # Load assessments and create index
        self.load_assessments(assessments_file)
    
    def load_assessments(self, assessments_file: str):
        """Load assessments and create FAISS index"""
        try:
            with open(assessments_file, 'r', encoding='utf-8') as f:
                self.assessments = json.load(f)
            
            logger.info(f"Loaded {len(self.assessments)} assessments")
            
            # Create embeddings for all assessments
            documents = []
            for assessment in self.assessments:
                if 'embedding_text' in assessment:
                    doc_text = assessment['embedding_text']
                else:
                    doc_text = f"{assessment['name']}. {assessment['description']}"
                documents.append(doc_text)
            
            # Generate embeddings
            logger.info("Generating embeddings...")
            self.embeddings = self.encoder.encode(documents, convert_to_numpy=True)
            
            # Create FAISS index
            try:
                import faiss
                
                # Get embedding dimension
                d = self.embeddings.shape[1]
                
                # Create simple flat index (fastest for small datasets)
                self.index = faiss.IndexFlatL2(d)
                
                # Add vectors to index
                self.index.add(self.embeddings.astype(np.float32))
                
                logger.info(f"✓ Created FAISS index with {len(self.embeddings)} vectors ({d} dimensions)")
            
            except ImportError:
                logger.warning("FAISS not available, using numpy-based search")
                self.index = None
            
            logger.info(f"✓ Indexed {len(self.assessments)} assessments")
        
        except Exception as e:
            logger.error(f"Error loading assessments: {e}")
    
    def recommend(self, query: str, n_results: int = 10) -> List[Dict]:
        """Generate recommendations for a query"""
        try:
            if not self.assessments:
                logger.error("No assessments loaded")
                return []
            
            # Encode query
            query_embedding = self.encoder.encode([query], convert_to_numpy=True)[0]
            
            # Search in FAISS
            if self.index is not None:
                try:
                    import faiss
                    
                    # FAISS search (L2 distance, so lower is better)
                    distances, indices = self.index.search(
                        np.array([query_embedding], dtype=np.float32),
                        min(n_results * 2, len(self.assessments))
                    )
                    
                    recommendations = []
                    for idx, distance in zip(indices[0], distances[0]):
                        if idx >= len(self.assessments):
                            continue
                        
                        assessment = self.assessments[int(idx)]
                        
                        # Convert L2 distance to similarity score (0-1)
                        # Lower distance = higher similarity
                        similarity = 1.0 / (1.0 + distance)
                        
                        recommendations.append({
                            'assessment_name': assessment['name'],
                            'assessment_url': assessment['url'],
                            'test_type': assessment['test_type'],
                            'relevance_score': float(similarity)
                        })
                
                except Exception as e:
                    logger.warning(f"FAISS search error, falling back to numpy: {e}")
                    recommendations = self._numpy_search(query_embedding, n_results * 2)
            else:
                # Fallback to numpy-based search
                recommendations = self._numpy_search(query_embedding, n_results * 2)
            
            # Balance assessment types
            recommendations = self.balance_recommendations(recommendations, query)
            
            # Return top N
            return recommendations[:n_results]
        
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []
    
    def _numpy_search(self, query_embedding: np.ndarray, n_results: int) -> List[Dict]:
        """Fallback search using numpy"""
        try:
            # Calculate cosine similarity with all embeddings
            similarities = []
            for i, embedding in enumerate(self.embeddings):
                # Cosine similarity
                cos_sim = np.dot(query_embedding, embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(embedding) + 1e-10
                )
                similarities.append((i, float(cos_sim)))
            
            # Sort by similarity (highest first)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Create recommendations
            recommendations = []
            for idx, similarity in similarities[:n_results]:
                assessment = self.assessments[idx]
                recommendations.append({
                    'assessment_name': assessment['name'],
                    'assessment_url': assessment['url'],
                    'test_type': assessment['test_type'],
                    'relevance_score': float((similarity + 1) / 2)  # Normalize to 0-1
                })
            
            return recommendations
        
        except Exception as e:
            logger.error(f"Error in numpy search: {e}")
            return []
    
    def balance_recommendations(self, recommendations: List[Dict], query: str) -> List[Dict]:
        """Balance K and P type assessments for multi-domain queries"""
        query_lower = query.lower()
        
        # Keywords for technical/knowledge assessments
        technical_keywords = ['developer', 'engineer', 'technical', 'programming', 
                            'coding', 'software', 'java', 'python', 'skill', 'knowledge',
                            'database', 'sql', 'api', 'framework']
        
        # Keywords for behavioral/personality assessments
        behavioral_keywords = ['collaborate', 'teamwork', 'leadership', 'communication',
                             'personality', 'behavior', 'trait', 'competency', 'culture',
                             'customer service', 'management', 'collaboration']
        
        needs_technical = any(kw in query_lower for kw in technical_keywords)
        needs_behavioral = any(kw in query_lower for kw in behavioral_keywords)
        
        if needs_technical and needs_behavioral:
            # Separate by type
            k_assessments = [r for r in recommendations if r['test_type'] == 'K']
            p_assessments = [r for r in recommendations if r['test_type'] == 'P']
            o_assessments = [r for r in recommendations if r['test_type'] == 'O']
            
            balanced = []
            max_per_type = max(1, len(recommendations) // 3)
            
            # Add equal numbers from each type
            balanced.extend(k_assessments[:max_per_type])
            balanced.extend(p_assessments[:max_per_type])
            balanced.extend(o_assessments[:max_per_type])
            
            # Sort by relevance score
            balanced.sort(key=lambda x: x['relevance_score'], reverse=True)
            return balanced
        
        return recommendations
    
    def get_stats(self) -> Dict:
        """Get system statistics"""
        k_count = sum(1 for a in self.assessments if a['test_type'] == 'K')
        p_count = sum(1 for a in self.assessments if a['test_type'] == 'P')
        o_count = sum(1 for a in self.assessments if a['test_type'] == 'O')
        
        return {
            'total_assessments': len(self.assessments),
            'knowledge_skills': k_count,
            'personality_behavior': p_count,
            'other': o_count,
            'embedding_dimension': self.embeddings.shape[1] if len(self.embeddings) > 0 else 0
        }


def main():
    """Test the recommendation system"""
    logger.info("Testing Recommendation System...")
    
    try:
        system = RecommendationSystem()
        stats = system.get_stats()
        logger.info(f"System Stats: {stats}")
        
        # Test query
        test_query = "I need a Java developer who is good at collaborating with teams"
        logger.info(f"\nTest Query: {test_query}")
        
        recommendations = system.recommend(test_query, n_results=10)
        
        if recommendations:
            logger.info(f"\nTop {len(recommendations)} Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                logger.info(f"{i}. {rec['assessment_name']} ({rec['test_type']}) - Score: {rec['relevance_score']:.3f}")
        else:
            logger.warning("No recommendations returned")
    
    except Exception as e:
        logger.error(f"Error in main: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()