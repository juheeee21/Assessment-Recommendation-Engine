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
    
    def __init__(self, assessments_file='data/shl_assessments_processed.json', model_name='all-MiniLM-L6-v2'):
        logger.info("Initializing Recommendation System...")
        self.encoder = SentenceTransformer(model_name)
        logger.info(f"✓ Loaded embedding model: {model_name}")
        self.embeddings = []
        self.index = None
        self.assessments = []
        self.load_assessments(assessments_file)
    
    def load_assessments(self, assessments_file):
        try:
            with open(assessments_file, 'r', encoding='utf-8') as f:
                self.assessments = json.load(f)
            logger.info(f"Loaded {len(self.assessments)} assessments")
            
            documents = []
            for assessment in self.assessments:
                if 'embedding_text' in assessment:
                    doc_text = assessment['embedding_text']
                else:
                    doc_text = f"{assessment['name']}. {assessment['description']}"
                documents.append(doc_text)
            
            logger.info("Generating embeddings...")
            self.embeddings = self.encoder.encode(documents, convert_to_numpy=True)
            
            try:
                import faiss
                d = self.embeddings.shape[1]
                self.index = faiss.IndexFlatL2(d)
                self.index.add(self.embeddings.astype(np.float32))
                logger.info(f"✓ Created FAISS index with {len(self.embeddings)} vectors ({d} dimensions)")
            except ImportError:
                logger.warning("FAISS not available, using numpy-based search")
                self.index = None
            
            logger.info(f"✓ Indexed {len(self.assessments)} assessments")
        
        except Exception as e:
            logger.error(f"Error loading assessments: {e}")
    
    def recommend(self, query, n_results=10):
        try:
            if not self.assessments:
                logger.error("No assessments loaded")
                return []
            
            query_embedding = self.encoder.encode([query], convert_to_numpy=True)[0]
            
            if self.index is not None:
                try:
                    import faiss
                    distances, indices = self.index.search(
                        np.array([query_embedding], dtype=np.float32),
                        min(n_results * 2, len(self.assessments))
                    )
                    
                    recommendations = []
                    for idx, distance in zip(indices[0], distances[0]):
                        if idx >= len(self.assessments):
                            continue
                        assessment = self.assessments[int(idx)]
                        similarity = 1.0 / (1.0 + distance)
                        recommendations.append({
                            'assessment_name': assessment['name'],
                            'assessment_url': assessment['url'],
                            'test_type': assessment['test_type'],
                            'relevance_score': float(similarity)
                        })
                
                except Exception as e:
                    logger.warning(f"FAISS search error, falling back: {e}")
                    recommendations = self._numpy_search(query_embedding, n_results * 2)
            else:
                recommendations = self._numpy_search(query_embedding, n_results * 2)
            
            recommendations = self.balance_recommendations(recommendations, query)
            return recommendations[:n_results]
        
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []
    
    def _numpy_search(self, query_embedding, n_results):
        try:
            similarities = []
            for i, embedding in enumerate(self.embeddings):
                cos_sim = np.dot(query_embedding, embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(embedding) + 1e-10
                )
                similarities.append((i, float(cos_sim)))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            recommendations = []
            for idx, similarity in similarities[:n_results]:
                assessment = self.assessments[idx]
                recommendations.append({
                    'assessment_name': assessment['name'],
                    'assessment_url': assessment['url'],
                    'test_type': assessment['test_type'],
                    'relevance_score': float((similarity + 1) / 2)
                })
            
            return recommendations
        
        except Exception as e:
            logger.error(f"Error in numpy search: {e}")
            return []
    
    def balance_recommendations(self, recommendations, query):
        query_lower = query.lower()
        technical_keywords = ['developer', 'engineer', 'technical', 'programming', 
                            'coding', 'software', 'java', 'python', 'skill', 'knowledge']
        behavioral_keywords = ['collaborate', 'teamwork', 'leadership', 'communication',
                             'personality', 'behavior', 'trait', 'competency', 'culture']
        
        needs_technical = any(kw in query_lower for kw in technical_keywords)
        needs_behavioral = any(kw in query_lower for kw in behavioral_keywords)
        
        if needs_technical and needs_behavioral:
            k_assessments = [r for r in recommendations if r['test_type'] == 'K']
            p_assessments = [r for r in recommendations if r['test_type'] == 'P']
            o_assessments = [r for r in recommendations if r['test_type'] == 'O']
            
            balanced = []
            max_per_type = max(1, len(recommendations) // 3)
            
            balanced.extend(k_assessments[:max_per_type])
            balanced.extend(p_assessments[:max_per_type])
            balanced.extend(o_assessments[:max_per_type])
            
            balanced.sort(key=lambda x: x['relevance_score'], reverse=True)
            return balanced
        
        return recommendations
    
    def get_stats(self):
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
    logger.info("Testing Recommendation System...")
    
    try:
        system = RecommendationSystem()
        stats = system.get_stats()
        logger.info(f"System Stats: {stats}")
        
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
