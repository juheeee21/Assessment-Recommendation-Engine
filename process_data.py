"""
Data Processing & Cleaning Pipeline
"""

import json
import pandas as pd
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """Process and clean assessment data"""
    
    def __init__(self, json_file: str = 'data/shl_assessments.json'):
        self.json_file = json_file
        self.assessments = []
    
    def load_data(self) -> List[Dict]:
        """Load assessments from JSON"""
        try:
            with open(self.json_file, 'r', encoding='utf-8') as f:
                self.assessments = json.load(f)
            logger.info(f"✓ Loaded {len(self.assessments)} assessments")
            return self.assessments
        except Exception as e:
            logger.error(f"Error loading: {e}")
            return []
    
    def clean_data(self):
        """Clean and standardize data"""
        logger.info("Cleaning data...")
        
        for assessment in self.assessments:
            assessment['name'] = assessment.get('name', '').strip()
            
            if not assessment['url'].startswith('http'):
                assessment['url'] = f"https://shl.com{assessment['url']}"
            
            assessment['description'] = assessment.get('description', '').strip()
            
            if assessment.get('test_type') not in ['K', 'P', 'O']:
                assessment['test_type'] = 'O'
            
            if isinstance(assessment.get('skills'), str):
                assessment['skills'] = [s.strip() for s in assessment['skills'].split(',')]
        
        seen_urls = set()
        cleaned = []
        for a in self.assessments:
            if a['url'] not in seen_urls:
                cleaned.append(a)
                seen_urls.add(a['url'])
        
        self.assessments = cleaned
        logger.info(f"✓ Cleaned: {len(self.assessments)} unique assessments")
    
    def create_embedding_text(self):
        """Create rich text for embedding"""
        logger.info("Creating embedding text...")
        
        for assessment in self.assessments:
            embedding_text = f"{assessment['name']}. {assessment['description']}. Skills: {', '.join(assessment['skills'])}"
            assessment['embedding_text'] = embedding_text
        
        logger.info("✓ Embedding text created")
    
    def save_processed(self, output_file: str = 'data/shl_assessments_processed.json'):
        """Save processed data"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.assessments, f, indent=2, ensure_ascii=False)
            logger.info(f"✓ Saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving: {e}")
    
    def get_stats(self) -> Dict:
        """Get data statistics"""
        stats = {
            'total_assessments': len(self.assessments),
            'knowledge_skills': sum(1 for a in self.assessments if a['test_type'] == 'K'),
            'personality_behavior': sum(1 for a in self.assessments if a['test_type'] == 'P'),
            'other': sum(1 for a in self.assessments if a['test_type'] == 'O'),
        }
        return stats
    
    def process(self):
        """Run complete processing pipeline"""
        logger.info("Starting data processing...")
        self.load_data()
        self.clean_data()
        self.create_embedding_text()
        self.save_processed()
        
        stats = self.get_stats()
        logger.info(f"\n--- STATISTICS ---")
        for key, value in stats.items():
            logger.info(f"{key}: {value}")
        
        return self.assessments


if __name__ == "__main__":
    processor = DataProcessor()
    processor.process()
