"""
Evaluation Script - Calculate Recall@10 and Generate Predictions
"""

import json
import pandas as pd
import logging
from typing import List, Dict, Set
from recommendation_system import RecommendationSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_recall_at_k(predicted_urls: List[str], relevant_urls: Set[str], k: int = 10) -> float:
    """Calculate Recall@K metric"""
    predicted_k = set(predicted_urls[:k])
    
    if len(relevant_urls) == 0:
        return 0.0
    
    intersection = predicted_k.intersection(relevant_urls)
    recall = len(intersection) / len(relevant_urls)
    return recall


def evaluate_on_training_set(train_file: str = 'data/train_labeled.json'):
    """Evaluate system on training set"""
    logger.info("Evaluating on training set...")
    
    try:
        with open(train_file, 'r') as f:
            train_data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Training file not found: {train_file}")
        logger.info("Create data/train_labeled.json with training data")
        return
    
    system = RecommendationSystem()
    recalls = []
    
    logger.info(f"\nEvaluating {len(train_data)} queries...\n")
    
    for i, item in enumerate(train_data, 1):
        query = item['query']
        relevant_urls = set(item.get('relevant_assessments', []))
        
        recommendations = system.recommend(query, n_results=10)
        predicted_urls = [rec['assessment_url'] for rec in recommendations]
        
        recall = calculate_recall_at_k(predicted_urls, relevant_urls, k=10)
        recalls.append(recall)
        
        logger.info(f"Query {i}: {query[:50]}...")
        logger.info(f"  Relevant: {len(relevant_urls)}, Found: {len(set(predicted_urls).intersection(relevant_urls))}")
        logger.info(f"  Recall@10: {recall:.3f}\n")
    
    mean_recall = sum(recalls) / len(recalls) if recalls else 0
    
    logger.info("=" * 60)
    logger.info(f"Mean Recall@10: {mean_recall:.3f}")
    logger.info("=" * 60)
    
    return {'mean_recall': mean_recall, 'recalls': recalls}


def generate_test_predictions(test_file: str = 'data/test_unlabeled.json', 
                             output_file: str = 'predictions.csv'):
    """Generate predictions for test set"""
    logger.info("Generating test predictions...")
    
    try:
        with open(test_file, 'r') as f:
            test_data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Test file not found: {test_file}")
        logger.info("Create data/test_unlabeled.json with test queries")
        return
    
    system = RecommendationSystem()
    predictions = []
    
    for i, item in enumerate(test_data, 1):
        query = item['query']
        logger.info(f"Processing query {i}/{len(test_data)}: {query[:50]}...")
        
        recommendations = system.recommend(query, n_results=10)
        
        for rec in recommendations:
            predictions.append({
                'Query': query,
                'Assessment_url': rec['assessment_url']
            })
    
    df = pd.DataFrame(predictions)
    df.to_csv(output_file, index=False)
    
    logger.info(f"\n✓ Saved {len(predictions)} predictions to {output_file}")
    logger.info(f"  Format: Query, Assessment_url")
    logger.info(f"  Rows: {len(predictions)}")


def create_sample_training_data():
    """Create sample training data"""
    sample_data = [
        {
            "query": "Java developer with communication skills",
            "relevant_assessments": [
                "https://www.shl.com/solutions/products/java-programming-test/",
                "https://www.shl.com/solutions/products/communication-assessment/"
            ]
        },
        {
            "query": "Project manager with leadership abilities",
            "relevant_assessments": [
                "https://www.shl.com/solutions/products/leadership-assessment/",
                "https://www.shl.com/solutions/products/project-management/"
            ]
        }
    ]
    
    import os
    os.makedirs('data', exist_ok=True)
    
    with open('data/train_labeled.json', 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    logger.info("✓ Created sample training data")


def create_sample_test_data():
    """Create sample test data"""
    sample_queries = [
        {"query": "Data analyst with SQL skills"},
        {"query": "Customer service with teamwork"},
        {"query": "Business analyst with analytical skills"}
    ]
    
    import os
    os.makedirs('data', exist_ok=True)
    
    with open('data/test_unlabeled.json', 'w') as f:
        json.dump(sample_queries, f, indent=2)
    
    logger.info("✓ Created sample test data")


def main():
    """Run evaluation pipeline"""
    logger.info("Starting evaluation pipeline...\n")
    
    import os
    if not os.path.exists('data/train_labeled.json'):
        create_sample_training_data()
    
    if not os.path.exists('data/test_unlabeled.json'):
        create_sample_test_data()
    
    logger.info("STEP 1: Evaluate on Training Set")
    logger.info("=" * 60)
    evaluate_on_training_set()
    
    logger.info("\n\nSTEP 2: Generate Test Predictions")
    logger.info("=" * 60)
    generate_test_predictions()
    
    logger.info("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()
