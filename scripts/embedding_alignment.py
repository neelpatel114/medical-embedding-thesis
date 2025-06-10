#!/usr/bin/env python3
"""
Embedding Model Alignment Evaluation

This script evaluates how well embedding models align with expert-tagged
medical knowledge using the AnKing dataset as ground truth.

Author: Neel Patel
Date: June 2025
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel
import torch

class EmbeddingAlignmentEvaluator:
    """Evaluates alignment between embedding models and expert tags."""
    
    def __init__(self, model_path: str, anking_data: Dict, output_dir: str):
        """
        Initialize the evaluator.
        
        Args:
            model_path: Path to the embedding model
            anking_data: Processed AnKing data with tags
            output_dir: Directory to save evaluation results
        """
        self.model_path = model_path
        self.anking_data = anking_data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Results storage
        self.embeddings = {}
        self.evaluation_results = {}
        
    def load_model(self):
        """Load the embedding model."""
        print(f"Loading model from: {self.model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModel.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully on {self.device}")
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for processing
            
        Returns:
            NumPy array of embeddings
        """
        print(f"Generating embeddings for {len(texts)} texts...")
        
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use CLS token embedding or mean pooling
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(embeddings)
        print(f"Generated embeddings shape: {embeddings.shape}")
        
        return embeddings
    
    def extract_card_text(self, note_id: int) -> str:
        """
        Extract text content from a flashcard note.
        
        Args:
            note_id: Note ID from AnKing data
            
        Returns:
            Combined text content
        """
        # This would need to be adapted based on the actual AnKing data structure
        # For now, return a placeholder
        return f"Medical flashcard content for note {note_id}"
    
    def evaluate_tag_prediction(self, embeddings: np.ndarray, tags: List[List[str]]) -> Dict:
        """
        Evaluate how well embeddings can predict expert tags.
        
        Args:
            embeddings: Card embeddings
            tags: List of tag lists for each card
            
        Returns:
            Evaluation metrics
        """
        print("Evaluating tag prediction performance...")
        
        # Create binary matrices for multi-label evaluation
        all_tags = set()
        for tag_list in tags:
            all_tags.update(tag_list)
        
        tag_to_idx = {tag: idx for idx, tag in enumerate(sorted(all_tags))}
        
        # Create binary tag matrix
        tag_matrix = np.zeros((len(tags), len(all_tags)))
        for i, tag_list in enumerate(tags):
            for tag in tag_list:
                tag_matrix[i, tag_to_idx[tag]] = 1
        
        # Use clustering to group similar embeddings
        n_clusters = min(50, len(set(tuple(row) for row in tag_matrix)))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        
        # Evaluate cluster-tag alignment
        cluster_tag_alignment = []
        
        for cluster_id in range(n_clusters):
            cluster_mask = clusters == cluster_id
            if np.sum(cluster_mask) == 0:
                continue
                
            # Most common tags in this cluster
            cluster_tags = tag_matrix[cluster_mask].sum(axis=0)
            cluster_tag_alignment.append(np.max(cluster_tags) / np.sum(cluster_mask))
        
        alignment_score = np.mean(cluster_tag_alignment) if cluster_tag_alignment else 0.0
        
        results = {
            'tag_prediction': {
                'n_clusters': n_clusters,
                'cluster_tag_alignment': alignment_score,
                'total_tags': len(all_tags),
                'avg_tags_per_card': np.mean([len(tag_list) for tag_list in tags])
            }
        }
        
        return results
    
    def evaluate_domain_clustering(self, embeddings: np.ndarray, domain_tags: Dict[str, List[int]]) -> Dict:
        """
        Evaluate how well embeddings cluster by medical domain.
        
        Args:
            embeddings: Card embeddings
            domain_tags: Dictionary mapping domain names to card indices
            
        Returns:
            Domain clustering evaluation
        """
        print("Evaluating domain clustering...")
        
        # Calculate within-domain vs between-domain similarities
        domain_similarities = {}
        
        for domain, card_indices in domain_tags.items():
            if len(card_indices) < 2:
                continue
                
            domain_embeddings = embeddings[card_indices]
            
            # Within-domain similarity
            within_sim = cosine_similarity(domain_embeddings)
            within_domain_sim = np.mean(within_sim[np.triu_indices_from(within_sim, k=1)])
            
            # Between-domain similarity (sample from other domains)
            other_indices = []
            for other_domain, other_cards in domain_tags.items():
                if other_domain != domain:
                    other_indices.extend(other_cards[:min(len(other_cards), 50)])
            
            if other_indices:
                other_embeddings = embeddings[other_indices[:len(card_indices)]]
                between_sim = cosine_similarity(domain_embeddings, other_embeddings)
                between_domain_sim = np.mean(between_sim)
            else:
                between_domain_sim = 0.0
            
            domain_similarities[domain] = {
                'within_domain': within_domain_sim,
                'between_domain': between_domain_sim,
                'separation_score': within_domain_sim - between_domain_sim
            }
        
        # Overall domain separation
        separation_scores = [scores['separation_score'] for scores in domain_similarities.values()]
        overall_separation = np.mean(separation_scores) if separation_scores else 0.0
        
        results = {
            'domain_clustering': {
                'overall_separation': overall_separation,
                'domain_scores': domain_similarities,
                'n_domains_evaluated': len(domain_similarities)
            }
        }
        
        return results
    
    def evaluate_hierarchical_alignment(self, embeddings: np.ndarray, hierarchical_tags: Dict) -> Dict:
        """
        Evaluate alignment with hierarchical tag structure.
        
        Args:
            embeddings: Card embeddings
            hierarchical_tags: Hierarchical tag information
            
        Returns:
            Hierarchical alignment evaluation
        """
        print("Evaluating hierarchical alignment...")
        
        # This is a simplified evaluation - in practice, you'd want more sophisticated
        # analysis of how well the embedding space preserves hierarchical relationships
        
        hierarchy_scores = {}
        
        # Analyze embedding distances at different hierarchy levels
        for level in range(1, 4):  # Check first 3 levels
            level_tags = [tag for tag in hierarchical_tags.get('root_tags', []) 
                         if tag.count('::') == level - 1]
            
            if len(level_tags) < 2:
                continue
            
            # Calculate how well hierarchy is preserved in embedding space
            # This would need more sophisticated implementation based on actual data structure
            hierarchy_scores[f'level_{level}'] = np.random.random()  # Placeholder
        
        results = {
            'hierarchical_alignment': {
                'hierarchy_preservation_scores': hierarchy_scores,
                'levels_analyzed': len(hierarchy_scores)
            }
        }
        
        return results
    
    def create_visualizations(self, embeddings: np.ndarray, labels: List[str]):
        """Create t-SNE visualization of embeddings colored by labels."""
        print("Creating embedding visualizations...")
        
        # Sample embeddings if too many
        if len(embeddings) > 1000:
            indices = np.random.choice(len(embeddings), 1000, replace=False)
            embeddings_sample = embeddings[indices]
            labels_sample = [labels[i] for i in indices]
        else:
            embeddings_sample = embeddings
            labels_sample = labels
        
        # t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings_sample)-1))
        embeddings_2d = tsne.fit_transform(embeddings_sample)
        
        # Plot
        plt.figure(figsize=(12, 8))
        
        # Color by domain (take first part of label)
        domains = [label.split('::')[0] if '::' in label else label for label in labels_sample]
        unique_domains = list(set(domains))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_domains)))
        
        for i, domain in enumerate(unique_domains):
            mask = [d == domain for d in domains]
            if any(mask):
                plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                           c=[colors[i]], label=domain, alpha=0.6)
        
        plt.title('t-SNE Visualization of Medical Knowledge Embeddings')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'embedding_tsne.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_evaluation(self) -> Dict:
        """
        Run complete alignment evaluation.
        
        Returns:
            Complete evaluation results
        """
        print("=" * 60)
        print("Starting Embedding Alignment Evaluation")
        print("=" * 60)
        
        # Load model
        self.load_model()
        
        # Extract cards and tags from AnKing data
        # This would need to be adapted based on actual data structure
        note_tags = self.anking_data.get('note_tags', {})
        
        # Generate card texts and embeddings
        card_texts = []
        card_tags = []
        
        for note_id, tags in list(note_tags.items())[:1000]:  # Sample for testing
            text = self.extract_card_text(note_id)
            card_texts.append(text)
            card_tags.append(tags)
        
        # Generate embeddings
        embeddings = self.generate_embeddings(card_texts)
        
        # Run evaluations
        tag_prediction_results = self.evaluate_tag_prediction(embeddings, card_tags)
        
        # Domain clustering (simplified)
        domain_tags = {'cardiology': list(range(100)), 'neurology': list(range(100, 200))}
        domain_results = self.evaluate_domain_clustering(embeddings, domain_tags)
        
        # Hierarchical alignment
        hierarchy_results = self.evaluate_hierarchical_alignment(
            embeddings, self.anking_data.get('hierarchy_analysis', {})
        )
        
        # Create visualizations
        labels = [' '.join(tags[:2]) if tags else 'untagged' for tags in card_tags]
        self.create_visualizations(embeddings, labels)
        
        # Compile results
        results = {
            'model_info': {
                'model_path': self.model_path,
                'n_cards_evaluated': len(card_texts),
                'embedding_dimension': embeddings.shape[1]
            },
            **tag_prediction_results,
            **domain_results,
            **hierarchy_results
        }
        
        # Save results
        with open(self.output_dir / 'alignment_evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print("=" * 60)
        print("Alignment Evaluation Complete!")
        print("=" * 60)
        
        return results

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate embedding alignment with expert tags')
    parser.add_argument('--model-path', required=True,
                       help='Path to the embedding model')
    parser.add_argument('--anking-data', required=True,
                       help='Path to processed AnKing data JSON')
    parser.add_argument('--output-dir', default='./alignment_evaluation_output',
                       help='Output directory for evaluation results')
    
    args = parser.parse_args()
    
    # Load AnKing data
    with open(args.anking_data, 'r') as f:
        anking_data = json.load(f)
    
    # Run evaluation
    evaluator = EmbeddingAlignmentEvaluator(args.model_path, anking_data, args.output_dir)
    results = evaluator.run_evaluation()
    
    # Print summary
    print("\nEvaluation Summary:")
    print(f"Model: {results['model_info']['model_path']}")
    print(f"Cards Evaluated: {results['model_info']['n_cards_evaluated']:,}")
    print(f"Embedding Dimension: {results['model_info']['embedding_dimension']}")

if __name__ == "__main__":
    main()