#!/usr/bin/env python3
"""
AnKing Dataset Analysis Script

This script analyzes the AnKing flashcard dataset to understand:
1. Tag hierarchy structure
2. Domain distribution
3. Card content characteristics
4. Expert-created knowledge organization patterns

Author: Neel Patel
Date: June 2025
"""

import json
import sqlite3
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import re
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Set

class AnKingAnalyzer:
    """Analyzes AnKing flashcard dataset for medical knowledge organization patterns."""
    
    def __init__(self, anki_db_path: str, output_dir: str):
        """
        Initialize the analyzer.
        
        Args:
            anki_db_path: Path to Anki collection.anki2 database
            output_dir: Directory to save analysis results
        """
        self.db_path = anki_db_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data containers
        self.cards_df = None
        self.notes_df = None
        self.tag_hierarchy = defaultdict(list)
        self.domain_stats = {}
        
    def load_anki_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load cards and notes from Anki database.
        
        Returns:
            Tuple of (cards_df, notes_df)
        """
        print("Loading AnKing dataset from Anki database...")
        
        conn = sqlite3.connect(self.db_path)
        
        # Load cards
        cards_query = """
        SELECT 
            id,
            nid as note_id,
            did as deck_id,
            ord,
            mod,
            usn,
            type,
            queue,
            due,
            ivl,
            factor,
            reps,
            lapses,
            left,
            odue,
            odid,
            flags,
            data
        FROM cards
        """
        
        # Load notes
        notes_query = """
        SELECT 
            id,
            guid,
            mid as model_id,
            mod,
            usn,
            tags,
            flds as fields,
            sfld as sort_field,
            csum,
            flags,
            data
        FROM notes
        """
        
        self.cards_df = pd.read_sql_query(cards_query, conn)
        self.notes_df = pd.read_sql_query(notes_query, conn)
        
        conn.close()
        
        print(f"Loaded {len(self.cards_df)} cards and {len(self.notes_df)} notes")
        return self.cards_df, self.notes_df
    
    def parse_tags(self) -> Dict[str, List[str]]:
        """
        Parse and analyze tag structure from notes.
        
        Returns:
            Dictionary mapping note_id to list of tags
        """
        print("Parsing tag structure...")
        
        note_tags = {}
        all_tags = set()
        
        for _, note in self.notes_df.iterrows():
            if pd.isna(note['tags']):
                tags = []
            else:
                # Tags are space-separated in Anki
                tags = [tag.strip() for tag in note['tags'].split() if tag.strip()]
            
            note_tags[note['id']] = tags
            all_tags.update(tags)
        
        print(f"Found {len(all_tags)} unique tags across {len(note_tags)} notes")
        return note_tags
    
    def analyze_tag_hierarchy(self, note_tags: Dict[str, List[str]]) -> Dict:
        """
        Analyze hierarchical structure of tags.
        
        Args:
            note_tags: Dictionary mapping note_id to tags
            
        Returns:
            Dictionary with hierarchy analysis
        """
        print("Analyzing tag hierarchy...")
        
        # Collect all tags
        all_tags = []
        for tags in note_tags.values():
            all_tags.extend(tags)
        
        tag_counts = Counter(all_tags)
        
        # Analyze hierarchy patterns (tags often use :: for hierarchy)
        hierarchy_levels = defaultdict(list)
        root_tags = set()
        
        for tag in tag_counts.keys():
            if '::' in tag:
                parts = tag.split('::')
                hierarchy_levels[len(parts)].append(tag)
                root_tags.add(parts[0])
            else:
                hierarchy_levels[1].append(tag)
                root_tags.add(tag)
        
        # Analyze medical domains (common root tags)
        medical_domains = {}
        for root_tag in root_tags:
            domain_tags = [tag for tag in tag_counts.keys() if tag.startswith(root_tag)]
            if len(domain_tags) > 5:  # Only include domains with substantial content
                medical_domains[root_tag] = {
                    'tag_count': len(domain_tags),
                    'total_cards': sum(tag_counts[tag] for tag in domain_tags),
                    'subtags': domain_tags
                }
        
        hierarchy_analysis = {
            'total_tags': len(tag_counts),
            'tag_counts': dict(tag_counts.most_common(50)),  # Top 50 tags
            'hierarchy_levels': {k: len(v) for k, v in hierarchy_levels.items()},
            'max_hierarchy_depth': max(hierarchy_levels.keys()) if hierarchy_levels else 0,
            'root_tags': list(root_tags),
            'medical_domains': medical_domains
        }
        
        return hierarchy_analysis
    
    def analyze_card_content(self) -> Dict:
        """
        Analyze the content characteristics of flashcards.
        
        Returns:
            Dictionary with content analysis
        """
        print("Analyzing card content...")
        
        content_stats = {
            'total_cards': len(self.cards_df),
            'total_notes': len(self.notes_df),
            'field_analysis': {},
            'content_lengths': {}
        }
        
        # Analyze note fields (front, back, extra, etc.)
        field_lengths = []
        
        for _, note in self.notes_df.iterrows():
            if pd.notna(note['fields']):
                fields = note['fields'].split('\x1f')  # Anki uses \x1f as field separator
                
                for i, field in enumerate(fields):
                    # Remove HTML tags for length calculation
                    clean_field = re.sub(r'<[^>]+>', '', field)
                    field_lengths.append(len(clean_field))
        
        if field_lengths:
            content_stats['content_lengths'] = {
                'mean_length': np.mean(field_lengths),
                'median_length': np.median(field_lengths),
                'std_length': np.std(field_lengths),
                'min_length': np.min(field_lengths),
                'max_length': np.max(field_lengths)
            }
        
        return content_stats
    
    def generate_domain_analysis(self, hierarchy_analysis: Dict) -> Dict:
        """
        Generate detailed analysis of medical domains.
        
        Args:
            hierarchy_analysis: Results from analyze_tag_hierarchy
            
        Returns:
            Detailed domain analysis
        """
        print("Generating domain analysis...")
        
        domains = hierarchy_analysis['medical_domains']
        
        # Sort domains by card count
        sorted_domains = sorted(domains.items(), 
                              key=lambda x: x[1]['total_cards'], 
                              reverse=True)
        
        domain_analysis = {
            'top_domains': dict(sorted_domains[:20]),  # Top 20 domains
            'domain_distribution': {
                name: info['total_cards'] 
                for name, info in sorted_domains[:10]
            },
            'coverage_analysis': {
                'total_domains': len(domains),
                'top_10_coverage': sum(info['total_cards'] for _, info in sorted_domains[:10])
            }
        }
        
        return domain_analysis
    
    def save_analysis_results(self, results: Dict):
        """Save analysis results to files."""
        print("Saving analysis results...")
        
        # Save JSON summary
        output_file = self.output_dir / "anking_analysis_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Analysis results saved to: {output_file}")
    
    def create_visualizations(self, results: Dict):
        """Create visualizations of the analysis results."""
        print("Creating visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        fig_size = (12, 8)
        
        # 1. Domain distribution
        if 'domain_analysis' in results and 'domain_distribution' in results['domain_analysis']:
            plt.figure(figsize=fig_size)
            domain_dist = results['domain_analysis']['domain_distribution']
            
            plt.bar(range(len(domain_dist)), list(domain_dist.values()))
            plt.xticks(range(len(domain_dist)), list(domain_dist.keys()), rotation=45, ha='right')
            plt.title('Top Medical Domains by Card Count')
            plt.xlabel('Medical Domain')
            plt.ylabel('Number of Cards')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'domain_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Tag hierarchy levels
        if 'hierarchy_analysis' in results and 'hierarchy_levels' in results['hierarchy_analysis']:
            plt.figure(figsize=(10, 6))
            hierarchy_levels = results['hierarchy_analysis']['hierarchy_levels']
            
            plt.bar(hierarchy_levels.keys(), hierarchy_levels.values())
            plt.title('Tag Hierarchy Distribution')
            plt.xlabel('Hierarchy Depth')
            plt.ylabel('Number of Tags')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'hierarchy_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print("Visualizations saved to output directory")
    
    def run_complete_analysis(self) -> Dict:
        """
        Run complete AnKing dataset analysis.
        
        Returns:
            Complete analysis results
        """
        print("=" * 60)
        print("Starting AnKing Dataset Analysis")
        print("=" * 60)
        
        # Load data
        self.load_anki_data()
        
        # Parse tags
        note_tags = self.parse_tags()
        
        # Analyze tag hierarchy
        hierarchy_analysis = self.analyze_tag_hierarchy(note_tags)
        
        # Analyze content
        content_analysis = self.analyze_card_content()
        
        # Domain analysis
        domain_analysis = self.generate_domain_analysis(hierarchy_analysis)
        
        # Compile results
        results = {
            'analysis_summary': {
                'total_cards': len(self.cards_df),
                'total_notes': len(self.notes_df),
                'total_unique_tags': hierarchy_analysis['total_tags'],
                'medical_domains': len(hierarchy_analysis['medical_domains'])
            },
            'hierarchy_analysis': hierarchy_analysis,
            'content_analysis': content_analysis,
            'domain_analysis': domain_analysis,
            'note_tags': note_tags  # For downstream processing
        }
        
        # Save results
        self.save_analysis_results(results)
        
        # Create visualizations
        self.create_visualizations(results)
        
        print("=" * 60)
        print("AnKing Analysis Complete!")
        print("=" * 60)
        
        return results

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze AnKing flashcard dataset')
    parser.add_argument('--anki-db', required=True, 
                       help='Path to Anki collection.anki2 database file')
    parser.add_argument('--output-dir', default='./anking_analysis_output',
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    # Run analysis
    analyzer = AnKingAnalyzer(args.anki_db, args.output_dir)
    results = analyzer.run_complete_analysis()
    
    # Print summary
    print("\nAnalysis Summary:")
    print(f"Total Cards: {results['analysis_summary']['total_cards']:,}")
    print(f"Total Notes: {results['analysis_summary']['total_notes']:,}")
    print(f"Unique Tags: {results['analysis_summary']['total_unique_tags']:,}")
    print(f"Medical Domains: {results['analysis_summary']['medical_domains']:,}")

if __name__ == "__main__":
    main()