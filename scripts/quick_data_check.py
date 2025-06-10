#!/usr/bin/env python3
"""
Quick Data Check Script

Validates that training data is available and properly formatted
before starting training jobs.

Author: Neel Patel
"""

import os
import glob
from pathlib import Path

def check_data_availability():
    """Check if training data is available and accessible."""
    
    print("=" * 60)
    print("Medical Embedding Training Data Check")
    print("=" * 60)
    
    # Check for existing data from previous project
    old_project_data = "/home/pateln3/medical_bert_project/data"
    
    if os.path.exists(old_project_data):
        print(f"[+] Found existing data directory: {old_project_data}")
        
        # Check enhanced data
        enhanced_path = os.path.join(old_project_data, "enhanced_textbooks", "bert")
        if os.path.exists(enhanced_path):
            enhanced_files = glob.glob(os.path.join(enhanced_path, "*.txt"))
            print(f"[+] Enhanced data: {len(enhanced_files):,} files")
            
            # Sample check
            if enhanced_files:
                sample_file = enhanced_files[0]
                try:
                    with open(sample_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        print(f"[+] Sample enhanced file: {len(content)} characters")
                        if len(content) < 50:
                            print(f"[!] Warning: Sample file seems very short")
                except Exception as e:
                    print(f"[-] Error reading enhanced sample: {e}")
        else:
            print("[-] Enhanced textbook data not found")
        
        # Check raw data  
        raw_path = os.path.join(old_project_data, "raw_textbooks", "bert")
        if os.path.exists(raw_path):
            raw_files = glob.glob(os.path.join(raw_path, "*.txt"))
            print(f"[+] Raw data: {len(raw_files):,} files")
            
            # Sample check
            if raw_files:
                sample_file = raw_files[0]
                try:
                    with open(sample_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        print(f"[+] Sample raw file: {len(content)} characters")
                        if len(content) < 50:
                            print(f"[!] Warning: Sample file seems very short")
                except Exception as e:
                    print(f"[-] Error reading raw sample: {e}")
        else:
            print("[-] Raw textbook data not found")
            
    else:
        print(f"[-] No existing data found at {old_project_data}")
        
    # Check current project data directory
    current_data = "data/textbooks"
    if os.path.exists(current_data):
        print(f"[+] Current project data directory exists: {current_data}")
    else:
        print(f"[!] Current project data directory not found: {current_data}")
        
    print("\nRecommendations:")
    if os.path.exists(old_project_data):
        print("1. Data is available from previous project")
        print("2. Training scripts will automatically use this data")
        print("3. Consider copying/linking data to current project if needed")
    else:
        print("1. Need to prepare training data")
        print("2. Copy medical textbook data to data/textbooks/")
        print("3. Ensure data is in proper format (.txt files)")
        
    print("=" * 60)

if __name__ == "__main__":
    check_data_availability()