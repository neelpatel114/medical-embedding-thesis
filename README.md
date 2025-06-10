# Medical Knowledge Organization Through Embedding Models

**Evaluating Alignment with Expert-Tagged Data**

*A thesis project by Neel Patel*  
*Master of Science in Engineering – Electrical Engineering*  
*University of Nevada, Las Vegas*

## Research Overview

This research investigates how effectively modern embedding models (BERT and ModernBERT) can capture and represent the organization of medical knowledge as defined by human experts. We use the AnKing flashcard dataset with its expert-curated tagging system as our gold standard for medical knowledge organization.

## Key Research Questions

1. How well do fine-tuned embedding models align with expert-created medical knowledge structures?
2. What impact do data enhancement techniques (LLM-based acronym expansion, readability improvements) have on alignment with expert tags?
3. Which embedding approaches best capture hierarchical medical knowledge organization?
4. How does performance vary across different medical domains and tag hierarchies?

## Methodology

### Models Evaluated
- **Base BERT** (bert-base-uncased)
- **BioBERT** (dmis-lab/biobert-v1.1) - Pre-trained on biomedical literature
- **ClinicalBERT** (emilyalsentzer/Bio_ClinicalBERT) - Pre-trained on clinical notes
- **Fine-tuned variants** on medical textbooks (raw vs enhanced data)

### Datasets
- **Training Data**: Medical textbooks (raw and LLM-enhanced)
- **Evaluation Data**: AnKing flashcard deck with expert-curated tags
- **Enhancement**: LLM-based acronym expansion and readability improvements

### Evaluation Framework
- Tag prediction accuracy
- Centroid aggregation analysis  
- Hierarchical evaluation metrics
- Cross-domain performance analysis

## Project Structure

```
medical_embedding_thesis/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .gitignore                  # Git ignore rules
├── data/
│   ├── anking/                 # AnKing flashcard dataset
│   ├── textbooks/              # Medical textbook training data
│   └── embeddings/             # Generated embeddings
├── models/
│   ├── bert_base/              # Base BERT model variants
│   │   ├── bert_raw_optimized/
│   │   └── bert_enhanced_optimized/
│   ├── biobert/                # BioBERT model variants
│   │   └── biobert_enhanced_optimized/
│   ├── clinicalbert/           # ClinicalBERT model variants
│   │   └── clinicalbert_enhanced_optimized/
│   └── enhanced_variants/       # Enhanced training approaches
├── scripts/
│   ├── anking_analysis.py      # AnKing dataset analysis
│   ├── embedding_alignment.py  # Expert-model alignment evaluation
│   ├── optimized_training.py   # Optimized training pipeline
│   └── quick_data_check.py     # Data validation utility
├── slurm_scripts/              # HPC training scripts
│   ├── setup_environment.sh    # Environment setup
│   ├── train_bert_raw.slurm    # Base BERT raw training
│   ├── train_bert_enhanced.slurm # Base BERT enhanced training
│   ├── train_biobert_enhanced.slurm # BioBERT enhanced training
│   ├── train_clinicalbert_enhanced.slurm # ClinicalBERT enhanced training
│   └── submit_all_training.sh  # Master training submission
├── notebooks/
│   └── thesis_analysis.ipynb   # Main analysis notebook
├── evaluation/                 # Evaluation results and metrics
├── results/                    # Experimental results and visualizations
├── logs/                       # Training logs and outputs
└── docs/                       # Documentation and thesis materials
```

## Key Contributions

1. **Novel Validation Framework**: Using expert-tagged flashcards as ground truth
2. **Quantitative Analysis**: Data enhancement impact on medical knowledge representation
3. **Practical Insights**: Guidelines for implementing embedding models in medical education
4. **Alignment Metrics**: Framework for evaluating expert-model alignment

## Getting Started

### Prerequisites
- Access to SLURM-managed HPC cluster with GPU nodes
- Python 3.9+ with conda/miniconda
- Medical textbook dataset (89,506 files available)
- AnKing flashcard database (collection.anki2)

### Quick Start

1. **Setup Environment**
   ```bash
   cd medical_embedding_thesis
   sbatch slurm_scripts/setup_environment.sh
   ```

2. **Validate Data**
   ```bash
   python3 scripts/quick_data_check.py
   ```

3. **Start Training**
   ```bash
   # Submit all training jobs
   cd slurm_scripts
   ./submit_all_training.sh
   
   # Or submit individual jobs
   sbatch train_bert_enhanced.slurm
   sbatch train_modernbert_enhanced.slurm
   ```

4. **Monitor Progress**
   ```bash
   # Check job status
   squeue -u $USER
   
   # Monitor training logs
   tail -f logs/bert_enhanced_*.log
   ```

### Training Configuration

| Model | Data Type | GPUs | Memory | Estimated Time |
|-------|-----------|------|--------|----------------|
| Base BERT | Raw/Enhanced | 2x A40 | 64GB | 24-36 hours |
| BioBERT | Enhanced | 2x A40 | 64GB | 24-36 hours |
| ClinicalBERT | Enhanced | 2x A40 | 64GB | 24-36 hours |

**Key Optimizations:**
- Multi-GPU training with proper data parallelization
- Optimized hyperparameters (batch size, learning rate, warmup)
- Local data copying for faster I/O
- Compatible library versions (PyTorch 2.0.1 + Transformers 4.30.2)
- Enhanced data preprocessing and cleaning

## Research Status

- [x] Literature review and background research
- [x] Dataset collection and preprocessing  
- [x] Optimized training pipeline development
- [x] SLURM cluster setup and configuration
- [ ] AnKing dataset analysis and tag structure examination
- [ ] Model training execution (4 variants)
- [ ] Embedding alignment evaluation
- [ ] Cross-domain performance analysis
- [ ] Results analysis and thesis writing

## Training Data

**Available Datasets:**
- **Enhanced Medical Textbooks**: 89,506 files with LLM-based improvements
- **Raw Medical Textbooks**: 89,506 files in original format
- **AnKing Flashcards**: Expert-tagged medical knowledge base

**Data Enhancements Applied:**
- Medical acronym expansion using DeepSeek
- Readability improvements while preserving technical accuracy
- Removal of non-medical content (copyright, publisher info)
- Consistent tokenization and formatting

## Model Variants

1. **BERT-Raw**: Standard BERT fine-tuned on raw medical textbooks
2. **BERT-Enhanced**: Standard BERT fine-tuned on LLM-enhanced medical textbooks  
3. **BioBERT-Enhanced**: BioBERT fine-tuned on enhanced medical textbooks
4. **ClinicalBERT-Enhanced**: ClinicalBERT fine-tuned on enhanced medical textbooks

### Architecture Comparison
- **Base BERT**: General domain pre-training on BookCorpus + Wikipedia
- **BioBERT**: Domain-specific pre-training on PubMed abstracts + PMC full-text articles
- **ClinicalBERT**: Domain-specific pre-training on clinical notes (MIMIC-III dataset)

## Evaluation Framework

**Alignment Metrics:**
- Tag prediction accuracy against expert classifications
- Domain clustering coherence scores
- Hierarchical knowledge preservation metrics
- Cross-domain generalization analysis

**Validation Approach:**
- AnKing expert-tagged flashcards as ground truth
- Multi-label classification evaluation
- Embedding space visualization and analysis
- Statistical significance testing across domains