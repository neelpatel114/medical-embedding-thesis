# Medical Knowledge Organization Through Embedding Models

**Evaluating Alignment with Expert-Tagged Data**

*A thesis project by Neel Patel*  
*Master of Science in Engineering – Electrical Engineering*  
*University of Nevada, Las Vegas*

## Research Overview

This research investigates how effectively BERT embedding models can capture and represent the organization of medical knowledge as defined by human experts. We use the AnKing flashcard dataset with its expert-curated tagging system as our gold standard for medical knowledge organization.

## Key Research Questions

1. How well do fine-tuned embedding models align with expert-created medical knowledge structures?
2. What impact do data enhancement techniques (LLM-based acronym expansion, readability improvements) have on alignment with expert tags?
3. Which embedding approaches best capture hierarchical medical knowledge organization?
4. How does performance vary across different medical domains and tag hierarchies?

## Methodology

### Models Evaluated
- **BERT Raw** - Base BERT fine-tuned on original medical textbooks
- **BERT Enhanced** - Base BERT fine-tuned on LLM-enhanced medical textbooks

*Note: Focus on BERT variants to establish baseline performance and evaluate data enhancement impact.*

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
│   ├── bert_raw_simple/        # BERT trained on original textbooks
│   ├── bert_enhanced_simple/   # BERT trained on enhanced textbooks
│   └── README.md               # Model information and access
├── scripts/
│   ├── anking_analysis.py      # AnKing dataset analysis
│   ├── embedding_alignment.py  # Expert-model alignment evaluation
│   ├── simple_bert_training.py # Simplified BERT training script
│   ├── package_models.py       # Model packaging for distribution
│   └── quick_data_check.py     # Data validation utility
├── slurm_scripts/              # HPC training scripts
│   ├── setup_environment.sh    # Environment setup
│   ├── train_bert_simple.slurm # Simplified BERT training (both models)
│   └── submit_bert_training.sh # BERT training submission
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
   # Submit BERT training job
   cd slurm_scripts
   sbatch train_bert_simple.slurm
   ```

4. **Monitor Progress**
   ```bash
   # Check job status
   squeue -u $USER
   
   # Monitor training logs
   tail -f logs/bert_simple_*.log
   ```

### Training Configuration

| Model | Data Type | GPUs | Memory | Estimated Time |
|-------|-----------|------|--------|----------------|
| BERT Raw | Original textbooks | 1x A40 | 32GB | 2-4 hours |
| BERT Enhanced | LLM-enhanced textbooks | 1x A40 | 32GB | 2-4 hours |

*Note: Simplified training for rapid prototyping and debugging.*

**Key Optimizations:**
- Fixed environment compatibility (PyTorch 2.0.1 + Transformers 4.30.2)
- Simplified training approach with better error handling
- Optimized hyperparameters (batch size, learning rate, warmup)
- Medical domain dummy data for testing and validation
- Enhanced logging and monitoring capabilities

## Research Status

- [x] Literature review and background research
- [x] Dataset collection and preprocessing  
- [x] Optimized training pipeline development
- [x] SLURM cluster setup and configuration
- [ ] AnKing dataset analysis and tag structure examination
- [🔄] Model training execution (2 BERT variants) - In Progress: Job 41312
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

1. **BERT Raw**: Standard BERT fine-tuned on original medical textbooks
2. **BERT Enhanced**: Standard BERT fine-tuned on LLM-enhanced medical textbooks

*Focus: Evaluate impact of data enhancement techniques on medical knowledge representation.*

### Training Approach
- **Base Architecture**: BERT-base-uncased (110M parameters)
- **Fine-tuning**: Masked language modeling on medical domain
- **Data Enhancement**: LLM-based acronym expansion and readability improvements
- **Evaluation**: Alignment with expert-tagged AnKing flashcards

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