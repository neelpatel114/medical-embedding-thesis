# Trained Models Download Guide

## Model Access Information

The trained models from this research are stored on the HPC cluster and can be accessed for download.

### 📍 **Model Locations**

**HPC Cluster Path**: `/home/pateln3/medical_embedding_thesis/models/`

| Model | Directory | Size | Description |
|-------|-----------|------|-------------|
| **BERT Raw** | `bert_raw_optimized/` | ~500MB | Base BERT on raw medical textbooks |
| **BERT Enhanced** | `bert_enhanced_optimized/` | ~500MB | Base BERT on LLM-enhanced textbooks |
| **BioBERT Enhanced** | `biobert/biobert_enhanced_optimized/` | ~500MB | BioBERT on enhanced textbooks |
| **ClinicalBERT Enhanced** | `clinicalbert/clinicalbert_enhanced_optimized/` | ~500MB | ClinicalBERT on enhanced textbooks |

**Total Size**: ~2GB for all models

### 📦 **Download Options**

#### Option 1: Individual Model Download
```bash
# Copy specific model from HPC
scp -r username@hpc:/home/pateln3/medical_embedding_thesis/models/bert_enhanced_optimized/ ./
```

#### Option 2: Packaged Models (Recommended)
```bash
# Run packaging script to create organized downloads
cd /home/pateln3/medical_embedding_thesis
python scripts/package_models.py

# This creates: packaged_models/
# - bert_raw_medical.tar.gz
# - bert_enhanced_medical.tar.gz  
# - biobert_enhanced_medical.tar.gz
# - clinicalbert_enhanced_medical.tar.gz
# - extract_models.sh (extraction script)
# - manifest.json (package info)
```

#### Option 3: Complete Model Archive
```bash
# Create single archive with all models
tar -czf medical_embedding_models.tar.gz models/
```

### 🔍 **Model Contents**

Each model directory contains:
```
model_directory/
├── pytorch_model.bin           # Main model weights (~400-500MB)
├── config.json                 # Model configuration
├── tokenizer_config.json       # Tokenizer settings
├── vocab.txt                   # Vocabulary
├── special_tokens_map.json     # Special tokens
├── training_args.bin           # Training hyperparameters
└── logs/                       # Training logs
    └── events.out.tfevents.*   # TensorBoard logs
```

### 💻 **Usage Example**

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Load a trained model
model_path = "./bert_enhanced_optimized/"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForMaskedLM.from_pretrained(model_path)

# Use for inference
text = "The patient shows symptoms of [MASK] disease."
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
```

### 📧 **Access Requests**

**For External Researchers:**
- Email: [your-email]@unlv.edu
- Subject: "Medical Embedding Models Access Request"
- Include: Research purpose and intended use

**For Thesis Committee:**
- Models available on request
- Can provide temporary download links
- HPC access can be arranged if needed

### 🎓 **Citation**

If you use these models in your research, please cite:
```
Patel, N. (2025). Medical Knowledge Organization Through Embedding Models: 
Evaluating Alignment with Expert-Tagged Data. Master's Thesis, 
University of Nevada, Las Vegas.
```

### 📊 **Model Performance**

See `results/` directory for:
- Training metrics and loss curves
- Evaluation results on AnKing dataset
- Alignment scores with expert tags
- Comparative analysis across models

### ⚙️ **Technical Requirements**

**To use these models you need:**
- Python 3.9+
- PyTorch 2.0+
- Transformers 4.30+
- ~2GB disk space per model
- GPU recommended for inference

**See `requirements.txt` for complete dependencies.**

---

*This document will be updated as models complete training and become available for download.*

**Last Updated**: $(date)  
**Training Status**: Check `squeue -u pateln3` for current progress