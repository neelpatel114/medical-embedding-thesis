#!/usr/bin/env python3
"""
Simplified BERT Training Script for Medical Domain

Focuses on training just 2 models with better error handling and debugging.
- BERT Raw (on original textbook data)  
- BERT Enhanced (on LLM-enhanced data)

Author: Neel Patel
Date: June 2025
"""

import os
import sys
import torch
import logging
from pathlib import Path
from datetime import datetime

from transformers import (
    AutoTokenizer, 
    AutoModelForMaskedLM,
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset

def setup_logging(output_dir: Path, model_type: str):
    """Setup logging configuration."""
    log_file = output_dir / f"{model_type}_training.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def create_dummy_data(tokenizer, num_samples=1000):
    """Create dummy medical data for testing if real data not available."""
    medical_texts = [
        "The patient presents with symptoms of cardiovascular disease.",
        "Diagnosis shows signs of pulmonary edema and hypertension.", 
        "Treatment includes ACE inhibitors and beta blockers.",
        "The pathophysiology involves inflammation of cardiac tissue.",
        "Clinical examination reveals abnormal heart rhythm patterns.",
        "Therapeutic intervention requires immediate medication adjustment.",
        "Medical history indicates previous episodes of chest pain.",
        "Laboratory results show elevated troponin levels indicating damage.",
        "Electrocardiogram demonstrates irregular electrical activity patterns.",
        "Patient education about lifestyle modifications is essential for recovery."
    ]
    
    # Repeat and vary the texts
    texts = []
    for i in range(num_samples):
        base_text = medical_texts[i % len(medical_texts)]
        # Add some variation
        if i % 3 == 0:
            texts.append(f"Case study: {base_text}")
        elif i % 3 == 1:
            texts.append(f"Clinical note: {base_text}")
        else:
            texts.append(base_text)
    
    return Dataset.from_dict({"text": texts})

def tokenize_function(examples, tokenizer, max_length=512):
    """Tokenize text data for training."""
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=True,
        max_length=max_length,
        return_special_tokens_mask=True
    )

def train_bert_model(model_type: str, output_dir: Path, enhanced_data: bool = False):
    """Train a single BERT model."""
    
    logger = setup_logging(output_dir, model_type)
    logger.info(f"Starting {model_type} training...")
    logger.info(f"Enhanced data: {enhanced_data}")
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # Setup
        model_name = "bert-base-uncased"
        logger.info(f"Loading model: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(model_name)
        
        logger.info(f"Model loaded successfully. Parameters: {model.num_parameters():,}")
        
        # Create/load dataset
        logger.info("Loading dataset...")
        
        # For now, use dummy data - replace with real data loading
        dataset = create_dummy_data(tokenizer, num_samples=2000)
        logger.info(f"Dataset created with {len(dataset)} samples")
        
        # Tokenize
        logger.info("Tokenizing dataset...")
        tokenized_dataset = dataset.map(
            lambda x: tokenize_function(x, tokenizer),
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # Split train/eval
        train_size = int(0.9 * len(tokenized_dataset))
        eval_size = len(tokenized_dataset) - train_size
        
        train_dataset = tokenized_dataset.select(range(train_size))
        eval_dataset = tokenized_dataset.select(range(train_size, train_size + eval_size))
        
        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Eval samples: {len(eval_dataset)}")
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=0.15
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            overwrite_output_dir=True,
            num_train_epochs=2,  # Short for testing
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=str(output_dir / "logs"),
            logging_steps=50,
            evaluation_strategy="steps",
            eval_steps=200,
            save_steps=500,
            save_total_limit=2,
            prediction_loss_only=True,
            report_to=None,  # Disable wandb for now
            dataloader_num_workers=2,
            fp16=torch.cuda.is_available(),
        )
        
        logger.info("Training arguments configured")
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
        )
        
        logger.info("Trainer initialized successfully")
        
        # Train
        logger.info("Starting training...")
        trainer.train()
        
        # Save model
        logger.info("Saving model...")
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
        # Save training info
        training_info = {
            "model_type": model_type,
            "model_name": model_name,
            "enhanced_data": enhanced_data,
            "training_samples": len(train_dataset),
            "eval_samples": len(eval_dataset),
            "epochs": training_args.num_train_epochs,
            "completed_at": datetime.now().isoformat(),
            "output_dir": str(output_dir)
        }
        
        with open(output_dir / "training_info.json", 'w') as f:
            json.dump(training_info, f, indent=2)
        
        logger.info(f"Training completed successfully!")
        logger.info(f"Model saved to: {output_dir}")
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train BERT models for medical domain')
    parser.add_argument('--model-type', required=True, choices=['bert_raw', 'bert_enhanced'],
                       help='Type of model to train')
    parser.add_argument('--output-dir', required=True, help='Output directory for model')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    enhanced_data = args.model_type == 'bert_enhanced'
    
    print(f"Starting training for {args.model_type}")
    print(f"Output directory: {output_dir}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.current_device()}")
    
    success = train_bert_model(args.model_type, output_dir, enhanced_data)
    
    if success:
        print(f"✓ Training completed successfully!")
        sys.exit(0)
    else:
        print(f"✗ Training failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()