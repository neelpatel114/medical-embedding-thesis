#!/usr/bin/env python3
"""
Optimized Medical BERT Training Script

This script implements optimized training for BERT models on medical data,
incorporating fixes from previous analysis:
- Correct data paths and preprocessing
- Multi-GPU utilization  
- Proper hyperparameters
- Enhanced data handling
- Robust logging and checkpointing

Author: Neel Patel
Date: June 2025
"""

import os
import json
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging
from dataclasses import dataclass

from transformers import (
    AutoTokenizer, AutoModelForMaskedLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup
)
from datasets import Dataset, load_from_disk
import wandb

@dataclass
class TrainingConfig:
    """Configuration for medical BERT training."""
    
    # Model configuration
    model_name: str = "bert-base-uncased"
    max_length: int = 512
    
    # Data configuration
    data_dir: str = "data/textbooks/processed"
    train_split: float = 0.95
    
    # Training hyperparameters
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_epochs: int = 15
    warmup_ratio: float = 0.1
    
    # MLM configuration
    mlm_probability: float = 0.15
    
    # Optimization
    fp16: bool = True
    dataloader_num_workers: int = 4
    
    # Logging and saving
    output_dir: str = "models/medical_bert"
    logging_steps: int = 100
    eval_steps: int = 1000
    save_steps: int = 2000
    save_total_limit: int = 3
    
    # Distributed training
    local_rank: int = -1
    
    # Data enhancement
    use_enhanced_data: bool = False
    acronym_expansion: bool = False

class MedicalTextDataset:
    """Handles medical text data preprocessing and loading."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.tokenizer = None
        
    def load_and_process_data(self, tokenizer) -> Dict:
        """
        Load and process medical text data.
        
        Args:
            tokenizer: Tokenizer to use for processing
            
        Returns:
            Dictionary with train/eval datasets
        """
        self.tokenizer = tokenizer
        
        logging.info(f"Loading data from: {self.config.data_dir}")
        
        # Determine data source based on configuration
        if self.config.use_enhanced_data:
            data_path = Path(self.config.data_dir) / "enhanced"
            logging.info("Using enhanced medical text data")
        else:
            data_path = Path(self.config.data_dir) / "raw"
            logging.info("Using raw medical text data")
        
        # Load text files
        texts = self._load_text_files(data_path)
        
        # Clean and filter texts
        texts = self._clean_texts(texts)
        
        # Create dataset
        dataset = Dataset.from_dict({"text": texts})
        
        # Train/eval split
        split_dataset = dataset.train_test_split(
            test_size=1.0 - self.config.train_split,
            shuffle=True,
            seed=42
        )
        
        # Tokenize
        tokenized_datasets = split_dataset.map(
            self._tokenize_function,
            batched=True,
            num_proc=self.config.dataloader_num_workers,
            remove_columns=["text"],
            desc="Tokenizing dataset"
        )
        
        logging.info(f"Dataset sizes - Train: {len(tokenized_datasets['train'])}, "
                    f"Eval: {len(tokenized_datasets['test'])}")
        
        return tokenized_datasets
    
    def _load_text_files(self, data_path: Path) -> List[str]:
        """Load text files from directory."""
        texts = []
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_path}")
        
        # Load .txt files
        txt_files = list(data_path.glob("*.txt"))
        for file_path in txt_files[:1000]:  # Limit for testing
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content and len(content) > 50:  # Filter very short texts
                        texts.append(content)
            except Exception as e:
                logging.warning(f"Error reading {file_path}: {e}")
        
        logging.info(f"Loaded {len(texts)} text samples")
        return texts
    
    def _clean_texts(self, texts: List[str]) -> List[str]:
        """Clean and filter medical texts."""
        cleaned_texts = []
        
        for text in texts:
            # Remove common non-medical content
            if any(phrase in text.lower() for phrase in [
                "copyright", "all rights reserved", "printed in", 
                "table of contents", "index", "bibliography"
            ]):
                continue
            
            # Basic cleaning
            text = text.strip()
            
            # Length filtering
            if 50 <= len(text) <= 5000:  # Reasonable length range
                cleaned_texts.append(text)
        
        logging.info(f"Filtered to {len(cleaned_texts)} clean texts")
        return cleaned_texts
    
    def _tokenize_function(self, examples):
        """Tokenize text examples."""
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding=False,  # Dynamic padding is more efficient
            max_length=self.config.max_length,
            return_special_tokens_mask=True
        )

class OptimizedMedicalTrainer:
    """Optimized trainer for medical BERT models."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.setup_logging()
        self.setup_distributed()
        
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{self.config.output_dir}/training.log"),
                logging.StreamHandler()
            ]
        )
        
        # Setup wandb if available
        if self.config.local_rank <= 0:
            try:
                wandb.init(
                    project="medical-bert-thesis",
                    config=self.config.__dict__,
                    name=f"medical_bert_{self.config.model_name.split('/')[-1]}"
                )
            except:
                logging.warning("Wandb not available, skipping initialization")
    
    def setup_distributed(self):
        """Setup distributed training if applicable."""
        if self.config.local_rank != -1:
            torch.cuda.set_device(self.config.local_rank)
            dist.init_process_group(backend='nccl')
            logging.info(f"Initialized distributed training on rank {self.config.local_rank}")
    
    def load_model_and_tokenizer(self):
        """Load model and tokenizer."""
        logging.info(f"Loading model: {self.config.model_name}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            use_fast=True
        )
        
        # Add special tokens if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = AutoModelForMaskedLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16 if self.config.fp16 else torch.float32
        )
        
        # Resize token embeddings if needed
        model.resize_token_embeddings(len(tokenizer))
        
        logging.info(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
        
        return model, tokenizer
    
    def create_training_arguments(self) -> TrainingArguments:
        """Create optimized training arguments."""
        
        # Calculate effective batch size
        effective_batch_size = (
            self.config.batch_size * 
            self.config.gradient_accumulation_steps * 
            max(1, torch.cuda.device_count())
        )
        
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            overwrite_output_dir=True,
            
            # Training configuration
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            
            # Optimization
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type="linear",
            
            # Hardware optimization
            fp16=self.config.fp16,
            dataloader_num_workers=self.config.dataloader_num_workers,
            dataloader_pin_memory=True,
            
            # Logging and evaluation
            logging_steps=self.config.logging_steps,
            evaluation_strategy="steps",
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            
            # Output and reporting
            logging_dir=f"{self.config.output_dir}/logs",
            report_to="wandb" if self.config.local_rank <= 0 else None,
            
            # Distributed training
            local_rank=self.config.local_rank,
            ddp_find_unused_parameters=False,
            
            # Memory optimization
            gradient_checkpointing=True,
            dataloader_drop_last=True,
            
            # Model saving
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
        
        logging.info(f"Effective batch size: {effective_batch_size}")
        return training_args
    
    def train(self):
        """Run optimized training."""
        logging.info("Starting optimized medical BERT training")
        
        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Load model and tokenizer
        model, tokenizer = self.load_model_and_tokenizer()
        
        # Prepare data
        dataset_handler = MedicalTextDataset(self.config)
        datasets = dataset_handler.load_and_process_data(tokenizer)
        
        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=self.config.mlm_probability,
            pad_to_multiple_of=8 if self.config.fp16 else None
        )
        
        # Create training arguments
        training_args = self.create_training_arguments()
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=datasets["train"],
            eval_dataset=datasets["test"],
            tokenizer=tokenizer,
        )
        
        # Add custom callbacks if needed
        # trainer.add_callback(CustomCallback())
        
        # Train
        logging.info("Starting training...")
        train_result = trainer.train()
        
        # Save final model
        trainer.save_model()
        trainer.save_state()
        
        # Evaluate
        eval_results = trainer.evaluate()
        
        # Log final results
        logging.info("Training completed!")
        logging.info(f"Final train loss: {train_result.training_loss:.4f}")
        logging.info(f"Final eval loss: {eval_results['eval_loss']:.4f}")
        logging.info(f"Final perplexity: {np.exp(eval_results['eval_loss']):.4f}")
        
        # Save training config
        with open(f"{self.config.output_dir}/training_config.json", 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        return train_result, eval_results

def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimized Medical BERT Training')
    
    # Model arguments
    parser.add_argument('--model-name', default='bert-base-uncased',
                       help='Base model to fine-tune')
    parser.add_argument('--output-dir', required=True,
                       help='Output directory for trained model')
    
    # Data arguments
    parser.add_argument('--data-dir', required=True,
                       help='Directory containing training data')
    parser.add_argument('--use-enhanced-data', action='store_true',
                       help='Use enhanced medical text data')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--num-epochs', type=int, default=15,
                       help='Number of training epochs')
    
    # System arguments
    parser.add_argument('--local-rank', type=int, default=-1,
                       help='Local rank for distributed training')
    
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        use_enhanced_data=args.use_enhanced_data,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        local_rank=args.local_rank
    )
    
    # Run training
    trainer = OptimizedMedicalTrainer(config)
    train_result, eval_results = trainer.train()
    
    print(f"Training completed! Final perplexity: {np.exp(eval_results['eval_loss']):.4f}")

if __name__ == "__main__":
    main()