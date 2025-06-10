#!/usr/bin/env python3
"""
Model Packaging Script

Creates clean, organized model packages for download and distribution.
Organizes trained models with proper documentation and metadata.

Author: Neel Patel
Date: June 2025
"""

import os
import shutil
import json
import tarfile
from pathlib import Path
from datetime import datetime

class ModelPackager:
    """Packages trained models for easy distribution and download."""
    
    def __init__(self, models_dir: str = "models", output_dir: str = "packaged_models"):
        self.models_dir = Path(models_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def get_model_info(self, model_path: Path) -> dict:
        """Extract model information for metadata."""
        info = {
            "model_name": model_path.name,
            "path": str(model_path),
            "files": [],
            "size_mb": 0,
            "created": datetime.now().isoformat()
        }
        
        if model_path.exists():
            for file in model_path.rglob("*"):
                if file.is_file():
                    size_mb = file.stat().st_size / (1024 * 1024)
                    info["files"].append({
                        "name": file.name,
                        "relative_path": str(file.relative_to(model_path)),
                        "size_mb": round(size_mb, 2)
                    })
                    info["size_mb"] += size_mb
            info["size_mb"] = round(info["size_mb"], 2)
        
        return info
    
    def create_model_package(self, model_path: Path, package_name: str):
        """Create a compressed package for a model."""
        if not model_path.exists():
            print(f"Warning: Model path {model_path} does not exist")
            return None
            
        # Create package directory
        package_dir = self.output_dir / package_name
        package_dir.mkdir(exist_ok=True)
        
        # Copy model files
        if model_path.exists():
            # Copy main model files (exclude large checkpoints by default)
            for file in model_path.iterdir():
                if file.is_file():
                    shutil.copy2(file, package_dir)
                elif file.is_dir() and file.name == "logs":
                    # Copy logs but compress them
                    logs_dir = package_dir / "logs"
                    logs_dir.mkdir(exist_ok=True)
                    for log_file in file.iterdir():
                        if log_file.suffix in ['.log', '.json']:
                            shutil.copy2(log_file, logs_dir)
        
        # Create metadata
        model_info = self.get_model_info(model_path)
        with open(package_dir / "model_info.json", 'w') as f:
            json.dump(model_info, f, indent=2)
        
        # Create README for this model
        readme_content = f"""# {package_name}

## Model Information
- **Model Type**: {package_name.replace('_', ' ').title()}
- **Training Date**: {datetime.now().strftime('%Y-%m-%d')}
- **Size**: {model_info['size_mb']:.1f} MB
- **Files**: {len(model_info['files'])} files

## Usage
```python
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained(".")
model = AutoModelForMaskedLM.from_pretrained(".")
```

## Files Included
"""
        for file_info in model_info['files']:
            readme_content += f"- `{file_info['name']}` ({file_info['size_mb']:.1f} MB)\n"
        
        with open(package_dir / "README.md", 'w') as f:
            f.write(readme_content)
        
        # Create compressed archive
        archive_path = self.output_dir / f"{package_name}.tar.gz"
        with tarfile.open(archive_path, 'w:gz') as tar:
            tar.add(package_dir, arcname=package_name)
        
        print(f"✓ Created package: {archive_path}")
        print(f"  Size: {archive_path.stat().st_size / (1024*1024):.1f} MB")
        
        return archive_path
    
    def package_all_models(self):
        """Package all trained models."""
        print("=" * 60)
        print("Medical Embedding Models - Packaging")
        print("=" * 60)
        
        # Expected model directories
        model_configs = [
            ("bert_raw_simple", "bert_raw_medical"),
            ("bert_enhanced_simple", "bert_enhanced_medical")
        ]
        
        packages_created = []
        
        for model_dir, package_name in model_configs:
            model_path = self.models_dir / model_dir
            print(f"\nProcessing {package_name}...")
            
            if model_path.exists():
                archive_path = self.create_model_package(model_path, package_name)
                if archive_path:
                    packages_created.append(archive_path)
            else:
                print(f"⚠️  Model not found: {model_path}")
        
        # Create master manifest
        manifest = {
            "created": datetime.now().isoformat(),
            "total_packages": len(packages_created),
            "packages": []
        }
        
        for package_path in packages_created:
            manifest["packages"].append({
                "name": package_path.stem,
                "file": package_path.name,
                "size_mb": round(package_path.stat().st_size / (1024*1024), 2)
            })
        
        with open(self.output_dir / "manifest.json", 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Create download script
        download_script = f'''#!/bin/bash
# Medical Embedding Models Download Script
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

echo "Medical Embedding Thesis - Model Download"
echo "========================================="

'''
        
        for package in manifest["packages"]:
            download_script += f'''
echo "Extracting {package['name']}..."
tar -xzf {package['file']}
echo "✓ {package['name']} extracted ({package['size_mb']} MB)"
'''
        
        download_script += '''
echo "========================================="
echo "All models extracted successfully!"
echo "See individual model directories for usage instructions."
'''
        
        with open(self.output_dir / "extract_models.sh", 'w') as f:
            f.write(download_script)
        
        os.chmod(self.output_dir / "extract_models.sh", 0o755)
        
        print(f"\n{'='*60}")
        print(f"Packaging Complete!")
        print(f"{'='*60}")
        print(f"Output directory: {self.output_dir}")
        print(f"Total packages: {len(packages_created)}")
        
        total_size = sum(p.stat().st_size for p in packages_created) / (1024*1024)
        print(f"Total size: {total_size:.1f} MB")
        
        print(f"\nTo download and extract:")
        print(f"1. Copy entire '{self.output_dir}' directory")
        print(f"2. Run: ./extract_models.sh")

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Package trained models for distribution')
    parser.add_argument('--models-dir', default='models', 
                       help='Directory containing trained models')
    parser.add_argument('--output-dir', default='packaged_models',
                       help='Output directory for packages')
    
    args = parser.parse_args()
    
    packager = ModelPackager(args.models_dir, args.output_dir)
    packager.package_all_models()

if __name__ == "__main__":
    main()