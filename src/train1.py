#train1.py
"""
Training script for fake news detection using existing test_data.py and Vilt.py
This script reuses your working data loading pipeline without modifications
Now includes Weights & Biases (wandb) integration for experiment tracking
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import wandb
from datetime import datetime

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent))

# Import your existing working modules
from src.preprocessing.Vilt import ViLTDataset
from src.preprocessing.collate_functions import vilt_collate_fn  # Import from your separate file
from transformers import ViltProcessor, ViltModel, ViltConfig
from torch.utils.data import DataLoader


# Use your existing data loading logic from test_data.py
def load_data_using_existing_pipeline():
    """Load data using the same logic as test_data.py"""
    print("Loading pre-split data...")
    
    # Check if split data files exist - same logic as test_data.py
    data_dir = Path("data")
    train_file = data_dir / r"C:\Users\OMEN\Desktop\Fake_News_Detection\data\Preprocessed\train.csv"
    val_file = data_dir / r"C:\Users\OMEN\Desktop\Fake_News_Detection\data\Preprocessed\val.csv" 
    test_file = data_dir / r"C:\Users\OMEN\Desktop\Fake_News_Detection\data\Preprocessed\test.csv"
    
    # Alternative file names to check - same as test_data.py
    alt_files = [
        (data_dir / "train_data.csv", data_dir / "val_data.csv", data_dir / "test_data.csv"),
        (data_dir / "train_split.csv", data_dir / "val_split.csv", data_dir / "test_split.csv"),
    ]
    
    # Load the split data - same logic as test_data.py
    train_df = val_df = test_df = None
    
    if all(f.exists() for f in [train_file, val_file, test_file]):
        print("✓ Found standard split files (train.csv, validation.csv, test.csv)")
        train_df = pd.read_csv(train_file)
        val_df = pd.read_csv(val_file)
        test_df = pd.read_csv(test_file)
    else:
        # Check alternative file names
        found = False
        for train_alt, val_alt, test_alt in alt_files:
            if all(f.exists() for f in [train_alt, val_alt, test_alt]):
                print(f"✓ Found split files: {train_alt.name}, {val_alt.name}, {test_alt.name}")
                train_df = pd.read_csv(train_alt)
                val_df = pd.read_csv(val_alt)
                test_df = pd.read_csv(test_alt)
                found = True
                break
    
    if train_df is None:
        raise FileNotFoundError("No split data files found. Please ensure you have the CSV files.")
                                                                                
    print(f"✓ Data loaded successfully")
    print(f"  Train size: {len(train_df)}")
    print(f"  Validation size: {len(val_df)}")
    print(f"  Test size: {len(test_df)}")
    
    return train_df, val_df, test_df


def create_datasets_and_loaders_using_existing():
    """Create datasets and loaders using your existing working code"""
    # Load data using existing pipeline
    train_df, val_df, test_df = load_data_using_existing_pipeline()
    
    # Initialize processor - same as test_data.py
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
    
    # Create datasets using your existing ViLTDataset class
    image_dir = "data/images"
    train_dataset = ViLTDataset(train_df, image_dir, processor)
    val_dataset = ViLTDataset(val_df, image_dir, processor)
    test_dataset = ViLTDataset(test_df, image_dir, processor)
    
    print(f"✓ Datasets created successfully")
    print(f"  Train dataset size: {len(train_dataset)}")
    print(f"  Validation dataset size: {len(val_dataset)}")
    print(f"  Test dataset size: {len(test_dataset)}")
    
    # Print column information - same as test_data.py
    col_info = train_dataset.get_column_info()
    print(f"\n📊 Dataset Column Information:")
    print(f"  Image path column: {col_info['image_path_column']}")
    print(f"  Text columns: {col_info['text_columns']}")
    print(f"  Label column: {col_info['label_column']}")
    
    # Create data loaders with your existing collate function
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, 
                            collate_fn=vilt_collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, 
                          collate_fn=vilt_collate_fn, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, 
                           collate_fn=vilt_collate_fn, num_workers=0)
    
    print(f"✓ DataLoaders created successfully")
    
    return train_loader, val_loader, test_loader


class ViltFakeNewsModel(nn.Module):
    """Simple ViLT model for fake news detection"""
    def __init__(self, model_name="dandelin/vilt-b32-mlm", num_labels=2, dropout_rate=0.1):
        super().__init__()
        
        self.num_labels = num_labels
        
        # Load ViLT model
        self.vilt = ViltModel.from_pretrained(model_name)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.vilt.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_labels)
        )
    
    def forward(self, input_ids, attention_mask, pixel_values, pixel_mask=None, **kwargs):
        # Forward pass through ViLT
        outputs = self.vilt(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            return_dict=True
        )
        
        # Get pooled output and classify
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        
        return {'logits': logits}


class FakeNewsTrainer:
    """Training class that uses your existing data pipeline with WandB integration"""
    
    def __init__(self, config, use_wandb=True):
        self.config = config
        self.use_wandb = use_wandb
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize WandB
        if self.use_wandb:
            self.init_wandb()
        
        # Initialize model
        self.model = ViltFakeNewsModel(
            num_labels=config['num_labels'],
            dropout_rate=config['dropout_rate']
        ).to(self.device)
        
        # Log model architecture to wandb
        if self.use_wandb:
            wandb.watch(self.model, log="all", log_freq=100)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config['num_epochs']
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        print(f"✓ Trainer initialized on device: {self.device}")
        if self.use_wandb:
            print(f"✓ WandB initialized - tracking at: {wandb.run.url}")
    
    def init_wandb(self):
        """Initialize Weights & Biases"""
        # Create a run name with timestamp
        run_name = f"vilt-fakenews-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # Initialize wandb
        wandb.init(
            project="fake-news-detection",
            name=run_name,
            config=self.config,
            tags=["vilt", "multimodal", "fake-news"],
            notes="ViLT-based fake news detection with multimodal input"
        )
        
        # Log additional info
        wandb.config.update({
            "model_architecture": "ViLT + Classification Head",
            "model_name": "dandelin/vilt-b32-mlm",
            "device": str(self.device),
            "pytorch_version": torch.__version__,
        })
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            pixel_values = batch['pixel_values'].to(self.device)
            pixel_mask = batch['pixel_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                pixel_mask=pixel_mask
            )
            
            # Calculate loss
            loss = self.criterion(outputs['logits'], labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            predictions = torch.argmax(outputs['logits'], dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Log to WandB every 10 batches
            if self.use_wandb and batch_idx % 10 == 0:
                wandb.log({
                    "batch/train_loss": loss.item(),
                    "batch/learning_rate": self.optimizer.param_groups[0]['lr'],
                    "batch/epoch": epoch + batch_idx / len(train_loader),
                })
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                pixel_values = batch['pixel_values'].to(self.device)
                pixel_mask = batch['pixel_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    pixel_mask=pixel_mask
                )
                
                # Calculate loss
                loss = self.criterion(outputs['logits'], labels)
                total_loss += loss.item()
                
                # Predictions
                predictions = torch.argmax(outputs['logits'], dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        return avg_loss, accuracy, precision, recall, f1
    
    def train(self, train_loader, val_loader):
        """Main training loop"""
        print(f"🚀 Starting training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        if self.use_wandb:
            # Log model info
            wandb.config.update({
                "total_parameters": sum(p.numel() for p in self.model.parameters()),
                "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            })
        
        best_val_accuracy = 0
        best_model_state = None
        
        for epoch in range(self.config['num_epochs']):
            print(f"\nEpoch {epoch + 1}/{self.config['num_epochs']}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_acc, val_precision, val_recall, val_f1 = self.validate_epoch(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Save metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Log to WandB
            if self.use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "train/loss": train_loss,
                    "train/accuracy": train_acc,
                    "val/loss": val_loss,
                    "val/accuracy": val_acc,
                    "val/precision": val_precision,
                    "val/recall": val_recall,
                    "val/f1": val_f1,
                    "learning_rate": self.optimizer.param_groups[0]['lr'],
                })
            
            # Print metrics
            print(f"📈 Training   - Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f}")
            print(f"📊 Validation - Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f}")
            print(f"📋 Metrics   - Precision: {val_precision:.4f} | Recall: {val_recall:.4f} | F1: {val_f1:.4f}")
            print(f"🔧 Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Save best model
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                best_model_state = self.model.state_dict().copy()
                self.save_checkpoint(f"best_model_epoch_{epoch+1}.pth", epoch)
                print(f"🏆 New best model! Validation accuracy: {best_val_accuracy:.4f}")
                
                # Log best model to WandB
                if self.use_wandb:
                    wandb.run.summary["best_val_accuracy"] = best_val_accuracy
                    wandb.run.summary["best_epoch"] = epoch + 1
            
            # Save regular checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pth", epoch)
        
        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
            print(f"\n✅ Loaded best model with validation accuracy: {best_val_accuracy:.4f}")
        
        return self.model
    
    def evaluate(self, test_loader):
        """Evaluate model on test set"""
        print("\n🧪 Evaluating on test set...")
        test_loss, test_acc, test_precision, test_recall, test_f1 = self.validate_epoch(test_loader)
        
        # Get predictions for confusion matrix
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                pixel_values = batch['pixel_values'].to(self.device)
                pixel_mask = batch['pixel_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    pixel_mask=pixel_mask
                )
                
                # Get probabilities and predictions
                probabilities = torch.softmax(outputs['logits'], dim=-1)
                predictions = torch.argmax(outputs['logits'], dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        # Log to WandB
        if self.use_wandb:
            wandb.log({
                "test/loss": test_loss,
                "test/accuracy": test_acc,
                "test/precision": test_precision,
                "test/recall": test_recall,
                "test/f1": test_f1,
            })
            
            # Log confusion matrix
            wandb.log({
                "test/confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=all_labels,
                    preds=all_predictions,
                    class_names=["Real", "Fake"]
                )
            })
            
            # Create a custom confusion matrix plot
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Real', 'Fake'], 
                       yticklabels=['Real', 'Fake'])
            plt.title('Confusion Matrix - Test Set')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            # Log the plot
            wandb.log({"test/confusion_matrix_heatmap": wandb.Image(plt)})
            plt.close()
        
        print(f"\n📊 Final Test Results:")
        print(f"=" * 50)
        print(f"Test Loss:      {test_loss:.4f}")
        print(f"Test Accuracy:  {test_acc:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall:    {test_recall:.4f}")
        print(f"Test F1 Score:  {test_f1:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"                Predicted")
        print(f"              Real  Fake")
        print(f"Actual Real   {cm[0,0]:4d}  {cm[0,1]:4d}")
        print(f"       Fake   {cm[1,0]:4d}  {cm[1,1]:4d}")
        
        return {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'confusion_matrix': cm.tolist()
        }
    
    def save_checkpoint(self, filename, epoch):
        """Save model checkpoint"""
        os.makedirs(self.config['save_dir'], exist_ok=True)
        filepath = os.path.join(self.config['save_dir'], filename)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }
        
        torch.save(checkpoint, filepath)
        
        # Save to WandB as well
        if self.use_wandb:
            if wandb.run is not None:
                import shutil

                dest = os.path.join(wandb.run.dir, "checkpoints", os.path.basename(filepath))
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.copy(filepath, dest)
        print(f"✓ Checkpoint saved: {filepath}")


def run_training(use_wandb=True, wandb_project="fake-news-detection"):
    """Main training function"""
    # Training configuration
    config = {
        # Model settings
        'num_labels': 2,
        'dropout_rate': 0.1,
        
        # Training settings
        'num_epochs': 30,
        'batch_size': 4,  # Same as your test_data.py
        'learning_rate': 2e-5,
        'weight_decay': 0.01,
        
        # Paths
        'save_dir': './checkpoints',
        
        # WandB settings
        'use_wandb': use_wandb,
        'wandb_project': wandb_project,
    }
    
    print("=" * 60)
    print("🚀 FAKE NEWS DETECTION - TRAINING PIPELINE WITH WANDB")
    print("=" * 60)
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 60)
    
    try:
        # Test your existing data loading pipeline first
        print("\n🔍 Testing data loading pipeline...")
        train_loader, val_loader, test_loader = create_datasets_and_loaders_using_existing()
        
        # Test loading a batch (same as your test_data.py)
        print("\n🧪 Testing batch loading...")
        try:
            batch = next(iter(train_loader))
            print(f"✅ Batch loaded successfully!")
            print(f"  Batch size: {batch['input_ids'].shape[0]}")
            print(f"  Input IDs shape: {batch['input_ids'].shape}")
            print(f"  Pixel values shape: {batch['pixel_values'].shape}")
            print(f"  Labels shape: {batch['labels'].shape}")
        except Exception as e:
            print(f"❌ Error loading batch: {e}")
            return False
        
        # Create trainer
        print(f"\n🏗️  Initializing trainer...")
        trainer = FakeNewsTrainer(config, use_wandb=use_wandb)
        
        # Start training
        print(f"\n🚀 Starting training...")
        trained_model = trainer.train(train_loader, val_loader)
        
        # Evaluate on test set
        test_results = trainer.evaluate(test_loader)
        
        # Save final results
        os.makedirs(config['save_dir'], exist_ok=True)
        results_file = os.path.join(config['save_dir'], 'training_results.json')
        with open(results_file, 'w') as f:
            json.dump({
                'config': config,
                'test_results': test_results,
                'train_losses': trainer.train_losses,
                'val_losses': trainer.val_losses,
                'train_accuracies': trainer.train_accuracies,
                'val_accuracies': trainer.val_accuracies
            }, f, indent=2)
        
        # Finish WandB run
        if use_wandb:
            # Log final results as summary
            wandb.run.summary.update(test_results)
            wandb.save(results_file)
            wandb.finish()
            print(f"📊 WandB run completed! Check your dashboard for detailed metrics.")
        
        print(f"\n🎉 Training completed successfully!")
        print(f"📁 Results saved to: {results_file}")
        print(f"💾 Model checkpoints saved to: {config['save_dir']}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        # Make sure to finish WandB run even if training fails
        if use_wandb:
            wandb.finish()
        
        return False


if __name__ == "__main__":
    # You can disable WandB by setting use_wandb=False
    success = run_training(use_wandb=True, wandb_project="fake-news-detection")
    if success:
        print(f"\n🎊 All done! Your fake news detection model is ready!")
        print(f"Check your WandB dashboard for detailed training metrics and visualizations!")
    else:
        print(f"\n Training failed. Please check the errors above.")