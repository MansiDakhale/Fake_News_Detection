#!/usr/bin/env python3
"""
ViLT-based Fake News Detection Training Script with WandB Integration and Early Stopping
Enhanced with comprehensive overfitting prevention techniques
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import wandb
from tqdm import tqdm
import warnings
import random
from PIL import Image, ImageEnhance, ImageFilter
import torchvision.transforms as transforms
warnings.filterwarnings('ignore')

# Import your custom modules
from transformers import ViltProcessor
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

# ==================== IMAGE AUGMENTATION ====================

class ImageAugmentation:
    """Image augmentation techniques for regularization"""
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, image):
        if random.random() > self.p:
            return image
            
        # Random augmentation selection
        augmentation_type = random.choice(['brightness', 'contrast', 'blur', 'rotate', 'flip'])
        
        if augmentation_type == 'brightness':
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(random.uniform(0.8, 1.2))
        elif augmentation_type == 'contrast':
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(random.uniform(0.8, 1.2))
        elif augmentation_type == 'blur':
            if random.random() < 0.3:
                image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.0)))
        elif augmentation_type == 'rotate':
            angle = random.uniform(-10, 10)
            image = image.rotate(angle, fillcolor=(255, 255, 255))
        elif augmentation_type == 'flip':
            if random.random() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                
        return image


# ==================== YOUR EXISTING CLASSES WITH ENHANCEMENTS ====================

class ViLTDataset:
    """Your existing dataset class with fixes and augmentation"""
    def __init__(self, df, image_dir, processor, max_length=80, is_training=False, 
                 use_augmentation=True, augmentation_prob=0.5):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.processor = processor
        self.max_length = max_length  # Increased default
        self.is_training = is_training
        self.use_augmentation = use_augmentation and is_training
        
        # Image augmentation
        self.augmentation = ImageAugmentation(p=augmentation_prob) if self.use_augmentation else None
        
        # Auto-detect columns
        self.image_path_col = self._detect_image_path_column()
        self.text_cols = self._detect_text_columns()
        self.label_col = self._detect_label_column()
        
        print(f"Using '{self.image_path_col}' as image path column")
        print(f"Using text columns: {self.text_cols}")
        print(f"Using '{self.label_col}' as label column")
        print(f"Using max_length: {self.max_length}")
        print(f"Augmentation enabled: {self.use_augmentation}")
        
    def _detect_image_path_column(self):
        candidates = ['local_image_path', 'image_path', 'image_file', 'filename', 
                     'image_name', 'image', 'file_path', 'path']
        for candidate in candidates:
            if candidate in self.df.columns:
                return candidate
        for col in self.df.columns:
            if 'image' in col.lower() or 'path' in col.lower():
                return col
        raise ValueError(f"No image path column found. Available columns: {list(self.df.columns)}")
    
    def _detect_text_columns(self):
        candidates = [['title', 'text'], ['headline', 'content'], ['title', 'body'],
                     ['text'], ['content'], ['title'], ['headline'], ['body']]
        for candidate_set in candidates:
            if all(col in self.df.columns for col in candidate_set):
                return candidate_set
        text_cols = []
        for col in self.df.columns:
            if any(keyword in col.lower() for keyword in ['text', 'title', 'content', 'headline', 'body']):
                text_cols.append(col)
        if text_cols:
            return text_cols[:2]
        raise ValueError(f"No text columns found. Available columns: {list(self.df.columns)}")
    
    def _detect_label_column(self):
        candidates = ['label', 'class', 'target', 'y', 'fake', 'real']
        for candidate in candidates:
            if candidate in self.df.columns:
                return candidate
        for col in self.df.columns:
            if 'label' in col.lower() or 'class' in col.lower():
                return col
        raise ValueError(f"No label column found. Available columns: {list(self.df.columns)}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Get image with augmentation
        try:
            image_path = os.path.join(self.image_dir, row[self.image_path_col])
            if not os.path.exists(image_path):
                image_path = row[self.image_path_col]
            if not os.path.exists(image_path):
                image = Image.new('RGB', (384, 384), color='white')
                print(f"Warning: Image not found: {image_path}, using placeholder")
            else:
                image = Image.open(image_path).convert('RGB')
                image = image.resize((384, 384), Image.Resampling.LANCZOS)
                
                # Apply augmentation if training
                if self.augmentation:
                    image = self.augmentation(image)
                    
        except Exception as e:
            print(f"Error loading image at index {idx}: {e}")
            image = Image.new('RGB', (384, 384), color='white')
        
        # Get text
        texts = []
        for col in self.text_cols:
            text = str(row[col]) if pd.notna(row[col]) else ""
            texts.append(text)
        combined_text = " ".join(texts).strip()
        if not combined_text:
            combined_text = "No text available"
        
        # Get label
        try:
            label = int(row[self.label_col])
        except (ValueError, KeyError):
            label_val = row[self.label_col]
            if isinstance(label_val, str):
                label_val = label_val.lower()
                if label_val in ['fake', 'false', '1', 'yes']:
                    label = 1
                else:
                    label = 0
            else:
                label = int(label_val) if pd.notna(label_val) else 0
        
        # Process with ViLT processor
        try:
            encoding = self.processor(
                images=image,
                text=combined_text,
                padding="max_length", 
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            # Remove batch dimension
            for key in encoding:
                if encoding[key].dim() > 1:
                    encoding[key] = encoding[key].squeeze(0)
            
            encoding['labels'] = torch.tensor(label, dtype=torch.long)
            return encoding
        
        except Exception as e:
            print(f"Error processing item at index {idx}: {e}")
            return {
                'input_ids': torch.zeros(self.max_length, dtype=torch.long),
                'attention_mask': torch.zeros(self.max_length, dtype=torch.long),
                'token_type_ids': torch.zeros(self.max_length, dtype=torch.long),
                'pixel_values': torch.zeros(3, 384, 384, dtype=torch.float),
                'pixel_mask': torch.ones(384, 384, dtype=torch.long),
                'labels': torch.tensor(label, dtype=torch.long)
            }
    
    def get_column_info(self):
        return {
            'image_path_column': self.image_path_col,
            'text_columns': self.text_cols,
            'label_column': self.label_col,
            'total_columns': list(self.df.columns)
        }


class ViltWithGroundingFixed(nn.Module):
    """Your existing model with enhanced regularization"""
    def __init__(self, model_name="dandelin/vilt-b32-mlm", num_labels=2, dropout_rate=0.3):
        super().__init__()
        from transformers import ViltModel, ViltConfig
        
        self.num_labels = num_labels
        config = ViltConfig.from_pretrained(model_name)
        
        # Add dropout to the base model config
        config.hidden_dropout_prob = dropout_rate
        config.attention_probs_dropout_prob = dropout_rate
        
        self.vilt = ViltModel.from_pretrained(model_name, config=config)
        
        # Fix buffer size issue with increased max length
        max_seq_len = 80  # Increased from 40
        self.vilt.embeddings.text_embeddings.register_buffer(
            "position_ids", 
            torch.arange(max_seq_len).expand((1, -1)),
            persistent=False
        )
        
        if hasattr(self.vilt.embeddings.text_embeddings, 'token_type_ids'):
            self.vilt.embeddings.text_embeddings.register_buffer(
                "token_type_ids",
                torch.zeros(1, max_seq_len, dtype=torch.long),
                persistent=False
            )
        
        # Enhanced classification head with more regularization
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.vilt.config.hidden_size, 512),
            nn.BatchNorm1d(512),  # Batch normalization
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),   # Additional layer
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),  # Reduced dropout for final layer
            nn.Linear(256, num_labels)
        )
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization"""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, input_ids, attention_mask, pixel_values, pixel_mask=None):
        max_len = 80  # Increased from 40
        if input_ids.size(-1) > max_len:
            input_ids = input_ids[:, :max_len]
            attention_mask = attention_mask[:, :max_len]
        
        outputs = self.vilt(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            return_dict=True
        )
        
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return {'logits': logits}


def vilt_collate_fn(batch):
    """Your existing collate function with enhanced error handling"""
    input_ids = []
    attention_masks = []
    token_type_ids = []
    pixel_values = []
    pixel_masks = []
    labels = []
    
    for item in batch:
        input_ids.append(item['input_ids'])
        attention_masks.append(item['attention_mask'])
        token_type_ids.append(item['token_type_ids'])
        pixel_values.append(item['pixel_values'])
        pixel_masks.append(item['pixel_mask'])
        labels.append(item['labels'])
    
    try:
        input_ids = torch.stack(input_ids)
        attention_masks = torch.stack(attention_masks)
        token_type_ids = torch.stack(token_type_ids)
        labels = torch.stack(labels)
    except RuntimeError as e:
        print(f"Error stacking text tensors: {e}")
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
        token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=0)
        labels = torch.stack(labels)
    
    target_size = (3, 384, 384)
    processed_pixel_values = []
    processed_pixel_masks = []
    
    for pv, pm in zip(pixel_values, pixel_masks):
        if pv.shape != target_size:
            if len(pv.shape) == 3:
                pv = F.interpolate(pv.unsqueeze(0), size=(384, 384), mode='bilinear', align_corners=False).squeeze(0)
            else:
                pv = torch.zeros(target_size)
        
        target_mask_size = (384, 384)
        if pm.shape != target_mask_size:
            if len(pm.shape) == 2:
                pm = F.interpolate(pm.unsqueeze(0).unsqueeze(0).float(), 
                                 size=(384, 384), mode='nearest').squeeze().long()
            else:
                pm = torch.ones(target_mask_size, dtype=torch.long)
        
        processed_pixel_values.append(pv)
        processed_pixel_masks.append(pm)
    
    try:
        pixel_values = torch.stack(processed_pixel_values)
        pixel_masks = torch.stack(processed_pixel_masks)
    except RuntimeError as e:
        print(f"Error stacking image tensors: {e}")
        batch_size = len(processed_pixel_values)
        pixel_values = torch.zeros(batch_size, 3, 384, 384)
        pixel_masks = torch.ones(batch_size, 384, 384, dtype=torch.long)
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_masks,
        'token_type_ids': token_type_ids,
        'pixel_values': pixel_values,
        'pixel_mask': pixel_masks,
        'labels': labels
    }


# ==================== TRAINING UTILITIES ====================

class EarlyStopping:
    """Enhanced early stopping utility with multiple metrics"""
    def __init__(self, patience=7, min_delta=0.001, restore_best_weights=True, 
                 monitor='val_loss', mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.monitor = monitor
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, current_score, model):
        if self.best_score is None:
            self.best_score = current_score
            self.save_checkpoint(model)
        elif self._is_better(current_score, self.best_score):
            self.best_score = current_score
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def _is_better(self, current, best):
        if self.mode == 'min':
            return current < best - self.min_delta
        else:
            return current > best + self.min_delta
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()


class LabelSmoothing(nn.Module):
    """Label smoothing for regularization"""
    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        
    def forward(self, pred, target):
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * F.log_softmax(pred, dim=1), dim=1))


def compute_metrics(predictions, labels):
    """Compute evaluation metrics"""
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def evaluate_model(model, dataloader, criterion, device, add_noise=False, noise_std=0.01):
    """Evaluate model on validation/test set with optional noise injection"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            pixel_mask = batch['pixel_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Add noise to pixel values if specified (for validation robustness)
            if add_noise:
                noise = torch.randn_like(pixel_values) * noise_std
                pixel_values = pixel_values + noise
            
            # Forward pass
            outputs = model(input_ids, attention_mask, pixel_values, pixel_mask)
            logits = outputs['logits']
            
            # Calculate loss
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            # Store predictions and labels
            all_predictions.extend(logits.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(np.array(all_predictions), np.array(all_labels))
    
    return avg_loss, metrics


# ==================== MAIN TRAINING FUNCTION ====================

def train_vilt_model(
    train_df_path,
    val_df_path,
    test_df_path,
    image_dir,
    model_name="dandelin/vilt-b32-mlm",
    batch_size=8,
    learning_rate=1e-5,  # Reduced learning rate
    num_epochs=20,
    patience=10,  # Increased patience
    max_length=80,  # Increased max_length
    dropout_rate=0.3,  # Increased dropout
    weight_decay=0.05,  # Increased weight decay
    warmup_steps=100,
    label_smoothing=0.1,  # Label smoothing
    gradient_clip_value=1.0,
    use_cosine_scheduler=True,  # Cosine annealing
    augmentation_prob=0.5,  # Image augmentation probability
    validation_noise=True,  # Add noise during validation
    project_name="vilt-fake-news-detection",
    run_name=None
):
    """
    Main training function with WandB integration and comprehensive overfitting prevention
    """
    
    # Initialize WandB
    wandb.init(
        project=project_name,
        name=run_name,
        config={
            "model_name": model_name,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "patience": patience,
            "max_length": max_length,
            "dropout_rate": dropout_rate,
            "weight_decay": weight_decay,
            "warmup_steps": warmup_steps,
            "label_smoothing": label_smoothing,
            "gradient_clip_value": gradient_clip_value,
            "use_cosine_scheduler": use_cosine_scheduler,
            "augmentation_prob": augmentation_prob,
            "validation_noise": validation_noise
        }
    )
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    train_df = pd.read_csv(train_df_path)
    val_df = pd.read_csv(val_df_path)
    test_df = pd.read_csv(test_df_path)
    
    print(f"Train size: {len(train_df)}")
    print(f"Validation size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")
    
    # Initialize processor
    processor = ViltProcessor.from_pretrained(model_name)
    
    # Create datasets with augmentation
    train_dataset = ViLTDataset(
        train_df, image_dir, processor, max_length, 
        is_training=True, use_augmentation=True, augmentation_prob=augmentation_prob
    )
    val_dataset = ViLTDataset(
        val_df, image_dir, processor, max_length, 
        is_training=False, use_augmentation=False
    )
    test_dataset = ViLTDataset(
        test_df, image_dir, processor, max_length, 
        is_training=False, use_augmentation=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=vilt_collate_fn,
        num_workers=0,
        drop_last=True  # Drop last incomplete batch
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=vilt_collate_fn,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=vilt_collate_fn,
        num_workers=0
    )
    
    # Initialize model
    print("Initializing model...")
    model = ViltWithGroundingFixed(model_name, num_labels=2, dropout_rate=dropout_rate)
    model.to(device)
    
    # Log model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Initialize optimizer with different learning rates for different parts
    optimizer_params = [
        {'params': model.vilt.parameters(), 'lr': learning_rate * 0.1},  # Lower LR for pretrained
        {'params': model.classifier.parameters(), 'lr': learning_rate}   # Higher LR for new layers
    ]
    optimizer = optim.AdamW(optimizer_params, weight_decay=weight_decay)
    
    # Enhanced scheduler
    total_steps = len(train_loader) * num_epochs
    if use_cosine_scheduler:
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
        )
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps - warmup_steps, eta_min=learning_rate * 0.01
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps]
        )
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
    
    # Loss function with label smoothing
    if label_smoothing > 0:
        criterion = LabelSmoothing(num_classes=2, smoothing=label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=patience, min_delta=0.001, monitor='val_loss', mode='min'
    )
    
    # Training loop
    print("Starting training...")
    best_val_accuracy = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        # Training phase
        model.train()
        total_train_loss = 0
        train_predictions = []
        train_labels = []
        
        train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        for batch_idx, batch in enumerate(train_pbar):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            pixel_mask = batch['pixel_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(input_ids, attention_mask, pixel_values, pixel_mask)
            logits = outputs['logits']
            
            # Calculate loss
            loss = criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_value)
            
            # Update weights
            optimizer.step()
            if use_cosine_scheduler:
                scheduler.step()
            
            # Store metrics
            total_train_loss += loss.item()
            train_predictions.extend(logits.detach().cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Log batch metrics to WandB
            if batch_idx % 50 == 0:
                wandb.log({
                    "batch_loss": loss.item(),
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    "epoch": epoch,
                    "batch": batch_idx
                })
        
        # Calculate training metrics
        avg_train_loss = total_train_loss / len(train_loader)
        train_metrics = compute_metrics(np.array(train_predictions), np.array(train_labels))
        train_losses.append(avg_train_loss)
        
        # Validation phase
        print("Evaluating on validation set...")
        val_loss, val_metrics = evaluate_model(
            model, val_loader, criterion, device, 
            add_noise=validation_noise, noise_std=0.01
        )
        val_losses.append(val_loss)
        
        # Update scheduler if using ReduceLROnPlateau
        if not use_cosine_scheduler:
            scheduler.step(val_loss)
        
        # Print epoch results
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
        
        # Calculate overfitting metric
        if len(train_losses) > 1 and len(val_losses) > 1:
            overfitting_ratio = val_loss / avg_train_loss
            print(f"Overfitting Ratio (Val/Train Loss): {overfitting_ratio:.3f}")
        
        # Log to WandB
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "train_accuracy": train_metrics['accuracy'],
            "train_f1": train_metrics['f1'],
            "val_loss": val_loss,
            "val_accuracy": val_metrics['accuracy'],
            "val_f1": val_metrics['f1'],
            "val_precision": val_metrics['precision'],
            "val_recall": val_metrics['recall'],
            "overfitting_ratio": val_loss / avg_train_loss if avg_train_loss > 0 else 1.0
        })
        
        # Save best model
        if val_metrics['accuracy'] > best_val_accuracy:
            best_val_accuracy = val_metrics['accuracy']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_metrics['accuracy'],
                'val_loss': val_loss,
                'config': wandb.config
            }, 'best_vilt_model.pth')
            print(f"New best model saved with validation accuracy: {val_metrics['accuracy']:.4f}")
        
        # Early stopping check
        if early_stopping(val_loss, model):
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Test evaluation
    print("\nEvaluating on test set...")
    test_loss, test_metrics = evaluate_model(model, test_loader, criterion, device)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test F1: {test_metrics['f1']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall: {test_metrics['recall']:.4f}")
    
    # Log final test metrics
    wandb.log({
        "test_loss": test_loss,
        "test_accuracy": test_metrics['accuracy'],
        "test_f1": test_metrics['f1'],
        "test_precision": test_metrics['precision'],
        "test_recall": test_metrics['recall']
    })
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_metrics': test_metrics,
        'config': wandb.config
    }, 'final_vilt_model.pth')
    
    wandb.finish()
    
    return model, test_metrics


# ==================== ENSEMBLE TRAINING ====================

def train_ensemble_models(
    train_df_path,
    val_df_path,
    test_df_path,
    image_dir,
    n_models=3,
    base_config=None
):
    """
    Train an ensemble of models with different random seeds for better generalization
    """
    if base_config is None:
        base_config = {
            "model_name": "dandelin/vilt-b32-mlm",
            "batch_size": 8,
            "learning_rate": 1e-5,
            "num_epochs": 15,
            "patience": 8,
            "max_length": 80,
            "dropout_rate": 0.3,
            "weight_decay": 0.05,
            "warmup_steps": 100,
            "label_smoothing": 0.1,
            "project_name": "vilt-fake-news-ensemble"
        }
    
    ensemble_results = []
    ensemble_models = []
    
    for i in range(n_models):
        print(f"\n{'='*60}")
        print(f"Training Ensemble Model {i+1}/{n_models}")
        print(f"{'='*60}")
        
        # Set different random seed for each model
        torch.manual_seed(42 + i)
        np.random.seed(42 + i)
        random.seed(42 + i)
        
        # Slightly vary hyperparameters for diversity
        config = base_config.copy()
        config['run_name'] = f"ensemble_model_{i+1}"
        config['dropout_rate'] = base_config['dropout_rate'] + np.random.uniform(-0.05, 0.05)
        config['learning_rate'] = base_config['learning_rate'] * np.random.uniform(0.8, 1.2)
        config['augmentation_prob'] = 0.4 + np.random.uniform(0, 0.2)
        
        # Train individual model
        model, test_metrics = train_vilt_model(
            train_df_path, val_df_path, test_df_path, image_dir, **config
        )
        
        ensemble_models.append(model)
        ensemble_results.append(test_metrics)
        
        # Save individual ensemble model
        torch.save({
            'model_state_dict': model.state_dict(),
            'test_metrics': test_metrics,
            'config': config
        }, f'ensemble_model_{i+1}.pth')
    
    # Print ensemble summary
    print(f"\n{'='*60}")
    print("ENSEMBLE TRAINING SUMMARY")
    print(f"{'='*60}")
    
    accuracies = [result['accuracy'] for result in ensemble_results]
    f1_scores = [result['f1'] for result in ensemble_results]
    
    print(f"Individual Model Accuracies: {[f'{acc:.4f}' for acc in accuracies]}")
    print(f"Mean Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"Individual Model F1 Scores: {[f'{f1:.4f}' for f1 in f1_scores]}")
    print(f"Mean F1 Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
    
    return ensemble_models, ensemble_results


def evaluate_ensemble(ensemble_models, test_loader, device):
    """
    Evaluate ensemble of models using voting
    """
    all_predictions = []
    all_labels = []
    
    # Set all models to eval mode
    for model in ensemble_models:
        model.eval()
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Ensemble Evaluation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            pixel_mask = batch['pixel_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Get predictions from all models
            batch_predictions = []
            for model in ensemble_models:
                outputs = model(input_ids, attention_mask, pixel_values, pixel_mask)
                logits = outputs['logits']
                probs = F.softmax(logits, dim=1)
                batch_predictions.append(probs.cpu().numpy())
            
            # Average predictions
            ensemble_probs = np.mean(batch_predictions, axis=0)
            all_predictions.extend(ensemble_probs)
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate ensemble metrics
    ensemble_metrics = compute_metrics(np.array(all_predictions), np.array(all_labels))
    return ensemble_metrics


# ==================== K-FOLD CROSS VALIDATION ====================

def k_fold_cross_validation(
    full_df_path,
    image_dir,
    k=5,
    base_config=None
):
    """
    Perform k-fold cross validation for more robust evaluation
    """
    from sklearn.model_selection import StratifiedKFold
    
    if base_config is None:
        base_config = {
            "model_name": "dandelin/vilt-b32-mlm",
            "batch_size": 8,
            "learning_rate": 1e-5,
            "num_epochs": 10,
            "patience": 5,
            "max_length": 80,
            "dropout_rate": 0.3,
            "weight_decay": 0.05,
            "project_name": "vilt-fake-news-kfold"
        }
    
    # Load full dataset
    full_df = pd.read_csv(full_df_path)
    
    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(full_df, full_df['label'])):
        print(f"\n{'='*60}")
        print(f"K-Fold Cross Validation - Fold {fold+1}/{k}")
        print(f"{'='*60}")
        
        # Split data for this fold
        train_fold_df = full_df.iloc[train_idx].reset_index(drop=True)
        val_fold_df = full_df.iloc[val_idx].reset_index(drop=True)
        
        # Save temporary CSV files
        train_fold_path = f'temp_train_fold_{fold}.csv'
        val_fold_path = f'temp_val_fold_{fold}.csv'
        
        train_fold_df.to_csv(train_fold_path, index=False)
        val_fold_df.to_csv(val_fold_path, index=False)
        
        # Update config for this fold
        fold_config = base_config.copy()
        fold_config['run_name'] = f"kfold_fold_{fold+1}"
        
        try:
            # Train model for this fold
            model, fold_metrics = train_vilt_model(
                train_fold_path, val_fold_path, val_fold_path, image_dir, **fold_config
            )
            
            fold_results.append(fold_metrics)
            
            # Save fold model
            torch.save({
                'model_state_dict': model.state_dict(),
                'fold_metrics': fold_metrics,
                'fold': fold
            }, f'kfold_model_fold_{fold+1}.pth')
            
        except Exception as e:
            print(f"Error in fold {fold+1}: {e}")
            continue
        
        finally:
            # Clean up temporary files
            if os.path.exists(train_fold_path):
                os.remove(train_fold_path)
            if os.path.exists(val_fold_path):
                os.remove(val_fold_path)
    
    # Calculate cross-validation statistics
    if fold_results:
        accuracies = [result['accuracy'] for result in fold_results]
        f1_scores = [result['f1'] for result in fold_results]
        
        print(f"\n{'='*60}")
        print("K-FOLD CROSS VALIDATION RESULTS")
        print(f"{'='*60}")
        print(f"Fold Accuracies: {[f'{acc:.4f}' for acc in accuracies]}")
        print(f"Mean CV Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        print(f"Fold F1 Scores: {[f'{f1:.4f}' for f1 in f1_scores]}")
        print(f"Mean CV F1 Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
        
        return fold_results
    else:
        print("No successful folds completed!")
        return []


# ==================== USAGE EXAMPLE ====================

if __name__ == "__main__":
    # Enhanced configuration with overfitting prevention
    config = {
        "train_df_path": "data/Preprocessed/train.csv",  # Update these paths
        "val_df_path": "data/Preprocessed/val.csv",
        "test_df_path": "data/Preprocessed/test.csv",
        "image_dir": "data/images",
        "model_name": "dandelin/vilt-b32-mlm",
        "batch_size": 8,
        "learning_rate": 1e-5,  # Reduced from 2e-5
        "num_epochs": 20,
        "patience": 10,  # Increased patience
        "max_length": 80,  # Increased from 40
        "dropout_rate": 0.3,  # Increased from 0.1
        "weight_decay": 0.05,  # Increased from 0.01
        "warmup_steps": 100,
        "label_smoothing": 0.1,  # Added label smoothing
        "gradient_clip_value": 1.0,
        "use_cosine_scheduler": True,  # Added cosine annealing
        "augmentation_prob": 0.5,  # Image augmentation
        "validation_noise": True,  # Validation with noise
        "project_name": "vilt-fake-news-detection-enhanced",
        "run_name": "vilt-overfitting-prevention-v1"
    }
    
    print("🚀 Starting Enhanced ViLT Training with Overfitting Prevention")
    print("="*70)
    
    # Train single model
    try:
        model, test_metrics = train_vilt_model(**config)
        print("\n🎉 Single model training completed successfully!")
        print(f"Final test accuracy: {test_metrics['accuracy']:.4f}")
        
        # Optional: Train ensemble for even better performance
        print("\n" + "="*70)
        print("🔄 Starting Ensemble Training (Optional)")
        print("="*70)
        
        ensemble_config = config.copy()
        ensemble_config['num_epochs'] = 15  # Reduce epochs for ensemble
        ensemble_config['patience'] = 8
        
        ensemble_models, ensemble_results = train_ensemble_models(
            config['train_df_path'],
            config['val_df_path'],
            config['test_df_path'],
            config['image_dir'],
            n_models=3,
            base_config=ensemble_config
        )
        
        print("\n🎉 Ensemble training completed successfully!")
        
        # Optional: K-Fold Cross Validation (comment out if not needed)
        """
        print("\n" + "="*70)
        print("📊 Starting K-Fold Cross Validation (Optional)")
        print("="*70)
        
        kfold_config = config.copy()
        kfold_config['num_epochs'] = 8  # Reduce epochs for k-fold
        kfold_config['patience'] = 4
        
        cv_results = k_fold_cross_validation(
            "data/Preprocessed/full_dataset.csv",  # Path to full dataset
            config['image_dir'],
            k=5,
            base_config=kfold_config
        )
        """
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("✅ All training procedures completed!")
    print("="*70)