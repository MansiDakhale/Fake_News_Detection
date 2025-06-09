# Fix 1: Updated Dataset with proper sequence length handling
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from transformers import ViltProcessor
import torchvision.transforms as transforms

class ViLTDataset(Dataset):
    def __init__(self, df, image_dir, processor, max_length=40):  # Changed to 40
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.processor = processor
        self.max_length = max_length  # Use the model's expected length
        
        # Auto-detect the image path column
        self.image_path_col = self._detect_image_path_column()
        print(f"Using '{self.image_path_col}' as image path column")
        
        # Auto-detect text columns
        self.text_cols = self._detect_text_columns()
        print(f"Using text columns: {self.text_cols}")
        
        # Auto-detect label column
        self.label_col = self._detect_label_column()
        print(f"Using '{self.label_col}' as label column")
        
        print(f"Using max_length: {self.max_length}")
        
    def _detect_image_path_column(self):
        """Auto-detect the column containing image paths"""
        candidates = [
            'local_image_path', 'image_path', 'image_file', 'filename', 
            'image_name', 'image', 'file_path', 'path'
        ]
        
        for candidate in candidates:
            if candidate in self.df.columns:
                return candidate
        
        for col in self.df.columns:
            if 'image' in col.lower() or 'path' in col.lower():
                return col
        
        raise ValueError(f"No image path column found. Available columns: {list(self.df.columns)}")
    
    def _detect_text_columns(self):
        """Auto-detect columns containing text data"""
        candidates = [
            ['title', 'text'], ['headline', 'content'], ['title', 'body'],
            ['text'], ['content'], ['title'], ['headline'], ['body']
        ]
        
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
        """Auto-detect the label column"""
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
        
        # Get image
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
        
        # Process with ViLT processor - USE THE MODEL'S EXPECTED LENGTH
        try:
            encoding = self.processor(
                images=image,
                text=combined_text,
                padding="max_length", 
                truncation=True,
                max_length=self.max_length,  # Use 40 instead of 512
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
            # Return dummy with correct dimensions
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


# Fix 2: Updated Model with proper configuration
import torch
import torch.nn as nn
from transformers import ViltModel, ViltConfig, ViltForQuestionAnswering

class ViltWithGroundingFixed(nn.Module):
    """
    ViLT model with proper configuration for the buffer size issue
    """
    def __init__(self, model_name="dandelin/vilt-b32-mlm", num_labels=2, dropout_rate=0.1):
        super().__init__()
        
        self.num_labels = num_labels
        
        # Load and modify config
        config = ViltConfig.from_pretrained(model_name)
        
        # Check the model's actual max sequence length
        print(f"Model config max_position_embeddings: {config.max_position_embeddings}")
        
        # Load the model
        self.vilt = ViltModel.from_pretrained(model_name, config=config)
        
        # Fix the buffer size issue by updating the position_ids buffer
        max_seq_len = 40  # Use the actual buffer size from the error
        self.vilt.embeddings.text_embeddings.register_buffer(
            "position_ids", 
            torch.arange(max_seq_len).expand((1, -1)),
            persistent=False
        )
        
        # Also update token_type_ids buffer if it exists
        if hasattr(self.vilt.embeddings.text_embeddings, 'token_type_ids'):
            self.vilt.embeddings.text_embeddings.register_buffer(
                "token_type_ids",
                torch.zeros(1, max_seq_len, dtype=torch.long),
                persistent=False
            )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.vilt.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_labels)
        )
    
    def forward(self, input_ids, attention_mask, pixel_values, pixel_mask=None):
        # Ensure inputs don't exceed the expected sequence length
        max_len = 40
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


# Fix 3: Alternative approach - Use a different base model
class ViltWithGroundingVQA(nn.Module):
    """
    Use VQA model which might have different buffer configurations
    """
    def __init__(self, num_labels=2, dropout_rate=0.1):
        super().__init__()
        
        self.num_labels = num_labels
        
        # Try VQA model which might have different sequence length expectations
        self.vilt = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        
        # Replace classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.vilt.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_labels)
        )
    
    def forward(self, input_ids, attention_mask, pixel_values, pixel_mask=None):
        # Get features from base model
        outputs = self.vilt.vilt(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            return_dict=True
        )
        
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        
        return {'logits': logits}


# Usage instructions:
"""
To fix the issue, update your training script as follows:

1. Use max_length=40 when creating the dataset:
   processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
   train_dataset = ViLTDataset(train_df, image_dir, processor, max_length=40)

2. Use the fixed model:
   model = ViltWithGroundingFixed()
   
   OR try the VQA-based model:
   model = ViltWithGroundingVQA()

3. The key changes:
   - Dataset now uses max_length=40 (matching the model's internal buffer)
   - Model fixes the buffer size issue by updating position_ids
   - Alternative VQA model might have different buffer configurations
"""