#HAMMERPP.py
"""
HAMMER++ (Hierarchical Attentive Multi-Modal Evidence Reasoning++) Implementation
For Multimodal Fake News Detection with Grounding Capabilities
Compatible with the existing ViLT training pipeline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from transformers import (
    AutoTokenizer, AutoModel, 
    ViTModel, ViTConfig,
    BertModel, BertConfig
)
import numpy as np
from PIL import Image
import math


# ==================== HAMMER++ CORE COMPONENTS ====================

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer layers"""
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
         # Fix: Handle variable sequence lengths
         seq_len = x.size(1)
         if seq_len > self.pe.size(1):
                # Extend positional encodings if needed
                self._extend_pe(seq_len, x.device)
         return x + self.pe[:, :seq_len]
    
    def _extend_pe(self, seq_len, device):
        """Extend positional encodings to a new length"""
        if seq_len > self.pe.size(1):
            new_pe = torch.zeros(1, seq_len, self.pe.size(2), device=device)
            new_pe[:, :self.pe.size(1), :] = self.pe
            position = torch.arange(self.pe.size(1), seq_len).unsqueeze(1).float()
            div_term = torch.exp(torch.arange(0, self.pe.size(2), 2).float() * 
                               -(math.log(10000.0) / self.pe.size(2)))
            new_pe[:, self.pe.size(1):, 0::2] = torch.sin(position * div_term)
            new_pe[:, self.pe.size(1):, 1::2] = torch.cos(position * div_term)
            self.pe = new_pe


class MultiHeadCrossAttention(nn.Module):
    """Cross-modal attention mechanism for HAMMER++"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, query, key, value, mask=None, return_attention=False):
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_k = key.size(1)
        
        # Linear transformations
        Q = self.w_q(query).view(batch_size, seq_len_q, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len_k, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len_k, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.d_model
        )
        
        output = self.w_o(context)
        output = self.layer_norm(output + query)  # Residual connection
        
        if return_attention:
            return output, attention_weights.mean(dim=1)  # Average over heads
        return output


class EvidenceReasoningModule(nn.Module):
    """Evidence reasoning module for HAMMER++"""
    def __init__(self, d_model, n_evidence_types=4, dropout=0.2):
        super().__init__()
        self.d_model = d_model
        self.n_evidence_types = n_evidence_types
        
        # Evidence type embeddings
        self.evidence_embeddings = nn.Embedding(n_evidence_types, d_model)
        
        # Evidence reasoning layers
        self.evidence_attention = nn.MultiheadAttention(
            d_model, num_heads=8, dropout=dropout, batch_first=True
        )
        
        self.evidence_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )
        
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        
    def forward(self, multimodal_features):
        """
        Args:
            multimodal_features: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = multimodal_features.shape
        
        # Create evidence type tokens
        evidence_types = torch.arange(self.n_evidence_types).unsqueeze(0).repeat(batch_size, 1)
        evidence_types = evidence_types.to(multimodal_features.device)
        evidence_embeds = self.evidence_embeddings(evidence_types)  # [batch, n_types, d_model]
        
        # Combine with multimodal features
        combined_features = torch.cat([evidence_embeds, multimodal_features], dim=1)
        
        # Self-attention for evidence reasoning
        attended_features, attention_weights = self.evidence_attention(
            combined_features, combined_features, combined_features
        )
        attended_features = self.layer_norm1(attended_features + combined_features)
        
        # MLP processing
        evidence_output = self.evidence_mlp(attended_features)
        evidence_output = self.layer_norm2(evidence_output + attended_features)
        
        # Split back to evidence and multimodal parts
        evidence_reasoning = evidence_output[:, :self.n_evidence_types]  # Evidence tokens
        enhanced_features = evidence_output[:, self.n_evidence_types:]   # Enhanced multimodal
        
        return enhanced_features, evidence_reasoning, attention_weights


class GroundingHead(nn.Module):
    """Grounding head for localizing manipulations"""
    def __init__(self, d_model, image_patches=196, text_tokens=80, dropout=0.2):
        super().__init__()
        self.d_model = d_model
        self.image_patches = image_patches
        self.text_tokens = text_tokens
        
        # Image grounding (patch-level)
        self.image_grounding = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # Text grounding (token-level)
        self.text_grounding = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # Cross-modal grounding
        self.cross_modal_grounding = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
    def forward(self, image_features, text_features, cross_modal_features=None):
        """
        Args:
            image_features: [batch_size, num_patches, d_model]
            text_features: [batch_size, seq_len, d_model]
            cross_modal_features: [batch_size, total_len, d_model]
        """
        # Image patch grounding
        image_grounding_scores = self.image_grounding(image_features)  # [batch, patches, 1]
        image_grounding_scores = image_grounding_scores.squeeze(-1)
        
        # Text token grounding
        text_grounding_scores = self.text_grounding(text_features)  # [batch, tokens, 1]
        text_grounding_scores = text_grounding_scores.squeeze(-1)
        
        grounding_outputs = {
            'image_grounding': image_grounding_scores,
            'text_grounding': text_grounding_scores
        }
        
        # Cross-modal grounding if features provided
        if cross_modal_features is not None:
            # Global cross-modal grounding
            global_image = image_features.mean(dim=1)  # [batch, d_model]
            global_text = text_features.mean(dim=1)    # [batch, d_model]
            global_cross = torch.cat([global_image, global_text], dim=-1)
            
            cross_modal_score = self.cross_modal_grounding(global_cross)
            grounding_outputs['cross_modal_grounding'] = cross_modal_score.squeeze(-1)
        
        return grounding_outputs


# ==================== MAIN HAMMER++ MODEL ====================

class HammerPlusPlusModel(nn.Module):
    """
    HAMMER++ model for multimodal fake news detection with grounding
    Compatible with the existing training pipeline
    """
    def __init__(
        self,
        vision_model_name="google/vit-base-patch16-224",
        text_model_name="bert-base-uncased",
        d_model=768,
        n_heads=12,
        n_layers=6,
        num_labels=2,
        dropout_rate=0.3,
        max_text_length=80,
        image_size=224,
        enable_grounding=True
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_labels = num_labels
        self.max_text_length = max_text_length
        self.enable_grounding = enable_grounding
        
        # Vision encoder
        vit_config = ViTConfig.from_pretrained(vision_model_name)
        vit_config.hidden_dropout_prob = dropout_rate
        vit_config.attention_probs_dropout_prob = dropout_rate
        self.vision_encoder = ViTModel.from_pretrained(vision_model_name, config=vit_config)
        
        # Text encoder  
        bert_config = BertConfig.from_pretrained(text_model_name)
        bert_config.hidden_dropout_prob = dropout_rate
        bert_config.attention_probs_dropout_prob = dropout_rate
        self.text_encoder = BertModel.from_pretrained(text_model_name, config=bert_config)
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        
        # Dimension alignment
        vision_dim = self.vision_encoder.config.hidden_size
        text_dim = self.text_encoder.config.hidden_size
        
        self.vision_projection = nn.Linear(vision_dim, d_model) if vision_dim != d_model else nn.Identity()
        self.text_projection = nn.Linear(text_dim, d_model) if text_dim != d_model else nn.Identity()
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Cross-modal fusion layers
        self.cross_modal_layers = nn.ModuleList([
            MultiHeadCrossAttention(d_model, n_heads, dropout_rate)
            for _ in range(n_layers)
        ])
        
        # Evidence reasoning module
        self.evidence_reasoning = EvidenceReasoningModule(d_model, dropout=dropout_rate)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.BatchNorm1d(d_model // 2),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model // 2, num_labels)
        )
        
        # Grounding head (if enabled)
        if self.enable_grounding:
            num_patches = (image_size // 16) ** 2  # For ViT-base-patch16
            self.grounding_head = GroundingHead(
                d_model, 
                image_patches=num_patches,
                text_tokens=max_text_length,
                dropout=dropout_rate
            )
        
        # Modal type embeddings
        self.modal_embeddings = nn.Embedding(2, d_model)  # 0: vision, 1: text
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights"""
        for module in [self.classifier, self.vision_projection, self.text_projection]:
            if isinstance(module, nn.Sequential):
                for submodule in module:
                    if isinstance(submodule, nn.Linear):
                        nn.init.xavier_uniform_(submodule.weight)
                        if submodule.bias is not None:
                            nn.init.constant_(submodule.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def encode_image(self, pixel_values):
        """Encode image using ViT"""
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        image_features = vision_outputs.last_hidden_state  # [batch, patches+1, dim]
        
        # Remove CLS token, keep only patch features
        image_patches = image_features[:, 1:, :]  # [batch, patches, dim]
        image_cls = image_features[:, 0, :]       # [batch, dim] - CLS token
        
        # Project to common dimension
        image_patches = self.vision_projection(image_patches)
        image_cls = self.vision_projection(image_cls)
        
        return image_patches, image_cls
    
    def encode_text(self, input_ids, attention_mask):
        """Encode text using BERT"""
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        text_features = text_outputs.last_hidden_state  # [batch, seq_len, dim]
        text_cls = text_outputs.pooler_output           # [batch, dim]
        
        # Project to common dimension
        text_features = self.text_projection(text_features)
        text_cls = self.text_projection(text_cls)
        
        return text_features, text_cls
    
    def cross_modal_fusion(self, image_features, text_features, attention_mask=None):
        """Perform hierarchical cross-modal fusion"""
        batch_size = image_features.size(0)
        
        # Add modal type embeddings
        vision_modal_emb = self.modal_embeddings(torch.zeros(1, dtype=torch.long, device=image_features.device))
        text_modal_emb = self.modal_embeddings(torch.ones(1, dtype=torch.long, device=text_features.device))
        
        image_features = image_features + vision_modal_emb.unsqueeze(0)
        text_features = text_features + text_modal_emb.unsqueeze(0)
        
        # Add positional encoding
        image_features = self.pos_encoding(image_features)
        text_features = self.pos_encoding(text_features)
        
        # Combine features
        combined_features = torch.cat([image_features, text_features], dim=1)

        # Create attention mask for cross-modal attention
        if attention_mask is not None:
            # Create mask for image patches (all valid)
            image_mask = torch.ones(batch_size, image_features.size(1), 
                               dtype=attention_mask.dtype, device=attention_mask.device)
            combined_mask = torch.cat([image_mask, attention_mask], dim=1)
        else:
            combined_mask = None
    

        
        # Cross-modal attention layers
        cross_attention_weights = []
        for layer in self.cross_modal_layers:
            combined_features, attention_weights = layer(
                combined_features, combined_features, combined_features, 
                mask=attention_mask, return_attention=True
            )
            cross_attention_weights.append(attention_weights)
        
        # Split back to image and text components
        num_image_patches = image_features.size(1)
        fused_image_features = combined_features[:, :num_image_patches]
        fused_text_features = combined_features[:, num_image_patches:]
        
        return fused_image_features, fused_text_features, combined_features, cross_attention_weights
    
    def forward(self, input_ids, attention_mask, pixel_values, pixel_mask=None, return_grounding=False):
        """
        Forward pass of HAMMER++
        
        Args:
            input_ids: Text token IDs [batch_size, seq_len]
            attention_mask: Text attention mask [batch_size, seq_len]
            pixel_values: Image tensor [batch_size, 3, H, W]
            pixel_mask: Image mask (unused for ViT)
            return_grounding: Whether to return grounding outputs
        """
        # Encode modalities
        image_patches, image_cls = self.encode_image(pixel_values)
        text_features, text_cls = self.encode_text(input_ids, attention_mask)
        
        # Cross-modal fusion
        fused_image, fused_text, combined_features, cross_attention = self.cross_modal_fusion(
            image_patches, text_features, attention_mask
        )
        
        # Evidence reasoning
        enhanced_features, evidence_reasoning, evidence_attention = self.evidence_reasoning(combined_features)
        
        # Global representation for classification
        global_representation = enhanced_features.mean(dim=1)  # [batch_size, d_model]
        
        # Classification
        logits = self.classifier(global_representation)
        
        outputs = {'logits': logits}
        
        # Grounding outputs (if enabled and requested)
        if self.enable_grounding and return_grounding:
            grounding_outputs = self.grounding_head(
                fused_image, fused_text, enhanced_features
            )
            outputs.update(grounding_outputs)
            
            # Add attention weights for visualization
            outputs['cross_attention_weights'] = cross_attention
            outputs['evidence_attention_weights'] = evidence_attention
        
        return outputs


# ==================== HAMMER++ DATASET WRAPPER ====================

class HammerPlusPlusDataset:
    """
    Dataset wrapper for HAMMER++ that's compatible with the existing ViLTDataset
    """
    def __init__(self, df, image_dir, tokenizer, max_length=80, is_training=False, 
                 use_augmentation=True, augmentation_prob=0.5, image_size=224):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_training = is_training
        self.image_size = image_size
        
        # Import augmentation from original code
        if use_augmentation and is_training:
            from PIL import Image, ImageEnhance, ImageFilter
            import random
            
            class ImageAugmentation:
                def __init__(self, p=0.5):
                    self.p = p
                    
                def __call__(self, image):
                    if random.random() > self.p:
                        return image
                        
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
            
            self.augmentation = ImageAugmentation(p=augmentation_prob)
        else:
            self.augmentation = None
        
        # Auto-detect columns (same as ViLTDataset)
        self.image_path_col = self._detect_image_path_column()
        self.text_cols = self._detect_text_columns()
        self.label_col = self._detect_label_column()
        
        # Image transforms
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"HAMMER++ Dataset initialized with {len(self.df)} samples")
        print(f"Using '{self.image_path_col}' as image path column")
        print(f"Using text columns: {self.text_cols}")
        print(f"Using '{self.label_col}' as label column")
    
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
        import os
        import pandas as pd
        from PIL import Image
        
        row = self.df.iloc[idx]
        
        # Load and process image
        try:
            image_path = os.path.join(self.image_dir, row[self.image_path_col])
            if not os.path.exists(image_path):
                image_path = row[self.image_path_col]
                
            if not os.path.exists(image_path):
                image = Image.new('RGB', (self.image_size, self.image_size), color='white')
            else:
                image = Image.open(image_path).convert('RGB')
                
            # Apply augmentation if training
            if self.augmentation:
                image = self.augmentation(image)
                
            pixel_values = self.image_transform(image)
            
        except Exception as e:
            print(f"Error loading image at index {idx}: {e}")
            pixel_values = torch.zeros(3, self.image_size, self.image_size)
        
        # Process text
        texts = []
        for col in self.text_cols:
            text = str(row[col]) if pd.notna(row[col]) else ""
            texts.append(text)
        combined_text = " ".join(texts).strip()
        if not combined_text:
            combined_text = "No text available"
        
        # Tokenize text
        encoding = self.tokenizer(
            combined_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
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
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'pixel_values': pixel_values,
            'pixel_mask': torch.ones(self.image_size, self.image_size, dtype=torch.long),  # Dummy mask
            'labels': torch.tensor(label, dtype=torch.long)
        }


# ==================== HAMMER++ COLLATE FUNCTION ====================

def hammer_collate_fn(batch):
    """Collate function for HAMMER++ dataset"""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    pixel_mask = torch.stack([item['pixel_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'pixel_values': pixel_values,
        'pixel_mask': pixel_mask,
        'labels': labels
    }


# ==================== GROUNDING LOSS FUNCTIONS ====================

class GroundingLoss(nn.Module):
    """Loss function for grounding supervision"""
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0):
        super().__init__()
        self.alpha = alpha  # Image grounding weight
        self.beta = beta    # Text grounding weight  
        self.gamma = gamma  # Cross-modal grounding weight
        
    def forward(self, grounding_outputs, grounding_targets=None):
        """
        Compute grounding loss
        
        Args:
            grounding_outputs: Dict with grounding predictions
            grounding_targets: Dict with ground truth grounding maps (if available)
        """
        total_loss = 0
        loss_components = {}
        
        if grounding_targets is not None:
            # Supervised grounding loss
            if 'image_grounding' in grounding_outputs and 'image_grounding' in grounding_targets:
                image_loss = F.binary_cross_entropy(
                    grounding_outputs['image_grounding'], 
                    grounding_targets['image_grounding']
                )
                total_loss += self.alpha * image_loss
                loss_components['image_grounding_loss'] = image_loss
                
            if 'text_grounding' in grounding_outputs and 'text_grounding' in grounding_targets:
                text_loss = F.binary_cross_entropy(
                    grounding_outputs['text_grounding'],
                    grounding_targets['text_grounding']
                )
                total_loss += self.beta * text_loss
                loss_components['text_grounding_loss'] = text_loss
        else:
            # Unsupervised grounding regularization
            # Encourage sparse but confident grounding
            if 'image_grounding' in grounding_outputs:
                image_grounding = grounding_outputs['image_grounding']
                # Sparsity regularization
                sparsity_penalty = torch.mean(image_grounding)
                # Confidence regularization  
                confidence_penalty = -torch.mean(
                    image_grounding * torch.log(image_grounding + 1e-8) + 
                    (1 - image_grounding) * torch.log(1 - image_grounding + 1e-8)
                )
                image_loss = sparsity_penalty + 0.1 * confidence_penalty
                total_loss += self.alpha * image_loss
                loss_components['image_regularization'] = image_loss
                
            if 'text_grounding' in grounding_outputs:
                text_grounding = grounding_outputs['text_grounding']
                sparsity_penalty = torch.mean(text_grounding)
                confidence_penalty = -torch.mean(
                    text_grounding * torch.log(text_grounding + 1e-8) + 
                    (1 - text_grounding) * torch.log(1 - text_grounding + 1e-8)
                )
                text_loss = sparsity_penalty + 0.1 * confidence_penalty
                total_loss += self.beta * text_loss
                loss_components['text_regularization'] = text_loss
        
        return total_loss, loss_components


# ==================== COMPATIBILITY LAYER ====================

def create_hammer_plus_plus_model(model_name="hammer++", num_labels=2, dropout_rate=0.3, **kwargs):
    """
    Factory function to create HAMMER++ model with same interface as ViLT model
    """
    return HammerPlusPlusModel(
        num_labels=num_labels,
        dropout_rate=dropout_rate,
        **kwargs
    )


def create_hammer_plus_plus_dataset(df, image_dir, processor_or_tokenizer, max_length=80, 
                                   is_training=False, use_augmentation=True, augmentation_prob=0.5):
    """
    Factory function to create HAMMER++ dataset compatible with existing pipeline
    """
    
    # Extract tokenizer from processor if needed
    if hasattr(processor_or_tokenizer, 'tokenizer'):
        tokenizer = processor_or_tokenizer.tokenizer
    else:
        tokenizer = processor_or_tokenizer
    
    return HammerPlusPlusDataset(
        df=df,
        image_dir=image_dir,
        tokenizer=tokenizer,
        max_length=max_length,
        is_training=is_training,
        use_augmentation=use_augmentation,
        augmentation_prob=augmentation_prob
    )


# ==================== TRAINING UTILITIES ====================

class HammerPlusPlusTrainer:
    """
    Training utilities for HAMMER++ model
    """
    def __init__(self, model, optimizer, device, enable_grounding=True, grounding_loss_weight=0.1):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.enable_grounding = enable_grounding
        self.grounding_loss_weight = grounding_loss_weight
        
        # Loss functions
        self.classification_loss_fn = nn.CrossEntropyLoss()
        if self.enable_grounding:
            self.grounding_loss_fn = GroundingLoss()
    
    def train_step(self, batch, grounding_targets=None):
        """Single training step"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move batch to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device)
        
        # Forward pass
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            pixel_values=batch['pixel_values'],
            pixel_mask=batch.get('pixel_mask'),
            return_grounding=self.enable_grounding
        )
        
        # Classification loss
        classification_loss = self.classification_loss_fn(outputs['logits'], batch['labels'])
        total_loss = classification_loss
        
        loss_components = {'classification_loss': classification_loss.item()}
        
        # Grounding loss (if enabled)
        if self.enable_grounding:
            grounding_outputs = {k: v for k, v in outputs.items() 
                               if k.endswith('_grounding')}
            if grounding_outputs:
                grounding_loss, grounding_components = self.grounding_loss_fn(
                    grounding_outputs, grounding_targets
                )
                total_loss += self.grounding_loss_weight * grounding_loss
                loss_components.update({f"grounding_{k}": v for k, v in grounding_components.items()})
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        # Get predictions
        with torch.no_grad():
            predictions = torch.argmax(outputs['logits'], dim=1)
            accuracy = (predictions == batch['labels']).float().mean().item()
        
        return {
            'total_loss': total_loss.item(),
            'accuracy': accuracy,
            **loss_components
        }
    
    def evaluate_step(self, batch, grounding_targets=None):
        """Single evaluation step"""
        self.model.eval()
        
        # Move batch to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device)
        
        with torch.no_grad():
            # Forward pass
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                pixel_values=batch['pixel_values'],
                pixel_mask=batch.get('pixel_mask'),
                return_grounding=self.enable_grounding
            )
            
            # Classification loss
            classification_loss = self.classification_loss_fn(outputs['logits'], batch['labels'])
            total_loss = classification_loss
            
            loss_components = {'classification_loss': classification_loss.item()}
            
            # Grounding loss (if enabled)
            if self.enable_grounding:
                grounding_outputs = {k: v for k, v in outputs.items() 
                                   if k.endswith('_grounding')}
                if grounding_outputs:
                    grounding_loss, grounding_components = self.grounding_loss_fn(
                        grounding_outputs, grounding_targets
                    )
                    total_loss += self.grounding_loss_weight * grounding_loss
                    loss_components.update({f"grounding_{k}": v for k, v in grounding_components.items()})
            
            # Get predictions
            predictions = torch.argmax(outputs['logits'], dim=1)
            accuracy = (predictions == batch['labels']).float().mean().item()
            
            # Collect predictions and probabilities
            probabilities = F.softmax(outputs['logits'], dim=1)
        
        return {
            'total_loss': total_loss.item(),
            'accuracy': accuracy,
            'predictions': predictions.cpu().numpy(),
            'probabilities': probabilities.cpu().numpy(),
            'labels': batch['labels'].cpu().numpy(),
            **loss_components
        }


# ==================== VISUALIZATION UTILITIES ====================

class HammerPlusPlusVisualizer:
    """
    Visualization utilities for HAMMER++ grounding results
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def visualize_grounding(self, image, text, save_path=None, figsize=(15, 5)):
        """
        Visualize grounding results for a single image-text pair
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from PIL import Image as PILImage
        import numpy as np
        
        # Prepare inputs
        if isinstance(image, str):
            image = PILImage.open(image).convert('RGB')
        
        # Transform image
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        pixel_values = transform(image).unsqueeze(0)
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=80,
            return_tensors='pt'
        )
        
        # Get model outputs
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(
                input_ids=encoding['input_ids'],
                attention_mask=encoding['attention_mask'],
                pixel_values=pixel_values,
                return_grounding=True
            )
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Image grounding
        if 'image_grounding' in outputs:
            image_grounding = outputs['image_grounding'][0].cpu().numpy()  # [num_patches]
            
            # Reshape to 2D grid (assuming 14x14 patches for 224x224 image)
            patch_size = int(np.sqrt(len(image_grounding)))
            grounding_map = image_grounding.reshape(patch_size, patch_size)
            
            # Overlay grounding on image
            axes[1].imshow(image)
            im = axes[1].imshow(grounding_map, alpha=0.6, cmap='Reds', 
                               extent=[0, image.width, image.height, 0])
            axes[1].set_title('Image Grounding Heatmap')
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Text grounding
        if 'text_grounding' in outputs:
            text_grounding = outputs['text_grounding'][0].cpu().numpy()  # [seq_len]
            tokens = self.tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
            
            # Filter out special tokens and padding
            attention_mask = encoding['attention_mask'][0].cpu().numpy()
            valid_indices = np.where(attention_mask == 1)[0]
            
            valid_tokens = [tokens[i] for i in valid_indices if tokens[i] not in ['[CLS]', '[SEP]', '[PAD]']]
            valid_scores = text_grounding[valid_indices][:len(valid_tokens)]
            
            # Create text visualization
            axes[2].barh(range(len(valid_tokens)), valid_scores)
            axes[2].set_yticks(range(len(valid_tokens)))
            axes[2].set_yticklabels(valid_tokens, fontsize=8)
            axes[2].set_xlabel('Grounding Score')
            axes[2].set_title('Text Token Grounding')
            axes[2].invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return outputs
    
    def visualize_attention(self, image, text, layer_idx=-1, head_idx=0, save_path=None):
        """
        Visualize cross-modal attention weights
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        from PIL import Image as PILImage
        
        # Prepare inputs (same as above)
        if isinstance(image, str):
            image = PILImage.open(image).convert('RGB')
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        pixel_values = transform(image).unsqueeze(0)
        
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=80,
            return_tensors='pt'
        )
        
        # Get attention weights
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(
                input_ids=encoding['input_ids'],
                attention_mask=encoding['attention_mask'],
                pixel_values=pixel_values,
                return_grounding=True
            )
        
        if 'cross_attention_weights' in outputs:
            attention_weights = outputs['cross_attention_weights'][layer_idx][0]  # [seq_len, seq_len]
            attention_weights = attention_weights.cpu().numpy()
            
            # Create attention heatmap
            plt.figure(figsize=(12, 10))
            sns.heatmap(attention_weights, cmap='Blues', cbar=True)
            plt.title(f'Cross-Modal Attention (Layer {layer_idx}, Head {head_idx})')
            plt.xlabel('Key Position')
            plt.ylabel('Query Position')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
        
        return outputs


# ==================== EVALUATION METRICS ====================

def compute_metrics(predictions, labels, probabilities=None):
    """
    Compute comprehensive evaluation metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, classification_report, confusion_matrix
    )
    
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(labels, predictions)
    metrics['precision'] = precision_score(labels, predictions, average='weighted', zero_division=0)
    metrics['recall'] = recall_score(labels, predictions, average='weighted', zero_division=0)
    metrics['f1'] = f1_score(labels, predictions, average='weighted', zero_division=0)
    
    # Per-class metrics
    metrics['precision_per_class'] = precision_score(labels, predictions, average=None, zero_division=0)
    metrics['recall_per_class'] = recall_score(labels, predictions, average=None, zero_division=0)
    metrics['f1_per_class'] = f1_score(labels, predictions, average=None, zero_division=0)
    
    # AUC if probabilities provided
    if probabilities is not None:
        if probabilities.shape[1] == 2:  # Binary classification
            metrics['auc'] = roc_auc_score(labels, probabilities[:, 1])
        else:  # Multi-class
            metrics['auc'] = roc_auc_score(labels, probabilities, multi_class='ovr', average='weighted')
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(labels, predictions)
    
    # Classification report
    metrics['classification_report'] = classification_report(labels, predictions, output_dict=True)
    
    return metrics


# ==================== EXAMPLE USAGE ====================

def example_usage():
    """
    Example of how to use HAMMER++ model
    """
    # Create model
    model = create_hammer_plus_plus_model(
        num_labels=2,
        dropout_rate=0.3,
        enable_grounding=True
    )
    
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Example data preparation (you would replace this with your actual data)
    import pandas as pd
    
    sample_df = pd.DataFrame({
        'local_image_path': ['image1.jpg', 'image2.jpg'],
        'title': ['Sample title 1', 'Sample title 2'],
        'text': ['Sample text 1', 'Sample text 2'],
        'label': [0, 1]
    })
    
    # Create dataset
    dataset = create_hammer_plus_plus_dataset(
        df=sample_df,
        image_dir='/path/to/images',
        processor_or_tokenizer=tokenizer,
        is_training=True
    )
    
    # Create data loader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=hammer_collate_fn
    )
    
    # Setup training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    trainer = HammerPlusPlusTrainer(model, optimizer, device)
    
    # Training loop example
    model.train()
    for epoch in range(5):  # Example: 5 epochs
        epoch_loss = 0
        epoch_acc = 0
        num_batches = 0
        
        for batch in dataloader:
            results = trainer.train_step(batch)
            epoch_loss += results['total_loss']
            epoch_acc += results['accuracy']
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        avg_acc = epoch_acc / num_batches
        
        print(f"Epoch {epoch+1}/5 - Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")
    
    # Visualization example
    visualizer = HammerPlusPlusVisualizer(model, tokenizer)
    
    # Note: Replace with actual image path and text
    # visualizer.visualize_grounding(
    #     image='/path/to/image.jpg',
    #     text='Sample news text to analyze',
    #     save_path='grounding_visualization.png'
    # )
    
    print("HAMMER++ model setup and training example completed!")


# ==================== MODEL CONFIGURATION ====================

class HammerPlusPlusConfig:
    """Configuration class for HAMMER++ model"""
    
    def __init__(self, **kwargs):
        # Model architecture
        self.vision_model_name = kwargs.get("vision_model_name", "google/vit-base-patch16-224")
        self.text_model_name = kwargs.get("text_model_name", "bert-base-uncased")
        self.d_model = kwargs.get("d_model", 768)
        self.n_heads = kwargs.get("n_heads", 12)
        self.n_layers = kwargs.get("n_layers", 6)
        self.num_labels = kwargs.get("num_labels", 2)
        self.dropout_rate = kwargs.get("dropout_rate", 0.3)
        
        # Input specifications
        self.max_text_length = kwargs.get("max_text_length", 80)
        self.image_size = kwargs.get("image_size", 224)
        
        # Feature flags
        self.enable_grounding = kwargs.get("enable_grounding", True)
        
        # Training hyperparameters
        self.learning_rate = kwargs.get("learning_rate", 1e-4)
        self.weight_decay = kwargs.get("weight_decay", 0.01)
        self.grounding_loss_weight = kwargs.get("grounding_loss_weight", 0.1)
        
        # Data augmentation
        self.use_augmentation = kwargs.get("use_augmentation", True)
        self.augmentation_prob = kwargs.get("augmentation_prob", 0.5)
    
    def to_dict(self):
        """Convert config to dictionary"""
        return {
            attr: getattr(self, attr) 
            for attr in dir(self) 
            if not attr.startswith('_') and not callable(getattr(self, attr))
        }
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary"""
        return cls(**config_dict)


# ==================== MAIN ENTRY POINT ====================

if __name__ == "__main__":
    print("HAMMER++ (Hierarchical Attentive Multi-Modal Evidence Reasoning++) Model")
    print("=" * 70)
    print("This implementation provides:")
    print("- Hierarchical cross-modal attention")
    print("- Evidence reasoning capabilities") 
    print("- Grounding for manipulation localization")
    print("- Compatible with existing ViLT training pipelines")
    print("=" * 70)
    
    # Run example usage
    try:
        example_usage()
    except Exception as e:
        print(f"Example usage failed (expected without real data): {e}")
        print("To use HAMMER++, create your dataset and follow the example_usage() function.")
    
    print("\nHAMMER++ implementation ready for use!")