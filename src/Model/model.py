"""
ViLT-based models for fake news detection
"""
import torch
import torch.nn as nn
from transformers import ViltModel, ViltConfig, ViltForQuestionAnswering


class ViltWithGrounding(nn.Module):
    """
    ViLT model with custom classification head for fake news detection
    """
    def __init__(self, model_name="dandelin/vilt-b32-mlm", num_labels=2, dropout_rate=0.1):
        super().__init__()
        
        self.num_labels = num_labels
        
        # Load base ViLT model
        self.vilt = ViltModel.from_pretrained(model_name)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.vilt.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_labels)
        )
        
        # Optional: Grounding/attention head for interpretability
        self.grounding_head = nn.Linear(self.vilt.config.hidden_size, 1)
        
    def forward(self, input_ids, attention_mask, pixel_values, pixel_mask=None, return_grounding=False):
        # Get ViLT outputs
        outputs = self.vilt(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            return_dict=True
        )
        
        # Use pooled output for classification
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        
        result = {'logits': logits}
        
        if return_grounding:
            # Grounding scores for interpretability
            grounding_scores = self.grounding_head(outputs.last_hidden_state)
            result['grounding_scores'] = grounding_scores
            result['hidden_states'] = outputs.last_hidden_state
        
        return result


class ViltWithGroundingV2(nn.Module):
    """
    Alternative implementation using pre-trained VQA model as base
    """
    def __init__(self, num_labels=2, dropout_rate=0.1):
        super().__init__()
        
        self.num_labels = num_labels
        
        # Load VQA model and modify for classification
        self.vilt = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        
        # Replace the QA head with binary classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.vilt.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_labels)
        )
        
    def forward(self, input_ids, attention_mask, pixel_values, pixel_mask=None):
        # Get base ViLT outputs
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


class ViltWithGroundingFixed(nn.Module):
    """
    ViLT model with configuration fixes for MLM models
    """
    def __init__(self, model_name="dandelin/vilt-b32-mlm", num_labels=2, dropout_rate=0.1):
        super().__init__()
        
        self.num_labels = num_labels
        
        # Load config and fix num_images if needed
        config = ViltConfig.from_pretrained(model_name)
        
        # Ensure num_images is properly set
        if not hasattr(config, 'num_images') or config.num_images <= 0:
            config.num_images = 1
            
        self.vilt = ViltModel.from_pretrained(model_name, config=config)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.vilt.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_labels)
        )
        
    def forward(self, input_ids, attention_mask, pixel_values, pixel_mask=None):
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


def create_model(model_type="base", model_name="dandelin/vilt-b32-mlm", num_labels=2, dropout_rate=0.1):
    """
    Factory function to create different model variants
    
    Args:
        model_type: "base", "vqa", or "fixed"
        model_name: HuggingFace model name
        num_labels: Number of classification labels
        dropout_rate: Dropout rate for regularization
    
    Returns:
        Initialized model
    """
    if model_type == "base":
        return ViltWithGrounding(model_name, num_labels, dropout_rate)
    elif model_type == "vqa":
        return ViltWithGroundingV2(num_labels, dropout_rate)
    elif model_type == "fixed":
        return ViltWithGroundingFixed(model_name, num_labels, dropout_rate)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# Model configuration constants
MODEL_CONFIGS = {
    "vilt_base": {
        "model_name": "dandelin/vilt-b32-mlm",
        "model_type": "base"
    },
    "vilt_vqa": {
        "model_name": "dandelin/vilt-b32-finetuned-vqa",
        "model_type": "vqa"
    },
    "vilt_fixed": {
        "model_name": "dandelin/vilt-b32-mlm",
        "model_type": "fixed"
    }
}