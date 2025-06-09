import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

def vilt_collate_fn(batch):
    """
    Custom collate function for ViLT dataset that handles variable-sized tensors
    """
    # Separate different types of data
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
    
    # Stack text-related tensors (these should already be the same size)
    try:
        input_ids = torch.stack(input_ids)
        attention_masks = torch.stack(attention_masks)
        token_type_ids = torch.stack(token_type_ids)
        labels = torch.stack(labels)
    except RuntimeError as e:
        print(f"Error stacking text tensors: {e}")
        # Fallback: pad sequences if they're different lengths
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
        token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=0)
        labels = torch.stack(labels)
    
    # Handle pixel values - ensure they're all the same size
    target_size = (3, 384, 384)  # Standard ViLT image size
    processed_pixel_values = []
    processed_pixel_masks = []
    
    for pv, pm in zip(pixel_values, pixel_masks):
        # Ensure pixel values have the right shape
        if pv.shape != target_size:
            if len(pv.shape) == 3:  # [C, H, W]
                pv = F.interpolate(pv.unsqueeze(0), size=(384, 384), mode='bilinear', align_corners=False).squeeze(0)
            else:
                print(f"Unexpected pixel_values shape: {pv.shape}, creating new tensor")
                pv = torch.zeros(target_size)
        
        # Ensure pixel masks have the right shape
        target_mask_size = (384, 384)
        if pm.shape != target_mask_size:
            if len(pm.shape) == 2:  # [H, W]
                pm = F.interpolate(pm.unsqueeze(0).unsqueeze(0).float(), 
                                 size=(384, 384), mode='nearest').squeeze().long()
            else:
                print(f"Unexpected pixel_mask shape: {pm.shape}, creating new tensor")
                pm = torch.ones(target_mask_size, dtype=torch.long)
        
        processed_pixel_values.append(pv)
        processed_pixel_masks.append(pm)
    
    # Stack the processed tensors
    try:
        pixel_values = torch.stack(processed_pixel_values)
        pixel_masks = torch.stack(processed_pixel_masks)
    except RuntimeError as e:
        print(f"Error stacking image tensors: {e}")
        # Create dummy tensors if stacking fails
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

# Updated dataset creation function with custom collate
def create_dataloaders_with_collate(train_dataset, val_dataset, test_dataset, batch_size=8):
    """
    Create DataLoaders with custom collate function
    """
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=vilt_collate_fn,
        num_workers=0  # Set to 0 to avoid multiprocessing issues on Windows
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
    
    return train_loader, val_loader, test_loader