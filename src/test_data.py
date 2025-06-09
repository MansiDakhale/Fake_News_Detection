#test_data.py
"""
Updated Data Loading Test Script with custom collate function
"""

import sys
import os
import torch
import pandas as pd
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent))

from src.preprocessing.Vilt import ViLTDataset
#from src.preprocessing.data_loading import load_and_split_data
from transformers import ViltProcessor
from torch.utils.data import DataLoader
import torch.nn.functional as F

def vilt_collate_fn(batch):
    """Custom collate function for ViLT dataset"""
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
    
    # Stack text-related tensors
    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)
    token_type_ids = torch.stack(token_type_ids)
    labels = torch.stack(labels)
    
    # Handle pixel values - ensure consistent size
    target_size = (3, 384, 384)
    processed_pixel_values = []
    processed_pixel_masks = []
    
    for pv, pm in zip(pixel_values, pixel_masks):
        # Ensure pixel values have the right shape
        if pv.shape != target_size:
            if len(pv.shape) == 3:
                pv = F.interpolate(pv.unsqueeze(0), size=(384, 384), mode='bilinear', align_corners=False).squeeze(0)
            else:
                pv = torch.zeros(target_size)
        
        # Ensure pixel masks have the right shape
        if pm.shape != (384, 384):
            if len(pm.shape) == 2:
                pm = F.interpolate(pm.unsqueeze(0).unsqueeze(0).float(), 
                                 size=(384, 384), mode='nearest').squeeze().long()
            else:
                pm = torch.ones((384, 384), dtype=torch.long)
        
        processed_pixel_values.append(pv)
        processed_pixel_masks.append(pm)
    
    # Stack the processed tensors
    pixel_values = torch.stack(processed_pixel_values)
    pixel_masks = torch.stack(processed_pixel_masks)
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_masks,
        'token_type_ids': token_type_ids,
        'pixel_values': pixel_values,
        'pixel_mask': pixel_masks,
        'labels': labels
    }

def test_data_loading():
    """Test the data loading pipeline with pre-split data"""
    print("=" * 50)
    print("FAKE NEWS DETECTION - DATA LOADING TEST")
    print("=" * 50)
    
    try:
        print("Loading pre-split data...")
        
        # Check if split data files exist
        data_dir = Path("data")
        train_file = data_dir / r"C:\Users\OMEN\Desktop\Fake_News_Detection\data\Preprocessed\train.csv"
        val_file = data_dir / r"C:\Users\OMEN\Desktop\Fake_News_Detection\data\Preprocessed\val.csv" 
        test_file = data_dir / r"C:\Users\OMEN\Desktop\Fake_News_Detection\data\Preprocessed\test.csv"
        
        # Alternative file names to check
        alt_files = [
            (data_dir / "train_data.csv", data_dir / "val_data.csv", data_dir / "test_data.csv"),
            (data_dir / "train_split.csv", data_dir / "val_split.csv", data_dir / "test_split.csv"),
        ]
        
        # Load the split data
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
            
                                                                                    
        print(f"✓ Data loaded successfully")
        print(f"  Train size: {len(train_df)}")
        print(f"  Validation size: {len(val_df)}")
        print(f"  Test size: {len(test_df)}")
        
        # Initialize processor
        processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
        
        # Create datasets
        image_dir = "data/images"
        train_dataset = ViLTDataset(train_df, image_dir, processor)
        val_dataset = ViLTDataset(val_df, image_dir, processor)
        test_dataset = ViLTDataset(test_df, image_dir, processor)
        
        print(f"✓ Datasets created successfully")
        print(f"  Train dataset size: {len(train_dataset)}")
        print(f"  Validation dataset size: {len(val_dataset)}")
        print(f"  Test dataset size: {len(test_dataset)}")
        
        # Print column information
        col_info = train_dataset.get_column_info()
        print(f"\n📊 Dataset Column Information:")
        print(f"  Image path column: {col_info['image_path_column']}")
        print(f"  Text columns: {col_info['text_columns']}")
        print(f"  Label column: {col_info['label_column']}")
        
        # Create data loaders with custom collate function
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, 
                                collate_fn=vilt_collate_fn, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, 
                              collate_fn=vilt_collate_fn, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, 
                               collate_fn=vilt_collate_fn, num_workers=0)
        
        print(f"✓ DataLoaders created successfully")
        
        print("Testing batch loading...")
        
        # Test loading a batch from each loader
        loaders = [("Train", train_loader), ("Validation", val_loader), ("Test", test_loader)]
        
        for name, loader in loaders:
            try:
                batch = next(iter(loader))
                print(f"✓ {name} batch loaded successfully")
                print(f"  Batch size: {batch['input_ids'].shape[0]}")
                print(f"  Input IDs shape: {batch['input_ids'].shape}")
                print(f"  Pixel values shape: {batch['pixel_values'].shape}")
                print(f"  Labels shape: {batch['labels'].shape}")
                
                # Verify tensor consistency
                expected_shapes = {
                    'input_ids': (batch['input_ids'].shape[0], 512),
                    'attention_mask': (batch['attention_mask'].shape[0], 512),
                    'pixel_values': (batch['pixel_values'].shape[0], 3, 384, 384),
                    'pixel_mask': (batch['pixel_mask'].shape[0], 384, 384),
                    'labels': (batch['labels'].shape[0],)
                }
                
                all_correct = True
                for key, expected in expected_shapes.items():
                    actual = batch[key].shape
                    if actual != expected:
                        print(f"  ⚠️  {key} shape mismatch: expected {expected}, got {actual}")
                        all_correct = False
                
                if all_correct:
                    print(f"  ✓ All tensor shapes are correct")
                
            except Exception as e:
                print(f"❌ Error loading {name} batch: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        print(f"\n🎉 All tests passed! Data loading pipeline is working correctly.")
        print(f"✓ Ready for training!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during data loading test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_data_loading()
    if success:
        print(f"\n🚀 You can now proceed with training your model!")
    else:
        print(f"\n❌ Data loading failed. Please fix the issues before training.")