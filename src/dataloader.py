import pandas as pd
from torch.utils.data import Dataset
import os
from src.preprocessing.Vilt import FakeNewsDataset, processor
from torch.utils.data import DataLoader


# Load the splits
train_df = r"C:\Users\OMEN\Desktop\Fake_News_Detection\data\splitted\train.csv"
val_df = r"C:\Users\OMEN\Desktop\Fake_News_Detection\data\splitted\val.csv"
test_df = r"C:\Users\OMEN\Desktop\Fake_News_Detection\data\splitted\test.csv"

image_dir1 = r"C:\Users\OMEN\Desktop\Fake_News_Detection\Images\Train_images"
image_dir2 = r"C:\Users\OMEN\Desktop\Fake_News_Detection\Images\Test_images"
image_dir3 = r"C:\Users\OMEN\Desktop\Fake_News_Detection\Images\Val_images"


# Reuse the same processor and Dataset class
train_dataset = FakeNewsDataset(train_df, image_dir1, processor)
test_dataset = FakeNewsDataset(test_df, image_dir2, processor)
val_dataset = FakeNewsDataset(val_df, image_dir3, processor)


train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)
test_loader = DataLoader(test_dataset, batch_size=8)

