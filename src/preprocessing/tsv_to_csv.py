import pandas as pd

# Load TSV
df = pd.read_csv(r'C:\Users\OMEN\Desktop\Fake_News_Detection\data\raw\multimodal\multimodal_train.tsv', sep='\t')

# Select relevant columns and rename
df_formatted = df[['id', 'clean_title', 'image_url', '2_way_label']].copy()
df_formatted.columns = ['id', 'text', 'image_url', 'label']

# Add extra fields
df_formatted['source'] = 'Fakeddit'
df_formatted['modality'] = df_formatted['image_url'].apply(lambda x: 'multimodal' if pd.notna(x) else 'text-only')
df_formatted['date'] = ''  # Optional if you have date column

# Save as CSV or Parquet
df_formatted.to_csv('formatted_fakeddit.csv', index=False)
