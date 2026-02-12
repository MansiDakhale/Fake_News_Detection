#  Multi-Modal Fake News Detection using HAMMER++

> A deep learning pipeline that detects fake news using both textual and visual content, powered by the HAMMER++ multi-modal transformer.

---

##  Overview

This project aims to detect fake news articles by leveraging both text and image modalities. It uses the powerful [HAMMER++](https://arxiv.org/abs/2310.03203) architecture, which performs hierarchical attention across modalities. Weights & Biases is used for experiment tracking, and the model is trained and validated on real-world fake news datasets like BoomLive and AltNews.

---

##  Model Architecture

- **Backbone**: HAMMER++
- **Input**: Text (e.g., headline, content) + Image (e.g., article/post image)
- **Embedding**: ViT & BERT
- **Training losses**:
  - Classification Loss
  - Grounding Loss (optional, currently disabled)
- **Framework**: PyTorch, HuggingFace Transformers

---

##  Key Results

| Metric              | Validation | Test     |
|---------------------|------------|----------|
| Accuracy            | 81.48%     | 80.87%   |
| F1-Score            | 81.46%     | 80.77%   |
| AUC-ROC             | 89.12%     | 91.78%   |
| Precision           | 81.47%     | 81.12%   |
| Recall              | 81.48%     | 80.87%   |
| Best Val F1-Score   | **81.96%** | -        |
| Epochs Trained      | 8          | 8450 steps |

> 🔗 [View Full W&B Logs](https://wandb.ai/mansidakhalee-indian-institute-of-information-technology/hammer-plus-plus-fake-news/runs/6rqu13wo)

---

##  Folder Structure

Fake_News_Detection/
│
├── data/ # Input datasets (BoomLive, AltNews, etc.)
├── models/ # HAMMER++ model code
├── utils/ # logger.py, visualization.py
├── logs/ # Training logs
├── train.py # Main training script
├── eval.py # Evaluation script
├── generate_report.py # Metric reports
├── wandb/ # W&B run metadata
├── requirements.txt
└── README.md


---

##  Setup & Training

### 1. Clone the Repo

```bash
git clone https://github.com/<your-username>/Fake_News_Detection.git
cd Fake_News_Detection

## Create Virtual Environment
venv_py311\Scripts\activate 

🛠 Features
 Dual-modality input (text + image)

 Fine-tuned HAMMER++ model

 W&B integration for tracking and visualization

 Precision-recall balanced output

 Grounding module placeholder (planned for manipulation localization)

 Visualizations
Visit WandB Dashboard to view:

Loss & Accuracy curves

Precision-Recall & ROC curves

Per-class evaluation

 Dataset Sources
BoomLive Fake News Dataset

AltNews Public Dataset

Indian Kanoon JSON

COVID-19 WHO Reports 

 Future Work
 Add grounding-based manipulation localization

 Deploy as a FastAPI/Streamlit tool

 Ingest real-time data from Reddit, BoomLive, Twitter

 Authors
Mansi Dakhale
Deepika Vishwakarma
Soumita Chatterjee
Master's in AI & ML, IIIT Lucknow
GitHub • LinkedIn
