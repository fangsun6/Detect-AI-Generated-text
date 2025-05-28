# 🤖 LLM Detection via Fine-Tuned DeBERTa and Embedding Visualization

This project aims to detect large language model (LLM)-generated text using a fine-tuned DeBERTa-v3 model and visualize text embeddings to compare LLM-generated content and human-written content.

---

## 📁 Project Contents

```
.
├── deberta_train_exp5.py                   # DeBERTa-v3 fine-tuning for binary classification (LLM vs. Human)
├── embed_vis.py                            # Embedding extraction & t-SNE visualization using SentenceTransformer
├── llm-detect-code.ipynb                   # (Optional) Notebook for exploration or inference
├── train1.csv                              # Training dataset with labeled text
├── nonTargetText_llm_slightly_modified_gen.csv  # Validation set of generated text
```

---

## 📌 Objectives

- Train a binary classifier to detect LLM-generated vs. human-written text.
- Fine-tune [DeBERTa-v3-small](https://huggingface.co/microsoft/deberta-v3-small) using a diverse corpus.
- Use SentenceTransformer and t-SNE for visualizing latent embedding space of LLM vs. human samples.

---

## 📦 Data Sources

- **LLM-generated text**: from datasets such as `pile2`, `Ultra`, `lmsys`, and others.
- **Human-written text**: labeled in the Human_LLM dataset.
- **Validation set**: `nonTargetText_llm_slightly_modified_gen.csv`.

All datasets are concatenated and preprocessed for binary classification.

---

## 🧠 Model: DeBERTa-v3 Fine-Tuning

Script: `deberta_train_exp5.py`  
Main features:
- Uses HuggingFace Transformers (`AutoModelForSequenceClassification`)
- Training pipeline built with HuggingFace `Trainer`
- Tokenization with max sequence length of 384
- Custom ROC-AUC metric for model evaluation
- Early stopping callback
- Cosine learning rate scheduler
- Mixed precision training (fp16)

TrainingArgs:
```python
per_device_train_batch_size = 256
num_train_epochs = 60
learning_rate = 1e-4
evaluation_strategy = "epoch"
metric_for_best_model = "roc_auc"
```

---

## 📊 Embedding Visualization

Script: `embed_vis.py`  
Highlights:
- Uses SentenceTransformer (`all-MiniLM-L6-v2`)
- Extracts embeddings for both LLM and human samples
- Applies t-SNE for dimensionality reduction
- Visualizes separation of embeddings with Seaborn scatter plot

Example Output:

![t-SNE Scatter](#) *(replace with actual image if needed)*

---

## 🚀 How to Run

### 1. Environment Setup

Install dependencies:
```bash
pip install transformers datasets scikit-learn pandas matplotlib seaborn sentence-transformers
```

Ensure GPU is available for best performance.

### 2. Run DeBERTa Training

```bash
python deberta_train_exp5.py
```

### 3. Visualize Embeddings

```bash
python embed_vis.py
```

---

## 📈 Outputs

- Fine-tuned DeBERTa-v3 model checkpoint
- ROC-AUC performance metrics
- t-SNE plot showing separation of LLM vs. human text embeddings

---

## 🧪 Evaluation Metric

```python
from sklearn.metrics import roc_auc_score
```
The model uses ROC-AUC score to evaluate binary classification performance.

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for more details.

---

## 🙏 Acknowledgements

- HuggingFace Transformers and Datasets libraries
- Kaggle dataset providers for LLM vs. human corpora
- SentenceTransformers by UKPLab
