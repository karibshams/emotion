# 🌟 Emotion Recognition — Advanced Transformer Fine-Tuning

> **Award-worthy**, clear, and step-by-step README for your emotion-classification project — enriched with insights from the BanglaMixEmotion IEEE paper.

---

## 📌 Project Title
**Emotion Recognition with Advanced Transformer Fine-Tuning**  
_Modular framework for multilingual / code-mixed emotion classification using Transformer backbones + advanced fine-tuning techniques._

---

## 🚀 Project Summary
This repository implements a robust and research-grade pipeline to fine-tune transformer models for **5-way emotion classification** (Happy, Love, Sadness, Anger, Fear) on product-review data. It focuses on reproducible experiments combining modular techniques such as **Polynomial Attention**, **Mixout**, **LoRA**, **BitFit**, and **Residual Fine-Tuning (ReFT)** to evaluate which combinations are most effective in multilingual and code-mixed scenarios.  

This work builds on the same motivation and dataset design as the *BanglaMixEmotion* study (a large, manually-annotated Bangla–English product-review corpus) and adapts several of its methodological ideas into an easy-to-run, experiment-driven codebase.

---

## 🎯 Why this project matters
- Handles **code-mixed** (Bangla–English) e-commerce reviews and multilingual inputs.
- Designed for **research reproducibility** and practical deployment.
- Modular — plug in different backbones and PEFT (parameter-efficient fine-tuning) strategies.
- Great starting point for emotion-aware recommenders, empathetic chatbots, and NLP analytics.

---

## 🧩 Project Structure
```
.
├── README.md
├── requirements.txt
├── data/
│   └── balanced_dataset.csv
├── src/
│   ├── preprocess.py
│   ├── models.py
│   ├── train.py
│   └── evaluate.py
├── experiments/
│   └── config.json
└── notebooks/
```
---

## 🗂 Dataset: expected format & preprocessing
**CSV Columns:**
`Product Name`, `Rating`, `Review`, `Product Category`, `Data Source`, `Sentiment`, `Emotion`

**Merged text construction:**
```python
df["text"] = (
    df["Product Name"] + " [SEP] " +
    df["Rating"].astype(str) + " [SEP] " +
    df["Review"] + " [SEP] " +
    df["Product Category"] + " [SEP] " +
    df["Data Source"] + " [SEP] " +
    df["Sentiment"]
)
```
**Class mapping:**
```python
{"Happy":0, "Love":1, "Sadness":2, "Anger":3, "Fear":4}
```

---

## 🏗 Model Components
- **Backbones supported:**
  - `bert-base-multilingual-uncased`
  - `roberta-base`
  - `xlm-roberta-base`
  - `answerdotai/ModernBERT-base`
- **Optional Modules:**
  - Polynomial Multi-Head Attention  
  - Mixout Regularization  
  - LoRA (Low-Rank Adaptation)  
  - BitFit (Bias Fine-Tuning)  
  - ReFT (Residual Fine-Tuning)

Each module can be toggled individually via configuration.

---

## 🛠 Installation
```bash
git clone https://github.com/<your-username>/emotion-transformer-finetuning.git
cd emotion-transformer-finetuning
pip install -r requirements.txt
```

**Or manually:**
```bash
pip install torch transformers scikit-learn pandas numpy tqdm
```

---

## ▶️ Quick Start
1. Place your dataset in `data/balanced_dataset.csv`  
2. Edit configuration in `src/train.py` or `experiments/config.json`
3. Run training:
   ```bash
   python src/train.py --config experiments/config.json
   ```
4. The best model checkpoint is saved as `best_model.pt`

---

## 🔬 Experimentation Guidelines
1. Start with a baseline (no PEFT, no PolyAttention)
2. Add techniques one by one (PolyAttention → Mixout → LoRA → BitFit → ReFT)
3. Use 70/10/20 data split (train/val/test)
4. Enable early stopping (patience = 3)
5. Log experiments via TensorBoard or Weights & Biases

---

## ♻ Reproducibility
- Checkpoints auto-saved per best validation loss
- RNG seeded for consistent splits
- Tokenizer–model pairing stored per checkpoint

---

## 📂 How to Access the Code
```bash
git clone https://github.com/<your-username>/emotion-transformer-finetuning.git
```
or download ZIP from the repository page.

Dataset reference available in the original *BanglaMixEmotion* publication.

---

## 🧭 Paper Context
This project aligns with the *BanglaMixEmotion* research, which introduced a large manually-annotated Bangla–English product-review corpus and extensive ablation experiments across transformer backbones with advanced fine-tuning strategies. The methods such as polynomial attention, mixout, and lightweight PEFT adaptations are inspired by that study for efficient, high-performance emotion classification.

---

## 🧪 Notes & Best Practices
- Test each module individually for interpretability.
- Adjust max token length depending on model capacity.
- Use mixed precision (FP16) for faster training on large models.
- Monitor memory usage when stacking multiple fine-tuning methods.

---

## 💬 Contact
**Author:** Karib Shams  
📧 [shams321karib@gmail.com]  
💼 [LinkedIn / Portfolio Link]

---

## 📝 Citation
If you use this repository, please cite:
- *BanglaMixEmotion* (IEEE paper)
- This GitHub repository

---

⭐ **If this work inspires you, please star ⭐ the repo to support open research!**
