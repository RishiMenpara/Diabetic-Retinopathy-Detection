# 👁️ Diabetic Retinopathy Detection System

An AI-powered web application for automated **Diabetic Retinopathy (DR) severity grading** from retinal fundus images using a **ConvNeXt-Tiny** deep learning model trained on the **EyePACS** dataset.

[![Streamlit App](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-FF4B4B)](https://huggingface.co/spaces/RishiMenpara/diabetic-retinopathy)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)



## 📋 DR Severity Classes (ICDR Scale)

| Grade | Class | Clinical Description |
|-------|-------|----------------------|
| 0 | **No DR** | No signs detected |
| 1 | **Mild DR** | Microaneurysms only |
| 2 | **Moderate DR** | More than microaneurysms, less than severe |
| 3 | **Severe DR** | Extensive hemorrhages / IRMA / venous beading |
| 4 | **Proliferative DR** | Neovascularization / vitreous hemorrhage |

---

## 📁 Project Structure

```
Diabetic-Retinopathy-Detection/
│
├── notebooks/
│   └── diabetic-retinopathy.ipynb   # Training notebook (Kaggle / Colab)
│
├── models/
│   └── best_convnext_model.pth      # Trained model weights (Git LFS)
│
├── streamlit/                        # Web application
│   ├── app.py                        # Streamlit app entry point
│   └── .streamlit/
│       └── config.toml               # Light-theme config
│
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

---

## 🧠 Model Details

| Property | Value |
|----------|-------|
| Architecture | ConvNeXt-Tiny |
| Pre-training | ImageNet-1K |
| Input size | 224 × 224 |
| Output classes | 5 (ICDR scale) |
| Parameters | ~28M |

### Training  (`notebooks/DR_ConvNeXt_Training.ipynb`)

- **Dataset:** [EyePACS — Diabetic Retinopathy Detection](https://www.kaggle.com/c/diabetic-retinopathy-detection) (Kaggle)
- **Augmentations:** Random crop, horizontal/vertical flip, colour jitter, rotation
- **Optimizer:** AdamW (lr = 1e-4, weight decay = 1e-4)
- **Scheduler:** Cosine Annealing LR
- **Loss:** Cross-Entropy with label smoothing (0.1)
- **Epochs:** 30

---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/Diabetic-Retinopathy-Detection.git
cd Diabetic-Retinopathy-Detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run streamlit/app.py
```

Open **http://localhost:8501** in your browser.

---

## 🌐 Deploy on Hugging Face Spaces

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space) → choose **Streamlit SDK**
2. Upload all files preserving the folder structure
3. For `models/best_convnext_model.pth` — upload via the HF web UI (handles large files automatically)
4. HF Spaces auto-builds from `requirements.txt` — live in ~3 minutes

---

## ⚠️ Medical Disclaimer

This tool is intended for **research and educational purposes only**.
It is **not** a substitute for professional medical advice, diagnosis, or treatment.
Always consult a qualified ophthalmologist for clinical decisions.

---

## 📄 License

MIT License © 2024 — see [LICENSE](LICENSE)
