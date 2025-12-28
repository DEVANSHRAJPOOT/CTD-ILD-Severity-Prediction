# 🫁 CTD-ILD Severity Prediction Using Clinical Data and MONAI-based CT Imaging

## 📌 Project Overview

This project presents an **end-to-end multimodal machine learning framework** for assessing **disease severity in Connective Tissue Disease–associated Interstitial Lung Disease (CTD-ILD)** by integrating:

- **Clinical data** (demographics, disease characteristics, functional tests)
- **Radiologic data** derived from **HRCT scans**
- **Deep learning–based segmentation** using the **MONAI framework**

Three predictive models are developed and compared:

1. **Model 1 – Clinical-Only Model**
2. **Model 2 – Radiology-Only Model**
3. **Model 3 – Combined Clinical + Radiology Model**

The pipeline is fully automated, modular, and designed to be extensible to real clinical datasets.

---

## 🎯 Research Objectives

- Identify **clinical factors** associated with CTD-ILD severity  
- Extract **quantitative radiologic features** from CT images using AI  
- Compare classical ML algorithms across unimodal and multimodal settings  
- Demonstrate an **end-to-end MONAI segmentation → radiomics → ML pipeline**

---

## 🧠 Disease Severity Definitions

### Clinical Severity Measures

#### Forced Vital Capacity (FVC)
- **Mild**: 61–80%  
- **Moderate**: 31–60%  
- **Severe**: <30%  

#### 6-Minute Walk Test (6-MWT)
- **<250 m**
- **250–350 m**
- **>400 m**

### Radiologic Severity

#### Warrick Score
Based on pathologic type and segmental extension:
- **Mild**: 0–7  
- **Moderate**: 8–15  
- **Severe**: >15  

---

## 🗂️ Project Structure

``` bash
CTD_ILD_Project/
│
├── data/
│ ├── clinical.csv # Synthetic clinical dataset
│ ├── radiology.csv # AI-derived radiologic features
│ ├── images/ # Synthetic CT volumes (.nii.gz)
│ ├── labels/ # Synthetic ground-truth masks
│ └── predicted_masks/ # MONAI-predicted segmentation masks
│
├── notebooks/
│ ├── 01_data_generation.ipynb
│ ├── 02_model1_clinical.ipynb
│ ├── 03_model2_radiology.ipynb
│ └── 04_model3_combined.ipynb
│
├── src/
│ ├── preprocessing.py
│ ├── models.py
│ └── evaluation.py
│
├── monai/
│ ├── simulate_ct_data.py
│ ├── 02_train_segmentation.py
│ ├── 03_inference_features.py
│ └── 04_export_radiology_csv.py
│
├── .venv/ # Main ML environment
├── monai_env/ # MONAI / PyTorch environment
└── README.md
```

---

## 🧪 Model 1 — Clinical-Only Machine Learning

### Input Features
- **Demographics**: Age, Sex, Smoking history  
- **Disease-specific**: CTD type, autoantibodies, histology, disease duration, comorbidities  

### Feature Selection
- **LASSO (L1-regularized Logistic Regression)**  
  Used to identify the most relevant predictors of disease severity.

### Algorithms Evaluated
- Naive Bayes  
- Decision Tree  
- Random Forest  
- Gradient Boosting  
- K-Nearest Neighbors  
- Support Vector Machine  

### Evaluation Metrics
- Accuracy  
- Precision, Recall, F1-Score  
- Confusion Matrix  
- AUROC (binary / multiclass OvR)  

---

## 🧠 Model 2 — Radiology-Only Machine Learning

### Imaging Pipeline
- **3D U-Net** implemented using **MONAI**
- Sliding-window inference for full CT volumes
- Automated segmentation of lung abnormalities

### Extracted Radiologic Features
- Ground-glass opacity (GGO) volume  
- Fibrosis volume  
- Honeycombing volume  
- Number of involved segments  

### Notes on Synthetic Data
This project uses **synthetically generated CT data** to validate the imaging pipeline.  
Due to the sparse and random nature of synthetic lesions, predicted lesion volumes are minimal.  
However, the **end-to-end segmentation and feature extraction pipeline is fully functional** and directly applicable to real datasets.

---

## 🔗 Model 3 — Combined Clinical + Radiologic Model

### Purpose
To evaluate whether **multimodal fusion** improves CTD-ILD severity prediction.

### Inputs
- Selected clinical features (from Model 1)
- AI-derived radiologic features (from MONAI pipeline)

### Outcome
- Clinical features dominate under synthetic conditions
- Radiologic pipeline feasibility is validated
- Multimodal ML framework demonstrated successfully

---

## 🤖 MONAI Framework Integration

This project uses **MONAI** for:
- Medical image preprocessing
- 3D UNet training
- Sliding-window inference
- Automated segmentation mask generation
- Radiomics feature extraction

The MONAI pipeline is modular and reusable for:
- Real HRCT datasets
- Other ILD subtypes
- Multicenter imaging studies

---

## ⚙️ Environments & Dependencies

### Main ML Environment (`.venv`)
- Python 3.10+
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- jupyter

### MONAI Environment (`monai_env`)
- Python 3.10
- torch, torchvision
- monai
- nibabel
- scikit-image

---

## 🚀 How to Run the Project

### 1️⃣ Clinical & ML Models
```bash
activate .venv
jupyter notebook
```

## Run notebooks in order:
- 01_data_generation.ipynb

- 02_model1_clinical.ipynb

- 03_model2_radiology.ipynb

- 04_model3_combined.ipynb

### 2️⃣ MONAI Imaging Pipeline

```bash 

activate monai_env
python monai/simulate_ct_data.py
python monai/02_train_segmentation.py
python monai/03_inference_features.py
python monai/04_export_radiology_csv.py

``` 

#### 📌 Key Contributions

- ✅ End-to-end multimodal ML framework for CTD-ILD

- ✅ Clinical feature selection using LASSO

- ✅ MONAI-based 3D CT segmentation pipeline

- ✅ Automated radiomics extraction

- ✅ Comparative ML model analysis

- ✅ Research-ready, modular architecture


## 📜 Disclaimer

- This project uses synthetic data for methodological demonstration only.
- Results are not intended for clinical use without validation on real patient data.


👨‍💻 Author
Devansh Rajpoot
AI / ML Research | Medical Imaging | Clinical Decision Support


---

