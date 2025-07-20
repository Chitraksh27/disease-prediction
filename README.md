# 🩺 Disease Classification using XGBoost + SMOTE

This project tackles multi-class disease classification based on symptom data. Given the highly imbalanced dataset with 38 disease classes and only 2000 records, standard models underperformed. This repo explores how to effectively use **XGBoost** combined with **SMOTE oversampling** to improve classification performance on rare diseases.

## 🔍 Problem Statement
- **Goal**: Predict the correct disease label (out of 38) based on symptoms.
- **Input**: Tabular symptom data (e.g., fever, nausea, rash)
- **Output**: Disease class label

## 📊 Dataset
- 2000 patient records
- 10 binary symptoms features
- 1 target label (disease class)
- Severe class imbalance: some diseases have <10 samples

## 🧪 Baseline Model
| Model         | Accuracy      | Macro F1     | Zero-F1 Classes   |
|---------------|-------------|-----------|------------|
| Decision Tree | 31%       | 0.29        | 9+    |
| **XGBoost + SMOTE** | **37%** | **0.35** | **1** | 

## ⚙️ Techniques Used
- **SMOTE** for minority class oversampling
- **XGBoost** (multi: softmax, 38 classes)
- Focused on **Macro F1-Score** due to class imbalance
- Preprocessing: `StandardScaler`, `LabelEncoder`

## 📈 Evaluation
```plaintext
Accuracy:        0.37
Macro F1-score:  0.35
Weighted F1:     0.37
```
Per-class F1 improved substantially. Only 1 class had F1 = 0 (vs 9 in baseline)

## 📦 Libraries
`pandas`  
`scikit-learn`  
`imbalanced-learn`  
`xgboost`  
`matplotlib`  

--- 

*Confusion Matrix*: ![Confusion Matrix](DiseasePred.png)