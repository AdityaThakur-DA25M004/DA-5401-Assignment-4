# DA5401 Assignment 4 — GMM-Based Synthetic Sampling for Fraud Detection

####  Name-Aditya Thakur &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  Roll No. - DA25M004

## 📌 Project Overview
This project investigates **Gaussian Mixture Model (GMM)-based synthetic sampling** to handle **severe class imbalance** in credit card fraud detection.  

We compare three approaches:
1. **Baseline Logistic Regression** (no resampling)
2. **GMM Oversampling** (probabilistic generation of minority samples)
3. **GMM + CBU (Clustering-Based Undersampling)** (hybrid approach)

**Evaluation Metrics:** Precision, Recall, F1-score, ROC AUC, PR AUC  
(*focus on minority/fraud class metrics*)

---

## 📊 Dataset Summary
- **Total samples:** 284,807  
- **Majority class (0 = Non-fraud):** 284,315 (~99.83%)  
- **Minority class (1 = Fraud):** 492 (~0.17%)  
- **Imbalance ratio:** ~578 : 1  
- **Null values:** None  

**Key Points:**
- Dataset is extremely imbalanced.  
- Accuracy is misleading (naive all-zero model → ~99.8%).  
- Use **minority-focused metrics** for evaluation.  
- Fraud detection is **recall-critical**.

---

## 🧑‍🔬 Stepwise Workflow

### 1️⃣ Data Preparation
- Split dataset into **train/test (80/20, stratified)**.  
- Train: 227,451 non-fraud, 394 fraud  
- Test: 56,864 non-fraud, 98 fraud  

### 2️⃣ Baseline Logistic Regression
- Scaled features using `StandardScaler`.  
- Trained `LogisticRegression(max_iter=1000, solver='lbfgs')`.  

**Fraud Class Performance:**
- Precision = 0.827  
- Recall = 0.633  
- F1-score = 0.717  
- ROC AUC = 0.9605  
- PR AUC = 0.7414  

**Observation:** High precision, low recall → misses ~37% of fraud cases.

### 3️⃣ GMM Oversampling
- Theoretical Foundation for SMOTE vs GMM.
- Fitted Gaussian Mixture Model on minority samples.  
- Selected optimal number of components using BIC/AIC.
- Explanation how GMM sampling method works. 
- Generated synthetic minority samples from GMM.  

**Fraud Class Performance:**
- Precision = 0.644  
- Recall = 0.867  
- F1-score = 0.739  

**Observation:** Recall improves significantly, precision drops → more false positives.

### 4️⃣ Clustering-Based Undersampling (CBU)
- Use Different metrics like Elbow method, DBI,CHI to determine optimum number of clusters.
- Applied KMeans to majority class.  
- Sampled representative points from clusters to reduce imbalance.  

**Observation:** Reduces redundancy while preserving diversity of majority samples.

### 5️⃣ GMM + CBU (Hybrid)
- Minority oversampled with GMM, majority undersampled with clusters.  

**Fraud Class Performance:**
- Precision = 0.721  
- Recall = 0.816  
- F1-score = 0.766 (**highest**)  

**Observation:** Best trade-off between precision and recall.

---

## 📊 Results Comparison

| Model                 | Precision | Recall | F1-score |
|------------------------|-----------|--------|----------|
| Baseline (LogReg)      | 0.827     | 0.633  | 0.717    |
| GMM (Oversampling)     | 0.644     | 0.867  | 0.739    |
| GMM + CBU (Over+Under) | 0.721     | 0.816  | 0.766    |

- ROC AUC ≈ 0.97  
- PR AUC ≈ 0.74  

---

## 📌 Key Insights
- **Baseline:** High precision, low recall → misses many frauds.  
- **GMM Oversampling:** High recall, lower precision → more frauds detected but more false alarms.  
- **GMM + CBU:** Best balance, highest F1-score → practical choice for fraud detection.

---

## 🚨 Caveats
- Too many GMM components may overfit minority samples.   
- Threshold tuning is critical depending on cost of FP vs FN using Validation set.
---



## ✅ Conclusion
**Recommendation:** Use **GMM + CBU** for fraud detection.  
- Balances recall and precision  
- Highest F1-score  
- Most effective for extremely imbalanced datasets
