# Global AI Job Market & Salary Trends 2025
## Predicting AI Job Salaries using Machine Learning & PySpark

## 📋 Overview
A machine learning project predicting AI job salaries globally using 
the Global AI Job Market & Salary Trends 2025 dataset (15,000 records, 
19 attributes).

**Research Question:** What factors most significantly influence AI job 
salaries globally in 2025, and can we accurately predict salary levels 
using machine learning and distributed computing techniques?

## 📁 Dataset
- **Name:** Global AI Job Market & Salary Trends 2025
- **Source:** [Kaggle](https://www.kaggle.com/datasets/bismasajjad/global-ai-job-market-and-salary-trends-2025)
- **Size:** 15,000 rows × 19 columns
- **Target Variable:** salary_usd (annual compensation in USD)
- **License:** Open licence (Kaggle)
### 📜 Credits
- **Dataset:** Global AI Job Market & Salary Trends 2025 by Bismasajjad on Kaggle

## 🛠️ Tools & Libraries
- Python, Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn (ML)
- Apache Spark / PySpark MLlib (HPC)
- Google Colab

## 📊 Exploratory Data Analysis
Key findings from EDA:
- Experience level is the strongest salary predictor (Entry: ~$60k → Executive: ~$178k)
- Geographic location shows significant variation — Switzerland ($153k) vs Ireland ($74k)
- Remote work ratio has minimal impact on salary
- Python and SQL are most demanded skills but not highest paying

## 🤖 Machine Learning — Neural Network MLP
Implemented a Multilayer Perceptron (MLP) Regressor for salary prediction.

**Architecture:**
| Parameter | Value |
|-----------|-------|
| Input layer | 24 features |
| Hidden layer 1 | 100 neurons |
| Hidden layer 2 | 50 neurons |
| Activation | ReLU |
| Solver | Adam |
| Max iterations | 500 |

**Results:**
| Metric | Value |
|--------|-------|
| R² | 0.6833 |
| MAE | $25,945 |
| RMSE | $37,639 |

## ⚡ HPC Implementation — PySpark RF + PCA
Implemented a distributed Random Forest Regressor using PCA-reduced 
features in Apache Spark.

**PCA:** Reduced feature vector to 5 principal components (49% variance explained)

**Results:**
| Metric | Value |
|--------|-------|
| R² | 0.5666 |
| MAE | $29,292 |
| RMSE | $40,790 |
| Training Time | 33.08s |

## 💡 Key Findings
- Experience level and geographic location are the strongest salary drivers
- Remote work ratio has no meaningful impact on AI salary
- Neural Network achieved R²=0.6833 — reasonable predictive capability
- PCA dimensionality reduction (49% variance) caused performance trade-off in RF+PCA
- Feature engineering quality impacts model performance more than framework choice

## 📸 Screenshots
### EDA 
<img width="1389" height="515" alt="2" src="https://github.com/user-attachments/assets/39df93cb-3ac1-4d61-bc11-4a2e1d9a1505" />
<img width="1587" height="719" alt="6" src="https://github.com/user-attachments/assets/6a30e0cc-9e5c-4865-b5b7-4e838a6676b5" />


### Neural Network Results 
<img width="832" height="235" alt="11" src="https://github.com/user-attachments/assets/04980f8c-9300-4dea-95bd-123433656aeb" />

### PySpark Results 
<img width="852" height="342" alt="12" src="https://github.com/user-attachments/assets/5b60ce02-9de9-40b7-ba60-a4833aa512be" />


## ⚠️ Disclaimer
This project was developed for educational purposes as part of the 
MSc Data Science and Analytics programme at Brunel University London.

## 👤 Author
**Gaural Rathod**  
MSc Data Science and Analytics — Brunel University London
