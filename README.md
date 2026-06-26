# Churn Prediction & Customer Segmentation

A machine learning project that predicts CUSTOMER CHURN using a **Random Forest Classifier** and segments customers into meaningful customer groups using **K-Means Clustering**. The project combines data preprocessing, predictive modeling, and clustering to provide business insights for customer retention.

---

## Dataset

- **Dataset:** Telco Customer Churn
- **Records:** 7,043 customers
- **Target Variable:** `Churn Value`

---

## Workflow

1. Performed Exploratory Data Analysis (EDA) to identify churn patterns.
2. Cleaned and preprocessed the dataset.
3. Converted categorical features using one-hot encoding.
4. Split the dataset into training (80%) and testing (20%).
5. Trained a Random Forest Classifier.
6. Improved recall using class balancing and hyperparameter tuning.
7. Evaluated the model using multiple performance metrics.
8. Performed feature importance analysis.
9. Validated the model using 5-Fold Cross Validation.
10. Generated churn probabilities for each customer.
11. Applied K-Means Clustering to segment customers.
12. Interpreted customer segments for business recommendations.

---

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

---

## Machine Learning Techniques

### Churn Prediction
- Random Forest Classifier
- Class Weight Balancing
- Hyperparameter Tuning
- Feature Importance Analysis
- 5-Fold Cross Validation
- ROC-AUC Analysis

### Customer Segmentation
- StandardScaler
- K-Means Clustering
- Elbow Method
- Cluster Profiling

---

## Model Performance

| Metric | Value |
|---------|------:|
| Accuracy | **77.57%** |
| Precision | **59.04%** |
| Recall | **75.25%** |
| F1-Score | **65.58%** |
| ROC-AUC Score | **0.857** |
| Cross Validation Accuracy | **77.94%** |
| Cross Validation Recall | **73.35%** |

---

## Customer Segments

| Cluster | Description |
|----------|-------------|
| Cluster 0 | Budget Loyal Customers |
| Cluster 1 | High Risk New Customers |
| Cluster 2 | Loyal Premium Customers |

---

## Key Insights

- Customers with shorter tenure showed a higher probability of churn.
- Monthly charges had a significant impact on customer churn.
- Contract type strongly influenced customer retention.
- Class balancing improved the model's ability to detect churning customers.
- K-Means clustering enabled targeted customer segmentation for retention strategies.

---

## Repository Structure

```text
Churn-Prediction-Customer-Segmentation/
│
├── data/
│   └── Telco_customer_churn.xlsx
│
├── notebooks/
│   └── churn_prediction_customer_segmentation.ipynb
│
├── src/
│   └── churn_prediction_customer_segmentation.py
│
├── requirements.txt
│
└── README.md
```

---

## How to Run

```bash
git clone <repository-url>
cd Churn-Prediction-Customer-Segmentation
pip install -r requirements.txt
```

Open the notebook and run all cells sequentially.

---

## Future Improvements

- Compare additional classification models (XGBoost, LightGBM, CatBoost)
- Perform automated hyperparameter tuning using GridSearchCV
- Handle class imbalance using SMOTE
- Deploy the model using Streamlit or Flask
- Build an interactive dashboard for business users

---

## Author

**~Avni Goel**
