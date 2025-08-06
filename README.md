# 🛡️ Credit Card Fraud Detection using Machine Learning
This project aims to build a machine learning model that detects fraudulent transactions in credit card datasets. Using real-world anonymized data, the model learns patterns of legitimate vs fraudulent behavior and predicts potential frauds with high accuracy.

# 📂 Dataset
The dataset used is the Kaggle Credit Card Fraud Detection dataset, which contains transactions made by European cardholders in September 2013.

Total transactions: 284,807

Fraudulent transactions: 492 (0.172%)

Features: 30 (anonymized via PCA except for Time, Amount, and Class)

# 🚀 Technologies Used
Python 🐍


Jupyter Notebook 📓


Pandas, NumPy


Matplotlib, Seaborn (for visualization)


Scikit-learn (for model building)


# 🧠 Machine Learning Models Used
Logistic Regression

Decision Tree Classifier

Random Forest Classifier

Evaluation metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC

# 📊 Model Evaluation
Due to class imbalance, performance was primarily evaluated using:

Precision

Recall

F1-Score

Confusion Matrix

ROC-AUC Curve

# 📈 Visualizations
Distribution of transaction amounts

Class distribution (Fraud vs Non-Fraud)

Correlation heatmap

Confusion matrices for each model

ROC curves

# 💡 Key Learnings
Real-world fraud detection involves highly imbalanced data.

Oversampling/undersampling techniques or anomaly detection models may further improve performance.

Precision-Recall tradeoff is crucial in such problems where false negatives are costly.

# ✅ To Do
Try SMOTE or ADASYN for handling imbalance

Hyperparameter tuning

Test with more ML models like XGBoost or LightGBM

Save model using Pickle or Joblib for deployment

# 📌 Conclusion
This project demonstrates a basic approach to detecting credit card fraud using supervised machine learning. With further improvements and integration, this can be adapted to real-time fraud detection systems.

# 📬 Contact
If you have any feedback or questions, feel free to reach out.
