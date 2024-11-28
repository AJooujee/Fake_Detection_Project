# Import Libraries
# Step 1: Preprocessing the Data
# - Import necessary libraries for data manipulation, visualization, and machine learning.
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import sys

# Step 1: Preprocessing the Data
# 1.1 Load datasets, combine them, and shuffle.
real_df = pd.read_csv('True.csv')
fake_df = pd.read_csv('Fake.csv')

# 1.2 Add labels: 0 for real news, 1 for fake news.
real_df['label'] = 0
fake_df['label'] = 1

# 1.3 Combine the datasets and shuffle.
df = pd.concat([real_df, fake_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 1.4 Display dataset information and class distribution.
print("Dataset Information:")
print(df.info())
print("Class Distribution:\n", df['label'].value_counts())

# 1.5 Text cleaning: lowercase, remove punctuation and numbers.
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return text

# Apply text cleaning.
df['text'] = df['text'].apply(clean_text)

# 1.6 Split data into training and testing sets.
X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Feature Engineering
# 2.1 Extract common features using TF-IDF.
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# 2.2 Save extracted features to CSV (incomplete).
# Placeholder: Save extracted TF-IDF features to CSV files for further analysis.
# Code goes here.

# 2.3 Compare features (incomplete).
# Placeholder: Compare unique and overlapping features between Fake.csv and True.csv.
# Code goes here.

# 2.4 Create visualizations (incomplete).
# Placeholder: Create word clouds, bar charts, and heatmaps for the features.
# Visualization code goes here.

# Step 3: Model Training and Evaluation
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    print(f"Evaluation Metrics for {model_name}:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\n")

# 3.1 Train Logistic Regression.
log_reg = LogisticRegression()
log_reg.fit(X_train_tfidf, y_train)
evaluate_model(log_reg, X_test_tfidf, y_test, "Logistic Regression")

# 3.2 Train Support Vector Machine (SVM).
svm = SVC(probability=True)
svm.fit(X_train_tfidf, y_train)
evaluate_model(svm, X_test_tfidf, y_test, "Support Vector Machine")

# 3.3 Train Random Forest model.
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_tfidf, y_train)
evaluate_model(rf, X_test_tfidf, y_test, "Random Forest")

# Step 4: Hyperparameter Tuning
# 4.1 Perform hyperparameter tuning for Logistic Regression.
param_grid = {'C': [0.01, 0.1, 1, 10]}
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid_search.fit(X_train_tfidf, y_train)

# 4.2 Evaluate the tuned Logistic Regression model.
best_log_reg = grid_search.best_estimator_
print("Best Parameters for Logistic Regression:", grid_search.best_params_)
evaluate_model(best_log_reg, X_test_tfidf, y_test, "Tuned Logistic Regression")

# Step 5: Input-Based Prediction System (incomplete).
# 5.1 Accept user input.
# Placeholder: Create a system to accept user-provided text (e.g., a headline).
# Code goes here.

# 5.2 Preprocess input text.
# Placeholder: Use the same cleaning pipeline to preprocess the input text.
# Code goes here.

# 5.3 Predict and display the result.
# Placeholder: Use the best-trained model to predict and display the likelihood of fake news.
# Code goes here.

# Step 5.4: Visualize the confusion matrix for the best model.
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(best_log_reg, X_test_tfidf, y_test, cmap=plt.cm.Blues)
plt.title("Confusion Matrix for Tuned Logistic Regression")
plt.show()
