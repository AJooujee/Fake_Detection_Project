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
from wordcloud import WordCloud
import seaborn as sns

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
# Convert TF-IDF features to a dense format for both datasets
fake_features = tfidf.fit_transform(fake_df['text']).toarray()  # Extract features from Fake.csv
real_features = tfidf.fit_transform(real_df['text']).toarray()  # Extract features from True.csv

# Get the feature names from TF-IDF
feature_names = tfidf.get_feature_names_out()

# Create DataFrames for Fake and Real news features
fake_features_df = pd.DataFrame(fake_features, columns=feature_names)
real_features_df = pd.DataFrame(real_features, columns=feature_names)

# Save the DataFrames to CSV files
fake_features_df.to_csv("Fake_TF_IDF_Features.csv", index=False)
real_features_df.to_csv("Real_TF_IDF_Features.csv", index=False)

print("Step 2.2: Extracted features saved to CSV.")

# 2.3 Compare features (incomplete).
# Get the feature names from TF-IDF
fake_terms = set(fake_features_df.columns)
real_terms = set(real_features_df.columns)

# Find unique terms
unique_fake_terms = fake_terms - real_terms
unique_real_terms = real_terms - fake_terms

# Find overlapping terms
overlapping_terms = fake_terms & real_terms

# Save the comparisons to CSV for analysis
unique_fake_df = pd.DataFrame({'Unique Fake Terms': list(unique_fake_terms)})
unique_real_df = pd.DataFrame({'Unique Real Terms': list(unique_real_terms)})
overlapping_df = pd.DataFrame({'Overlapping Terms': list(overlapping_terms)})

unique_fake_df.to_csv("Unique_Fake_Terms.csv", index=False)
unique_real_df.to_csv("Unique_Real_Terms.csv", index=False)
overlapping_df.to_csv("Overlapping_Terms.csv", index=False)

print("Step 2.3: Unique and overlapping terms saved to CSV.")

# 2.4 Create visualizations (incomplete).
# 2.4.1: Generate Word Clouds
# Word cloud for fake news
fake_text = " ".join(fake_df['text'])
fake_wordcloud = WordCloud(width=800, height=400, background_color="white").generate(fake_text)
plt.figure(figsize=(10, 5))
plt.imshow(fake_wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud - Fake News")
plt.show()

# Word cloud for real news
real_text = " ".join(real_df['text'])
real_wordcloud = WordCloud(width=800, height=400, background_color="white").generate(real_text)
plt.figure(figsize=(10, 5))
plt.imshow(real_wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud - Real News")
plt.show()

# 2.4.2: Generate Bar Charts for Top Terms
# Get the top 20 terms from fake and real news
top_fake_terms = fake_features_df.sum(axis=0).sort_values(ascending=False).head(20)
top_real_terms = real_features_df.sum(axis=0).sort_values(ascending=False).head(20)

# Bar chart for fake news
plt.figure(figsize=(10, 6))
top_fake_terms.plot(kind='bar', title="Top 20 Terms - Fake News")
plt.ylabel("TF-IDF Score")
plt.show()

# Bar chart for real news
plt.figure(figsize=(10, 6))
top_real_terms.plot(kind='bar', title="Top 20 Terms - Real News")
plt.ylabel("TF-IDF Score")
plt.show()

# 2.4.3: Generate Heatmap for Co-occurrence in Fake News
# Create a co-occurrence matrix
fake_cooccurrence = fake_features_df.T.dot(fake_features_df)
plt.figure(figsize=(12, 10))
sns.heatmap(fake_cooccurrence.iloc[:20, :20], cmap="Blues", xticklabels=20, yticklabels=20)
plt.title("Co-occurrence Heatmap - Fake News (Top 20 Terms)")
plt.show()

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
def get_user_input():
    print("Enter a news headline or short article to check if it's fake news.")
    user_text = input("Your text: ")
    return user_text

# 5.2 Preprocess input text.
# Placeholder: Use the same cleaning pipeline to preprocess the input text.
def preprocess_input(user_text):
    return clean_text(user_text)

# 5.3 Predict and display the result.
# Placeholder: Use the best-trained model to predict and display the likelihood of fake news.
def predict_fake_news(user_text):
    # Preprocess the input text
    preprocessed_text = preprocess_input(user_text)
    # Transform the text using the TF-IDF vectorizer
    transformed_text = tfidf.transform([preprocessed_text])
    # Predict using the best model
    prediction = best_log_reg.predict(transformed_text)
    probability = best_log_reg.predict_proba(transformed_text)
    # Display results
    if prediction[0] == 1:
        print(f"The input text is classified as FAKE NEWS with {probability[0][1]*100:.2f}% confidence.")
    else:
        print(f"The input text is classified as REAL NEWS with {probability[0][0]*100:.2f}% confidence.")

# Run the input-based system
user_text = get_user_input()
predict_fake_news(user_text)

# Step 5.4: Visualize the confusion matrix for the best model.
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(best_log_reg, X_test_tfidf, y_test, cmap=plt.cm.Blues)
plt.title("Confusion Matrix for Tuned Logistic Regression")
plt.show()
