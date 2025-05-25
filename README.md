# 🎓 Advanced Sentiment Analysis on Student Feedback

This repository presents a complete pipeline for performing sentiment analysis on student feedback data using Natural Language Processing (NLP) techniques and ensemble machine learning models. It integrates advanced text preprocessing, feature engineering, and class balancing using SMOTE to produce high-accuracy predictions of student sentiments.

---

## 🔍 Overview

📌 **Objective**: To analyze and classify student feedback into sentiment categories using a combination of text preprocessing, TF-IDF feature extraction, and an ensemble of machine learning models.

📌 **Tech Stack**: Python · Scikit-learn · NLTK · VADER · TF-IDF · SMOTE · Random Forest · Linear SVM · SGDClassifier

📌 **Key Highlights**:
- Custom preprocessing with lemmatization and stopword removal
- Sentiment scoring using **VADER**
- TF-IDF vectorization with n-gram range (1,3) and 15,000 max features
- Balanced training using **SMOTE**
- Ensemble classification using **VotingClassifier**

---

## 🧠 Model Architecture

1. **Text Preprocessing**
   - Lowercasing
   - Removing special characters and punctuation
   - Tokenization
   - Stopword removal
   - Lemmatization

2. **Feature Engineering**
   - TF-IDF Vectorization (up to trigrams)
   - VADER Sentiment Scores: `neg`, `neu`, `pos`, `compound`
   - Length-based features: text length, word count

3. **Class Balancing**
   - Synthetic Minority Over-sampling Technique (**SMOTE**)

4. **Classification**
   - Ensemble Voting Classifier using:
     - `SGDClassifier (log loss)`
     - `LinearSVC`
     - `RandomForestClassifier`

---
