# Problem Proposal: End-to-End Fake News Detection System

**Course:** Machine Learning (CSC354)  
**Instructor:** Yasmeen Khaliq  
**Submission Deadline:** 17th May, 2026  

---

## 1. Problem Statement

With the rapid expansion of digital media and social networks, the spread of misinformation and "fake news" has become a critical global issue. Fake news can manipulate public opinion, disrupt social harmony, and influence political outcomes. The vast volume of articles published daily makes manual fact-checking impossible. 

The objective of this project is to design, implement, and deploy an end-to-end Machine Learning pipeline capable of automatically classifying news articles as either **Real** or **Fake**. This system will leverage Natural Language Processing (NLP) techniques to process text data and train classification algorithms to identify linguistic patterns, stylometry, and structural differences that distinguish credible journalism from deceptive content.

## 2. Problem Characteristics

* **Problem Type:** Binary Classification (Supervised Learning)
* **Domain:** Natural Language Processing (NLP) / Text Classification
* **Objective Function:** Maximize classification accuracy while minimizing false positives and false negatives (measured via F1-Score).

## 3. Dataset Identification

We will utilize the **"Fake and Real News Dataset"** available publicly on Kaggle. 

* **Source:** [Kaggle - Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
* **Size:** Approximately 44,898 records, well exceeding the project's minimum requirement of 1,000 records.
* **Composition:** 
  * ~21,417 Real news articles
  * ~23,481 Fake news articles
* **Features Provided:**
  * `title`: The headline of the news article.
  * `text`: The main body content of the article.
  * `subject`: The category of the news (e.g., politics, world news).
  * `date`: The date the article was published.

## 4. Feature Definition

**Input Features (X):**
The primary input will be the textual content of the articles. We will combine the `title` and `text` fields to form a comprehensive text feature. Additional engineered features will include:
* **Text Length & Word Count:** To capture differences in article verbosity.
* **Average Word Length:** To analyze vocabulary complexity.
* **Uppercase Character Count:** To detect sensationalism (often present in fake news).
* **Punctuation Presence (e.g., exclamation marks):** To capture emotional tone.
* **TF-IDF Vectors:** The text will be vectorized using Term Frequency-Inverse Document Frequency (TF-IDF) to represent the importance of specific words.

**Target Variable (Y):**
* `label`: A binary indicator where **0 represents Real News** and **1 represents Fake News**.

## 5. Proposed Methodology

The project will follow a complete ML lifecycle:
1. **Data Preprocessing:** Cleaning text, removing stop words, lemmatization, handling outliers, and normalizing numeric features.
2. **Exploratory Data Analysis (EDA):** Visualizing distributions (histograms, boxplots) and feature correlations (heatmaps, scatter plots).
3. **Feature Engineering:** Creating derived features, feature selection (Chi-squared test), and dimensionality reduction (Truncated SVD).
4. **Model Training:** Training at least three classification algorithms (e.g., Logistic Regression, Naive Bayes, Support Vector Machines).
5. **Evaluation:** Comparing models using Accuracy, Precision, Recall, F1-Score, and Confusion Matrices to select the best performing model.
6. **Deployment:** Developing a user-friendly web application using Streamlit that allows users to input text or a URL to get real-time authenticity predictions.

## 6. Expected Impact

A successful deployment of this model will demonstrate the practical application of machine learning in solving a prevalent real-world problem. It will provide users with a tool to quickly assess the credibility of news content, promoting media literacy and reducing the spread of misinformation.
