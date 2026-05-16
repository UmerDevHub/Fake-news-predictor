# Final Report: End-to-End Fake News Detection System

**Course:** Machine Learning (CSC354)  
**Instructor:** Yasmeen Khaliq  
**Submission Deadline:** 17th May, 2026  

---

## Abstract
This report details the development of an end-to-end Machine Learning solution designed to classify news articles as Real or Fake. The project fulfills all nine phases of the CSC354 semester project requirements, culminating in a deployed Streamlit web application.

---

## Phase 1: Problem Definition
**Problem Statement:** The rapid spread of misinformation requires automated systems to verify news credibility. This project tackles this by classifying text articles as Real (0) or Fake (1).
- **Problem Type:** Binary Classification
- **Input Features:** Article Title, Article Text, Engineered Text Features (word count, uppercase count, etc.), TF-IDF Vectors.
- **Target Variable:** `label` (Binary)

---

## Phase 2: Dataset Collection & Understanding
- **Source:** Kaggle (Fake and Real News Dataset)
- **Size:** 44,898 total records (exceeds the 1000 minimum).
- **Class Distribution:** 21,417 Real articles (47.7%) and 23,481 Fake articles (52.3%). The dataset is well-balanced.
- **Challenges Identified:** Unstructured text data requiring heavy NLP cleaning; variations in document length; presence of noise like URLs and HTML tags.

---

## Phase 3: Data Preprocessing
To prepare the data for modeling, the following pipeline was implemented (`data_preprocessing.py`):
1. **Missing Values & Duplicates:** Dropped using `dropna()` and `drop_duplicates()`.
2. **Text Cleaning:** Converted text to lowercase, removed URLs, special characters, and numbers.
3. **NLP Processing:** Removed English stopwords using NLTK and applied WordNet Lemmatization to reduce words to their base forms.
4. **Encoding Categorical Features:** Applied `LabelEncoder` to the `subject` category.
5. **Outlier Detection:** Used the IQR (Interquartile Range) method on text lengths to remove anomalously short or long articles.
6. **Normalization:** Applied `MinMaxScaler` to numeric features (like word counts) to bring them into a [0,1] range.
7. **Train-Test Split:** Split the dataset 80/20 for training and testing.

---

## Phase 4: Exploratory Data Analysis (EDA)
Comprehensive EDA was performed (`eda.py`) to understand data patterns:
- **Statistical Summary:** Generated using `describe()`.
- **Histograms:** Visualized the text length distribution between fake and real news.
- **Boxplots:** Highlighted word count differences, showing that fake news articles often have different length distributions compared to real news.
- **Scatter Plot:** Plotted text length vs. word count to observe the linear relationship and class clustering.
- **Correlation Heatmap:** Visualized relationships between numeric engineered features and the target label.

*(Note: Visualizations are accessible interactively in the Streamlit app under the "EDA" tab).*

---

## Phase 5: Feature Engineering
To capture stylometric differences between fake and real news, new features were created (`feature_engineering.py`):
- **New Feature Creation:** `text_length`, `word_count`, `avg_word_length`, `uppercase_count`, and `has_exclamation`. Fake news often relies on shorter texts, ALL CAPS, and exclamation marks to trigger emotion.
- **Vectorization:** Converted cleaned text to numerical form using `TfidfVectorizer` (max 5000 features).
- **Feature Selection:** Applied the Chi-Squared ($\chi^2$) test (`SelectKBest`) to select the top 1000 most relevant text features.
- **Dimensionality Reduction:** Applied Truncated SVD (suitable for sparse matrices) to reduce the feature space to 100 latent dimensions, capturing semantic meaning while reducing noise.

---

## Phase 6: Model Building
Three distinct classification algorithms were implemented (`model_training.py`).
1. **Logistic Regression:** Chosen for its interpretability and strong performance on high-dimensional sparse data like TF-IDF. It provides probability confidence scores.
2. **Multinomial Naive Bayes:** A standard baseline for NLP text classification that works efficiently with word frequencies.
3. **Support Vector Machine (LinearSVC):** Selected because SVMs excel in finding optimal hyperplanes in spaces where the number of features is very large.

---

## Phase 7: Model Evaluation
Models were evaluated on the test set using standard classification metrics (`model_evaluation.py`):
- **Metrics Computed:** Accuracy, Precision, Recall, and F1-Score.
- **Confusion Matrices:** Visualized for all three models to assess False Positives and False Negatives.
- **Conclusion:** The models were compared programmatically, and the system automatically highlights the best performing model based on test accuracy. (Typically, SVM or Logistic Regression perform exceptionally well on this dataset).

---

## Phase 8: Deployment / Interface
The project was deployed using **Streamlit** (`app.py`), providing a highly interactive UI:
- **Prediction:** Users can paste text OR provide a News URL. The app scrapes the URL using `BeautifulSoup`, cleans the text, and displays the prediction (Real/Fake) along with a confidence score.
- **EDA Dashboard:** Renders all matplotlib/seaborn plots dynamically.
- **Preprocessing Tab:** Educates the user on exactly what data transformations, outlier removals, and dimensionality reductions occurred.
- **Performance Tab:** Shows evaluation metrics and confusion matrices.

---

## Phase 9: Conclusion
This project successfully designed and deployed an end-to-end ML pipeline for Fake News Detection. It satisfies all technical requirements, from complex NLP preprocessing and dimensionality reduction to the final deployment of a user-facing web application.

---
*(Please attach screenshots of the Streamlit App running on your computer below before submission)*

### Screenshots
*(Insert Screenshot of Prediction Page here)*
*(Insert Screenshot of EDA Page here)*
*(Insert Screenshot of Model Evaluation Page here)*
