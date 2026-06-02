# Fake News Detection System

## Getting Started

- Install dependencies: `pip install -r requirements.txt`
- Run the app: `streamlit run app.py`
- Source code is organized under `src/` and the Streamlit entry point is `app.py`

## 🚨 Fake News Detection

A compact, reproducible pipeline for classifying news articles as **Real** or **Fake** using NLP preprocessing, TF‑IDF text features, and classical classifiers (Logistic Regression, Naive Bayes, SVM). Includes a Streamlit UI for exploration and inference.

---

## ✨ Quick Start

1. Create and activate a virtual environment and install deps:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # PowerShell
pip install -r requirements.txt
```

2. Place your datasets in the `data/` folder:

- `data/Fake.csv` — fake news examples
- `data/True.csv` — real news examples

3. Run the Streamlit app:

```powershell
streamlit run app.py
```

---

## 🔎 What’s Included

- `app.py` — Streamlit UI for EDA, model comparison, and live prediction
- `src/data_preprocessing.py` — load, clean, and preprocess text
- `src/feature_engineering.py` — TF‑IDF + engineered numeric/text features
- `src/model_training.py` — train Logistic Regression, Naive Bayes, SVM
- `src/model_evaluation.py` — accuracy / precision / recall / F1 and confusion matrix plotting

---

## 📊 How to Get Model Accuracies

Run this small script (or follow the example in the app):

```python
from src.data_preprocessing import load_data, preprocess_data, split_data
from src.feature_engineering import create_tfidf_features
from src.model_training import train_all_models
from src.model_evaluation import compare_models

df = preprocess_data(load_data())
X = df['clean_text']; y = df['label']
X_train, X_test, y_train, y_test = split_data(X, y)
X_train_tfidf, X_test_tfidf, _ = create_tfidf_features(X_train, X_test)
models = train_all_models(X_train_tfidf, y_train)
results = compare_models(models, X_test_tfidf, y_test)
print(results)
```

This prints a table with `Accuracy`, `Precision`, `Recall`, and `F1-Score` for each model.

---

## ⚠️ Notes & Best Practices

- Large binary artifacts were removed from the Git history to comply with GitHub limits (100 MB). Do not commit large `.pkl` files; use `git-lfs` if needed.
- Recommended `.gitignore` entry: `/data/*.pkl`
- Keep raw CSVs in `data/` locally and do not push large processed artifacts.

---

## 🛠️ Development Tips

- Run unit tests on processing functions after changes to `clean_text()` or feature extraction.
- Keep models small for quicker iteration — use `max_features` or dimensionality reduction in TF‑IDF.

---

## 📝 License & Contact

- Author: Umer Nisar — umernisar053@gmail.com
- License: MIT

Feel free to open issues or PRs to improve the pipeline or UI.
# Fake News Detection System

## Getting Started


**Fake News Detection**

Detects whether a news article is real or fake using NLP preprocessing, TF‑IDF features, and classical classifiers (Logistic Regression, Naive Bayes, SVM).

**Why This Project**
- **Purpose:** Provide a reproducible, explainable pipeline for fake-news classification with an interactive Streamlit UI.
- **Audience:** Data scientists, students, and developers learning NLP classification and model evaluation.

**Contents**
- **Code:** Core logic lives under [src/](src/). Key modules:
    - [data_preprocessing.py](src/data_preprocessing.py#L1) — loading, cleaning, encoding, splitting.
    - [feature_engineering.py](src/feature_engineering.py#L1) — TF‑IDF, text features, feature selection.
    - [model_training.py](src/model_training.py#L1) — trains Logistic Regression, Naive Bayes, SVM.
    - [model_evaluation.py](src/model_evaluation.py#L1) — accuracy/precision/recall/F1 and confusion matrix plotting.
- **UI:** [app.py](app.py#L1) — Streamlit app for EDA, prediction, and model comparison.
- **Data:** the original CSV sources were `data/Fake.csv` and `data/True.csv` (not committed if large).

**Quick Start**
1. Create and activate a Python environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # PowerShell
pip install -r requirements.txt
```

2. Restore the dataset into the `data/` folder.
     - If `data/Fake.csv` and `data/True.csv` were removed from the repo (history clean), copy them back into `data/` before running training or evaluation.

3. Run the Streamlit UI locally:

```powershell
streamlit run app.py
```

4. Train or evaluate models programmatically (example):

```powershell
python -c "from src.data_preprocessing import load_data, preprocess_data; df=preprocess_data(load_data()); print(df.shape)"
```

**Where To Find Accuracies**
- Model evaluation utilities are in [src/model_evaluation.py](src/model_evaluation.py#L1). Use `compare_models()` to get a DataFrame of accuracies, precision, recall and F1 for trained models.

Example evaluation snippet:

```python
from src.data_preprocessing import load_data, preprocess_data, split_data
from src.feature_engineering import create_tfidf_features
from src.model_training import train_all_models
from src.model_evaluation import compare_models

df = preprocess_data(load_data())
X = df['clean_text']; y = df['label']
X_train, X_test, y_train, y_test = split_data(X, y)
X_train_tfidf, X_test_tfidf, _ = create_tfidf_features(X_train, X_test)
models = train_all_models(X_train_tfidf, y_train)
results = compare_models(models, X_test_tfidf, y_test)
print(results)
```

**Notes about large files & Git**
- Large pickle/artifact files (e.g., `data/saved_data.pkl`) were removed from history to keep the repo under GitHub's 100 MB limit. Use `git-lfs` for large binary artifacts if you need to track them.
- Add local artifacts to `.gitignore`: `/data/*.pkl`

**Project Structure (summary)**
- `app.py` — Streamlit entrypoint
- `requirements.txt` — Python dependencies
- `src/` — package with preprocessing, features, training, evaluation
- `data/` — place CSVs here (not tracked if large)

**Development & Contribution**
- Add tests and small, focused PRs. If you change data processing, include a note about changes to `clean_text()` or `preprocess_data()`.

**Contact & License**
- Author: Umer Nisar — umernisar053@gmail.com
- License: MIT (update if different)

Enjoy exploring and improving the fake-news classifier — open an issue or PR for enhancements.


## Phase 1: Problem Definition

**Problem Statement:** Detecting whether a news article is Real or Fake using Natural Language Processing and Machine Learning techniques.

**Problem Type:** Binary Classification

**Input Features:** News article text (title + body)

**Target Variable:** Label (0 = Real News, 1 = Fake News)

## Phase 2: Dataset

**Source:** Kaggle - Fake and Real News Dataset
**Link:** https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

**Dataset Size:** ~44,898 records (23,481 Fake + 21,417 True)

**Features:**
- title: News article title
- text: News article body
- subject: News category (politicsNews, worldnews, politics, Government News, US_News, left-news, Middle-east)
- date: Publication date

**Class Distribution:**
- Real News: ~21,417 articles (47.7%)
- Fake News: ~23,481 articles (52.3%)
- Approximately balanced dataset

**Challenges:**
- Text data requires NLP preprocessing (tokenization, stopword removal, lemmatization)
- Variable article lengths with potential outliers
- Some articles contain URLs, special characters, and HTML artifacts

## Phase 3: Data Preprocessing

| Step | Technique | Implementation |
|------|-----------|----------------|
| 1 | Duplicate Removal | `drop_duplicates()` |
| 2 | Missing Value Handling | `dropna()` |
| 3 | Text Cleaning | Lowercasing, URL removal, special character removal |
| 4 | Stopword Removal | NLTK English stopwords |
| 5 | Lemmatization | WordNetLemmatizer |
| 6 | Categorical Encoding | LabelEncoder on 'subject' column |
| 7 | Outlier Detection | IQR method on text length — removes articles outside Q1-1.5×IQR to Q3+1.5×IQR |
| 8 | Normalization | Min-Max scaling on numeric features (text_length, word_count, etc.) |
| 9 | Train-Test Split | 80/20 split with random_state=42 |

## Phase 4: Exploratory Data Analysis

Visualizations performed:
- **Histogram:** Text length distribution by class
- **Boxplot:** Word count distribution by class
- **Heatmap:** Correlation matrix of numeric features
- **Scatter Plot:** Text length vs word count colored by class
- **Bar Chart:** Class distribution and subject distribution

Statistical summary with `df.describe()` is also displayed.

## Phase 5: Feature Engineering

### Derived Features
| Feature | Rationale |
|---------|-----------|
| `text_length` | Fake news tends to be shorter with less factual detail |
| `word_count` | Real articles typically have higher word counts from thorough reporting |
| `avg_word_length` | Fake news uses simpler, shorter words for emotional impact |
| `uppercase_count` | Fake news uses ALL CAPS for sensationalism and attention-grabbing |
| `has_exclamation` | Exclamation marks indicate emotionally charged, sensational writing |

### TF-IDF Vectorization
- 5000 max features for initial dimensionality control
- Inherently normalizes word frequencies

### Feature Selection
- **Chi-Squared (χ²) Test:** Selects top 1000 most discriminative features from 5000 TF-IDF features
- Measures dependence between each word feature and the fake/real target

### Dimensionality Reduction
- **Truncated SVD:** Reduces 5000 features to 100 latent semantic dimensions
- Captures hidden relationships between words while removing noise

## Phase 6: Model Building

Three classification algorithms implemented:

| Model | Why Selected |
|-------|-------------|
| **Logistic Regression** | Works well with high-dimensional sparse TF-IDF data; provides probability estimates for confidence; highly interpretable feature coefficients |
| **Multinomial Naive Bayes** | Designed for text/word frequency data; assumes feature independence; very fast training and prediction |
| **SVM (LinearSVC)** | Finds maximum-margin separating hyperplane; excels in high-dimensional spaces where features outnumber samples |

## Phase 7: Model Evaluation

Metrics used:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix (visualized as heatmaps)

All models are compared in a table and the best model is automatically selected.

## Phase 8: Deployment

**Technology:** Streamlit

**Application Features:**
- **Prediction Page:** Enter text or paste a URL to check if news is fake or real
- **EDA Page:** Interactive visualizations of dataset analysis
- **Preprocessing & Features Page:** Shows all preprocessing steps, outlier detection results, feature selection scores, and dimensionality reduction analysis
- **Model Performance Page:** Model comparison table, justifications, and confusion matrices
- Input validation and error handling
- URL scraping with BeautifulSoup

## Project Structure

```
FAKE NEWS DETECTION/
├── app.py                    # Phase 8: Streamlit Interface
├── data_preprocessing.py     # Phase 3: Data Preprocessing
├── eda.py                    # Phase 4: Exploratory Data Analysis
├── feature_engineering.py    # Phase 5: Feature Engineering
├── model_training.py         # Phase 6: Model Building
├── model_evaluation.py       # Phase 7: Model Evaluation
├── requirements.txt          # Dependencies
├── README.md                 # Documentation
└── data/
    ├── Fake.csv              # Fake news dataset
    └── True.csv              # Real news dataset
```

## Setup Instructions

1. Download dataset from Kaggle and place `Fake.csv` and `True.csv` in the `data/` folder
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `streamlit run app.py`
