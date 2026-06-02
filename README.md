# Fake News Detection System

## Getting Started

- Install dependencies: `pip install -r requirements.txt`
- Run the app: `streamlit run app.py`
- Source code is organized under `src/` and the Streamlit entry point is `app.py`

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
