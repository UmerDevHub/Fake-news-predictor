import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import requests
from bs4 import BeautifulSoup
from data_preprocessing import load_data, preprocess_data, clean_text, split_data, detect_outliers, normalize_features
from feature_engineering import (create_tfidf_features, create_text_features,
                                  select_features_chi2, apply_dimensionality_reduction)
from model_training import train_all_models
from model_evaluation import compare_models, plot_confusion_matrix, get_best_model
from eda import (class_distribution, text_length_histogram, word_count_boxplot,
                 correlation_heatmap, scatter_plot, subject_distribution)

st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="wide")

st.title("📰 Fake News Detection System")

if not os.path.exists('data/Fake.csv') or not os.path.exists('data/True.csv'):
    st.error("Dataset not found! Please download Fake.csv and True.csv from Kaggle and place them in the 'data' folder.")
    st.info("Download from: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset")
    st.stop()


@st.cache_data
def load_and_process():
    data_path = 'data/saved_data.pkl'
    if os.path.exists(data_path):
        with open(data_path, 'rb') as f:
            return pickle.load(f)
            
    df_raw = load_data()
    df = preprocess_data(df_raw)
    df = create_text_features(df)
    # Outlier detection using IQR method
    df_clean, outliers, outlier_stats = detect_outliers(df)
    # Normalize numeric features
    numeric_cols = ['text_length', 'word_count', 'avg_word_length', 'uppercase_count']
    df_normalized, scaler = normalize_features(df_clean.copy(), numeric_cols)
    
    with open(data_path, 'wb') as f:
        pickle.dump((df_clean, outlier_stats), f)
        
    return df_clean, outlier_stats


@st.cache_resource
def build_models(_df):
    model_path = 'data/saved_models.pkl'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)

    X_train, X_test, y_train, y_test = split_data(_df['clean_text'], _df['label'])
    X_train_tfidf, X_test_tfidf, tfidf = create_tfidf_features(X_train, X_test)
    # Feature Selection using Chi-Squared test
    X_train_sel, X_test_sel, selector, top_features = select_features_chi2(
        X_train_tfidf, y_train, X_test_tfidf, tfidf, k=1000
    )
    # Dimensionality Reduction using Truncated SVD
    X_train_svd, X_test_svd, svd, explained_var = apply_dimensionality_reduction(
        X_train_tfidf, X_test_tfidf, n_components=100
    )
    # Train models on original TF-IDF features (best performance)
    models = train_all_models(X_train_tfidf, y_train)
    
    artifacts = (models, tfidf, X_test_tfidf, y_test, top_features, explained_var)
    with open(model_path, 'wb') as f:
        pickle.dump(artifacts, f)
        
    return artifacts


with st.spinner("Loading and processing data..."):
    df, outlier_stats = load_and_process()

with st.spinner("Training models..."):
    models, tfidf, X_test_tfidf, y_test, top_features, explained_var = build_models(df)

page = st.sidebar.selectbox("Navigation", [
    "🔍 Prediction", "📊 EDA", "🔧 Preprocessing & Features", "📈 Model Performance"
])

if page == "🔍 Prediction":
    st.header("Check News Authenticity")

    tab1, tab2 = st.tabs(["Enter Text", "Enter URL"])

    with tab1:
        news_input = st.text_area("Paste news article text here:", height=200)
        predict_btn = st.button("Check News", key="text_btn")

        if predict_btn and news_input:
            cleaned = clean_text(news_input)
            features = tfidf.transform([cleaned])

            st.subheader("Prediction Results")
            for name, model in models.items():
                prediction = model.predict(features)[0]
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(features)[0]
                    confidence = max(proba) * 100
                    score_text = f" | Confidence: {confidence:.1f}%"
                else:
                    score_text = ""

                if prediction == 1:
                    st.error(f"{name}: ❌ FAKE NEWS{score_text}")
                else:
                    st.success(f"{name}: ✅ REAL NEWS{score_text}")

    with tab2:
        url_input = st.text_input("Paste news URL here:")
        url_btn = st.button("Check URL", key="url_btn")

        if url_btn and url_input:
            try:
                response = requests.get(url_input, timeout=10)
                soup = BeautifulSoup(response.text, 'html.parser')
                paragraphs = soup.find_all('p')
                article_text = ' '.join([p.get_text() for p in paragraphs])

                if len(article_text) < 50:
                    st.warning("Could not extract enough text from this URL.")
                else:
                    st.text_area("Extracted Text:", article_text[:1000], height=150, disabled=True)
                    cleaned = clean_text(article_text)
                    features = tfidf.transform([cleaned])

                    st.subheader("Prediction Results")
                    for name, model in models.items():
                        prediction = model.predict(features)[0]
                        if hasattr(model, 'predict_proba'):
                            proba = model.predict_proba(features)[0]
                            confidence = max(proba) * 100
                            score_text = f" | Confidence: {confidence:.1f}%"
                        else:
                            score_text = ""

                        if prediction == 1:
                            st.error(f"{name}: ❌ FAKE NEWS{score_text}")
                        else:
                            st.success(f"{name}: ✅ REAL NEWS{score_text}")
            except Exception as e:
                st.error(f"Could not fetch URL: {str(e)}")

elif page == "📊 EDA":
    st.header("Exploratory Data Analysis")

    st.subheader("Dataset Overview")
    st.write(f"**Total Records:** {len(df)}")
    st.write(f"**Features:** {list(df.columns)}")
    st.dataframe(df.head(10))

    st.subheader("Statistical Summary")
    st.dataframe(df.describe())

    st.subheader("Class Distribution")
    col1, col2 = st.columns(2)
    with col1:
        label_counts = df['label'].value_counts()
        st.write(f"- Real News: **{label_counts.get(0, 0)}** articles")
        st.write(f"- Fake News: **{label_counts.get(1, 0)}** articles")
        st.pyplot(class_distribution(df))
    with col2:
        st.subheader("Subject Distribution")
        st.pyplot(subject_distribution(df))

    st.subheader("Text Length Distribution (Histogram)")
    st.pyplot(text_length_histogram(df))

    st.subheader("Word Count Distribution (Boxplot)")
    st.pyplot(word_count_boxplot(df))

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Correlation Heatmap")
        st.pyplot(correlation_heatmap(df))
    with col4:
        st.subheader("Text Length vs Word Count (Scatter)")
        st.pyplot(scatter_plot(df))

elif page == "🔧 Preprocessing & Features":
    st.header("Data Preprocessing & Feature Engineering")

    # --- Preprocessing Steps ---
    st.subheader("Phase 3: Data Preprocessing Steps")
    st.markdown("""
    | Step | Technique | Description |
    |------|-----------|-------------|
    | 1 | Duplicate Removal | `drop_duplicates()` — removed identical records |
    | 2 | Missing Value Handling | `dropna()` — removed rows with null values |
    | 3 | Text Cleaning | Lowercasing, URL removal, special character removal |
    | 4 | Stopword Removal | Removed common English stopwords using NLTK |
    | 5 | Lemmatization | Reduced words to base form using WordNetLemmatizer |
    | 6 | Categorical Encoding | LabelEncoder applied to 'subject' column |
    | 7 | Outlier Detection | IQR method on text length (see below) |
    | 8 | Normalization | Min-Max scaling applied to numeric features |
    """)

    # --- Outlier Detection Results ---
    st.subheader("Outlier Detection (IQR Method)")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Records Analyzed", outlier_stats['total_records'])
        st.metric("Outliers Detected", outlier_stats['total_outliers'])
        st.metric("Outlier Percentage", f"{outlier_stats['outlier_percentage']}%")
    with col2:
        st.metric("Q1 (25th percentile)", outlier_stats['Q1'])
        st.metric("Q3 (75th percentile)", outlier_stats['Q3'])
        st.metric("IQR", outlier_stats['IQR'])
    st.info(f"**Bounds:** Lower = {outlier_stats['lower_bound']}, Upper = {outlier_stats['upper_bound']}. "
            f"Articles outside these bounds were removed as outliers.")

    # --- Feature Engineering ---
    st.subheader("Phase 5: Feature Engineering")
    st.markdown("""
    **Derived Features Created:**

    | Feature | Rationale |
    |---------|-----------|
    | `text_length` | Fake news tends to be shorter with less factual detail |
    | `word_count` | Real articles typically have higher word counts |
    | `avg_word_length` | Fake news uses simpler, shorter words for emotional impact |
    | `uppercase_count` | Fake news uses ALL CAPS for sensationalism |
    | `has_exclamation` | Exclamation marks indicate emotionally charged writing |
    """)

    # --- Feature Selection ---
    st.subheader("Feature Selection (Chi-Squared Test)")
    st.write("Top 20 most discriminative words selected by χ² test from 5000 TF-IDF features:")
    st.dataframe(top_features.reset_index(drop=True), use_container_width=True)
    st.info("Chi-squared feature selection reduced 5000 TF-IDF features to the top 1000 most informative features.")

    # --- Dimensionality Reduction ---
    st.subheader("Dimensionality Reduction (Truncated SVD)")
    st.metric("Explained Variance (100 components)", f"{explained_var * 100:.2f}%")
    st.write(f"Truncated SVD reduced 5000 TF-IDF features to 100 latent dimensions, "
             f"capturing **{explained_var * 100:.2f}%** of the total variance.")
    st.info("Truncated SVD (similar to PCA for sparse matrices) captures latent semantic "
            "relationships between words, reducing noise while preserving important patterns.")

elif page == "📈 Model Performance":
    st.header("Model Evaluation")

    results = compare_models(models, X_test_tfidf, y_test)

    st.subheader("Model Comparison")
    st.dataframe(results, use_container_width=True)

    best = get_best_model(results)
    st.success(f"🏆 Best Model: **{best['Model']}** with Accuracy: **{best['Accuracy']}**")

    st.subheader("Why These Models?")
    st.markdown("""
    | Model | Justification |
    |-------|---------------|
    | **Logistic Regression** | Works well with high-dimensional sparse TF-IDF data; provides probability estimates; highly interpretable |
    | **Naive Bayes** | Designed for text/word frequency data; assumes feature independence; very fast training |
    | **SVM (LinearSVC)** | Finds maximum-margin hyperplane; excels in high-dimensional spaces like TF-IDF |
    """)

    st.subheader("Confusion Matrices")
    cols = st.columns(3)
    for i, (name, model) in enumerate(models.items()):
        with cols[i]:
            st.pyplot(plot_confusion_matrix(model, X_test_tfidf, y_test, name))
