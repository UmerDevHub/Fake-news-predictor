import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import TruncatedSVD


def create_text_features(df):
    # creating extra columns in the dataset for text features
    df['text_length'] = df['clean_text'].apply(len)
    df['word_count'] = df['clean_text'].apply(lambda x: len(str(x).split()))
    
    # Calculate average word length
    df['avg_word_length'] = df['clean_text'].apply(
        lambda x: np.mean([len(w) for w in str(x).split()]) if len(str(x).split()) > 0 else 0
    )
    
    # Count uppercase characters and exclamation marks for fake news detection
    df['uppercase_count'] = df['text'].apply(lambda x: sum(1 for c in str(x) if c.isupper()))
    df['has_exclamation'] = df['text'].apply(lambda x: 1 if '!' in str(x) else 0)
    
    return df


def create_tfidf_features(X_train, X_test, max_features=5000):
    # Create TF-IDF Vectorizer object
    tfidf = TfidfVectorizer(max_features=max_features)
    
    # Train and transform the training set
    X_train_tfidf = tfidf.fit_transform(X_train)
    
    # Transform the test set
    X_test_tfidf = tfidf.transform(X_test)
    
    return X_train_tfidf, X_test_tfidf, tfidf


def select_features_chi2(X_train, y_train, X_test, tfidf, k=1000):
    # Feature selection using Chi-Squared statistical test
    selector = SelectKBest(chi2, k=k)
    
    # Fit and transform
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    # Get feature names and scores
    feature_names = tfidf.get_feature_names_out()
    scores = selector.scores_
    top_indices = selector.get_support(indices=True)
    
    # Store top features in a dataframe
    top_features = pd.DataFrame({
        'Feature': feature_names[top_indices],
        'Chi2_Score': scores[top_indices]
    }).sort_values('Chi2_Score', ascending=False).head(20)
    
    return X_train_selected, X_test_selected, selector, top_features


def apply_dimensionality_reduction(X_train, X_test, n_components=100):
    # Dimensionality Reduction using Truncated SVD
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    
    # Fit and transform
    X_train_svd = svd.fit_transform(X_train)
    X_test_svd = svd.transform(X_test)
    
    explained_variance = svd.explained_variance_ratio_.sum()
    return X_train_svd, X_test_svd, svd, explained_variance
