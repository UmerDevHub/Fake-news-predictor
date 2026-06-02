import pickle
import warnings
warnings.filterwarnings('ignore')

from src.data_preprocessing import load_data, preprocess_data, detect_outliers, normalize_features, split_data
from src.feature_engineering import create_text_features, create_tfidf_features, select_features_chi2, apply_dimensionality_reduction
from src.model_training import train_all_models

print("Step 1/3: Loading and cleaning 44,000 articles (This takes a moment)...")
df_raw = load_data()
df = preprocess_data(df_raw)
df = create_text_features(df)
df_clean, outliers, outlier_stats = detect_outliers(df)
numeric_cols = ['text_length', 'word_count', 'avg_word_length', 'uppercase_count']
df_normalized, scaler = normalize_features(df_clean.copy(), numeric_cols)

print("Step 2/3: Saving preprocessed data to hard disk...")
with open('data/saved_data.pkl', 'wb') as f:
    pickle.dump((df_clean, outlier_stats), f)

print("Step 3/3: Training Machine Learning models...")
X_train, X_test, y_train, y_test = split_data(df_clean['clean_text'], df_clean['label'])
X_train_tfidf, X_test_tfidf, tfidf = create_tfidf_features(X_train, X_test)

X_train_sel, X_test_sel, selector, top_features = select_features_chi2(X_train_tfidf, y_train, X_test_tfidf, tfidf, k=1000)
X_train_svd, X_test_svd, svd, explained_var = apply_dimensionality_reduction(X_train_tfidf, X_test_tfidf, n_components=100)

models = train_all_models(X_train_tfidf, y_train)

artifacts = (models, tfidf, X_test_tfidf, y_test, top_features, explained_var)
with open('data/saved_models.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

print("\n✅ SUCCESS: All data and models are permanently saved to data/ directory!")
print("Your Streamlit app will now load instantly every time.")
