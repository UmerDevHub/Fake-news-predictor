#!/usr/bin/env python3
"""evaluate.py

Simple runner to train models and print evaluation metrics (accuracy, precision, recall, F1).

Usage:
    python evaluate.py

It expects `data/Fake.csv` and `data/True.csv` to be present in the `data/` folder.
If the CSVs are missing, the script prints instructions and exits.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

def main():
    try:
        from data_preprocessing import load_data, preprocess_data, split_data
        from feature_engineering import create_text_features, create_tfidf_features
        from model_training import train_all_models
        from model_evaluation import compare_models
    except Exception as e:
        print("Error importing project modules:", e)
        return 2

    data_dir = ROOT / "data"
    fake_csv = data_dir / "Fake.csv"
    true_csv = data_dir / "True.csv"
    if not fake_csv.exists() or not true_csv.exists():
        print("Data files missing.")
        print("Place the original CSVs in the 'data/' folder:")
        print("  - data/Fake.csv")
        print("  - data/True.csv")
        print("After placing them, re-run: python evaluate.py")
        return 1

    try:
        df = load_data()
    except FileNotFoundError as e:
        print("Failed to load data:", e)
        return 1

    df = preprocess_data(df)
    df = create_text_features(df)

    X = df['clean_text']
    y = df['label']

    X_train, X_test, y_train, y_test = split_data(X, y)

    X_train_tfidf, X_test_tfidf, _ = create_tfidf_features(X_train, X_test)

    models = train_all_models(X_train_tfidf, y_train)
    results = compare_models(models, X_test_tfidf, y_test)

    print('\nModel evaluation results:\n')
    print(results.to_string(index=False))

    return 0

if __name__ == '__main__':
    raise SystemExit(main())
