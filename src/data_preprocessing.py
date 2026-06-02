import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load NLTK packages
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)


def load_data():
    # Let's first load the required dataset using pandas' read CSV function
    fake = pd.read_csv('data/Fake.csv')
    true = pd.read_csv('data/True.csv')
    
    # Add labels for classification (1 for Fake, 0 for Real)
    fake['label'] = 1
    true['label'] = 0
    
    # Combine datasets
    df = pd.concat([fake, true], axis=0)
    df = df.reset_index(drop=True)
    return df


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]
    return ' '.join(words)


def detect_outliers(df):
    df = df.copy()
    text_lengths = df['clean_text'].apply(len)
    
    # Calculating IQR for outliers
    Q1 = text_lengths.quantile(0.25)
    Q3 = text_lengths.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outlier_mask = (text_lengths < lower_bound) | (text_lengths > upper_bound)
    outliers = df[outlier_mask]
    df_clean = df[~outlier_mask]
    
    stats = {
        'Q1': round(Q1, 2), 'Q3': round(Q3, 2), 'IQR': round(IQR, 2),
        'lower_bound': round(lower_bound, 2), 'upper_bound': round(upper_bound, 2),
        'total_outliers': len(outliers), 'total_records': len(df),
        'outlier_percentage': round(len(outliers) / len(df) * 100, 2)
    }
    return df_clean, outliers, stats


def encode_subject(df):
    # creating object
    le = LabelEncoder()
    # creating extra column in the dataset – label encoded
    df['subject_encoded'] = le.fit_transform(df['subject'])
    return df, le


def normalize_features(df, feature_cols):
    # machine learning algorithms work better with normalized numbers
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df, scaler


def preprocess_data(df):
    # Step 1: Remove duplicates
    df = df.drop_duplicates()
    
    # Step 2: Handle missing values by dropping
    df = df.dropna()
    
    # Step 3: Combine title and text
    df['text'] = df['title'] + ' ' + df['text']
    
    # Step 4: Clean text
    df['clean_text'] = df['text'].apply(clean_text)
    
    # Step 5: Encode subject column
    df, _ = encode_subject(df)
    return df


def split_data(X, y):
    # Split dataset into training set and test set
    # 80% training and 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
