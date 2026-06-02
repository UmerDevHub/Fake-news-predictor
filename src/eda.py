import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def class_distribution(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df['label'].value_counts()
    ax.bar(['Real News', 'Fake News'], [counts[0], counts[1]], color=['green', 'red'])
    ax.set_title('Class Distribution')
    ax.set_ylabel('Count')
    plt.tight_layout()
    return fig


def text_length_histogram(df):
    fig, ax = plt.subplots(figsize=(8, 4))
    df['text_length'] = df['clean_text'].apply(len)
    ax.hist(df[df['label'] == 0]['text_length'], bins=50, alpha=0.5, label='Real', color='green')
    ax.hist(df[df['label'] == 1]['text_length'], bins=50, alpha=0.5, label='Fake', color='red')
    ax.set_title('Text Length Distribution')
    ax.set_xlabel('Text Length')
    ax.set_ylabel('Frequency')
    ax.legend()
    plt.tight_layout()
    return fig


def word_count_boxplot(df):
    fig, ax = plt.subplots(figsize=(8, 4))
    df['word_count'] = df['clean_text'].apply(lambda x: len(str(x).split()))
    data = [df[df['label'] == 0]['word_count'], df[df['label'] == 1]['word_count']]
    ax.boxplot(data)
    ax.set_xticklabels(['Real News', 'Fake News'])
    ax.set_title('Word Count Distribution')
    ax.set_ylabel('Word Count')
    plt.tight_layout()
    return fig


def correlation_heatmap(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    df['text_length'] = df['clean_text'].apply(len)
    df['word_count'] = df['clean_text'].apply(lambda x: len(str(x).split()))
    df['avg_word_length'] = df['clean_text'].apply(
        lambda x: np.mean([len(w) for w in str(x).split()]) if len(str(x).split()) > 0 else 0
    )
    numeric_cols = df[['text_length', 'word_count', 'avg_word_length', 'label']]
    sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Heatmap')
    plt.tight_layout()
    return fig


def scatter_plot(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    df['text_length'] = df['clean_text'].apply(len)
    df['word_count'] = df['clean_text'].apply(lambda x: len(str(x).split()))
    real = df[df['label'] == 0]
    fake = df[df['label'] == 1]
    ax.scatter(real['text_length'], real['word_count'], c='green', alpha=0.3, s=10, label='Real')
    ax.scatter(fake['text_length'], fake['word_count'], c='red', alpha=0.3, s=10, label='Fake')
    ax.set_xlabel('Text Length')
    ax.set_ylabel('Word Count')
    ax.set_title('Text Length vs Word Count')
    ax.legend()
    plt.tight_layout()
    return fig


def subject_distribution(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    df.groupby(['subject', 'label']).size().unstack().plot(kind='bar', ax=ax, color=['green', 'red'])
    ax.set_title('Subject Distribution by Label')
    ax.set_xlabel('Subject')
    ax.set_ylabel('Count')
    ax.legend(['Real', 'Fake'])
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig
