import streamlit as st
import os
import pickle
import requests
from bs4 import BeautifulSoup

from src.data_preprocessing import (
    load_data,
    preprocess_data,
    clean_text,
    split_data,
    detect_outliers,
    normalize_features,
)
from src.feature_engineering import (
    create_tfidf_features,
    create_text_features,
    select_features_chi2,
    apply_dimensionality_reduction,
)
from src.model_training import train_all_models
from src.model_evaluation import compare_models, plot_confusion_matrix, get_best_model
from src.eda import (
    class_distribution,
    text_length_histogram,
    word_count_boxplot,
    correlation_heatmap,
    scatter_plot,
    subject_distribution,
)

st.set_page_config(
    page_title='Fake News Detection',
    page_icon='📰',
    layout='wide',
    initial_sidebar_state='expanded',
)

st.markdown(
    """
    <style>
    .main { background-color: #f5f7fb; }
    .reportview-container .markdown-text-container { font-family: 'Segoe UI', sans-serif; }
    .stButton>button { background-color: #0f62fe; color: white; }
    .stButton>button:hover { background-color: #0353e9; }
    .card {
        border-radius: 20px;
        padding: 24px;
        background: white;
        box-shadow: 0 18px 45px rgba(15, 38, 70, 0.08);
        margin-bottom: 24px;
    }
    .metric-label { color: #5a6d8c; }
    </style>
    """,
    unsafe_allow_html=True,
)

if not os.path.exists('data/Fake.csv') or not os.path.exists('data/True.csv'):
    st.title('🛑 Dataset Missing')
    st.error('The news dataset files are missing from the `data/` folder.')
    st.write('Please download `Fake.csv` and `True.csv` from Kaggle and place them into `data/`.')
    st.stop()

@st.cache_data(show_spinner=False)
def load_and_process():
    data_path = 'data/saved_data.pkl'
    if os.path.exists(data_path):
        with open(data_path, 'rb') as f:
            return pickle.load(f)

    df_raw = load_data()
    df = preprocess_data(df_raw)
    df = create_text_features(df)
    df_clean, outliers, outlier_stats = detect_outliers(df)
    numeric_cols = ['text_length', 'word_count', 'avg_word_length', 'uppercase_count']
    df_normalized, scaler = normalize_features(df_clean.copy(), numeric_cols)

    with open(data_path, 'wb') as f:
        pickle.dump((df_clean, outlier_stats), f)

    return df_clean, outlier_stats

@st.cache_resource(show_spinner=False)
def build_models(df):
    model_path = 'data/saved_models.pkl'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)

    X_train, X_test, y_train, y_test = split_data(df['clean_text'], df['label'])
    X_train_tfidf, X_test_tfidf, tfidf = create_tfidf_features(X_train, X_test)
    _, _, _, top_features = select_features_chi2(
        X_train_tfidf, y_train, X_test_tfidf, tfidf, k=1000
    )
    _, _, svd, explained_var = apply_dimensionality_reduction(
        X_train_tfidf, X_test_tfidf, n_components=100
    )
    models = train_all_models(X_train_tfidf, y_train)

    artifacts = (models, tfidf, X_test_tfidf, y_test, top_features, explained_var)
    with open(model_path, 'wb') as f:
        pickle.dump(artifacts, f)

    return artifacts

with st.spinner('Loading dataset and preparing features...'):
    df, outlier_stats = load_and_process()

with st.spinner('Loading or training models...'):
    models, tfidf, X_test_tfidf, y_test, top_features, explained_var = build_models(df)

st.sidebar.title('Fake News Hub')
st.sidebar.markdown('A polished interface for news credibility prediction and analysis.')
st.sidebar.markdown('---')
status = 'Yes' if os.path.exists('data/saved_models.pkl') else 'No'
st.sidebar.markdown(
    f'**Dataset:** Fake vs Real News  \n**Records:** {len(df)}  \n**Cached models:** {status}'
)
st.sidebar.markdown('---')

page = st.sidebar.radio('Navigate', [
    '🔍 Prediction',
    '📊 EDA',
    '🔧 Preprocessing & Features',
    '📈 Model Performance',
    'ℹ️ About',
])

if page == '🔍 Prediction':
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header('News Authenticity Checker')
    st.write('Paste news content or enter a URL to validate whether the article is likely real or fake. Results display across three reliable models.')
    st.markdown('</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(['Paste Text', 'Analyze URL'])

    def display_prediction(column, name, model, features):
        prediction = model.predict(features)[0]
        confidence = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features)[0]
            confidence = f'{max(proba) * 100:.1f}%'
        label = '✅ REAL NEWS' if prediction == 0 else '❌ FAKE NEWS'
        if prediction == 0:
            column.success(f'**{name}**\n{label}')
        else:
            column.error(f'**{name}**\n{label}')
        if confidence is not None:
            column.caption(f'Confidence: {confidence}')

    with tab1:
        news_input = st.text_area('Paste the news article text here', height=250)
        if st.button('Analyze Text', key='text_btn'):
            if not news_input.strip():
                st.warning('Please enter article text before analyzing.')
            else:
                cleaned = clean_text(news_input)
                features = tfidf.transform([cleaned])
                cols = st.columns(len(models))
                for idx, (name, model) in enumerate(models.items()):
                    display_prediction(cols[idx], name, model, features)

    with tab2:
        url_input = st.text_input('Paste the news URL here')
        if st.button('Analyze URL', key='url_btn'):
            if not url_input.strip():
                st.warning('Please enter a URL before analyzing.')
            else:
                try:
                    response = requests.get(url_input, timeout=12)
                    soup = BeautifulSoup(response.text, 'html.parser')
                    paragraphs = soup.find_all('p')
                    article_text = ' '.join([p.get_text() for p in paragraphs])
                    if len(article_text) < 100:
                        st.warning('Unable to extract enough article text from the URL.')
                    else:
                        st.info('Extracted content from the URL successfully.')
                        st.text_area('Extracted preview', article_text[:1200], height=180, disabled=True)
                        cleaned = clean_text(article_text)
                        features = tfidf.transform([cleaned])
                        cols = st.columns(len(models))
                        for idx, (name, model) in enumerate(models.items()):
                            display_prediction(cols[idx], name, model, features)
                except Exception as e:
                    st.error(f'Unable to fetch the URL: {e}')

elif page == '📊 EDA':
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header('Exploratory Data Analysis')
    st.write('Explore class balance, subject distribution, and text statistics for fake vs real news.')
    st.markdown('</div>', unsafe_allow_html=True)

    top1, top2, top3 = st.columns(3)
    top1.metric('Total articles', len(df))
    top2.metric('Real articles', int(df['label'].value_counts().get(0, 0)))
    top3.metric('Fake articles', int(df['label'].value_counts().get(1, 0)))

    st.subheader('Dataset preview')
    st.dataframe(df.head(8), use_container_width=True)

    st.subheader('Class and subject distribution')
    dist_col1, dist_col2 = st.columns(2)
    with dist_col1:
        st.pyplot(class_distribution(df))
    with dist_col2:
        st.pyplot(subject_distribution(df))

    st.subheader('Text statistics')
    stats_col1, stats_col2 = st.columns(2)
    with stats_col1:
        st.pyplot(text_length_histogram(df))
    with stats_col2:
        st.pyplot(word_count_boxplot(df))

    more_col1, more_col2 = st.columns(2)
    with more_col1:
        st.pyplot(correlation_heatmap(df))
    with more_col2:
        st.pyplot(scatter_plot(df))

elif page == '🔧 Preprocessing & Features':
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header('Preprocessing & Feature Engineering')
    st.write('Review the transformations that prepare news text for modeling and the most powerful extracted features.')
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        '- **Duplicate removal**: drops repeated rows\n'
        '- **Missing value handling**: removes null records\n'
        '- **Text cleaning**: lowercasing, URL removal, punctuation stripping, stopword filtering, and lemmatization\n'
        '- **Feature construction**: text length, word count, avg word length, uppercase count, exclamation flags\n'
        '- **Outlier removal**: IQR filtering on article length\n'
        '- **TF-IDF + χ² selection**: identifies the most discriminative words for classification\n'
    )

    metrics_col1, metrics_col2 = st.columns(2)
    metrics_col1.metric('Outliers removed', outlier_stats['total_outliers'])
    metrics_col1.metric('Outlier percentage', f"{outlier_stats['outlier_percentage']}%")
    metrics_col2.metric('IQR value', f"{outlier_stats['IQR']}")
    metrics_col2.metric('Total analyzed', outlier_stats['total_records'])

    st.subheader('Top selected TF-IDF features')
    st.dataframe(top_features.reset_index(drop=True), use_container_width=True)

    st.subheader('Dimensionality reduction')
    st.metric('Explained variance', f'{explained_var * 100:.2f}%')
    st.info('Truncated SVD compresses the high-dimensional TF-IDF matrix into a smaller semantic feature space.')

elif page == '📈 Model Performance':
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header('Model Evaluation')
    st.write('Compare evaluation metrics and inspect confusion matrices for the trained classifiers.')
    st.markdown('</div>', unsafe_allow_html=True)

    results = compare_models(models, X_test_tfidf, y_test)
    st.dataframe(results, use_container_width=True)

    best = get_best_model(results)
    st.success(f'🏆 Best model: **{best["Model"]}** with accuracy **{best["Accuracy"]}**')

    st.subheader('Confusion matrices')
    matrix_cols = st.columns(3)
    for idx, (name, model) in enumerate(models.items()):
        with matrix_cols[idx]:
            st.pyplot(plot_confusion_matrix(model, X_test_tfidf, y_test, name))

    st.markdown(
        'Logistic Regression usually performs best on TF-IDF text data. Naive Bayes is fast and interprets word distributions naturally, while linear SVM provides strong boundaries for high-dimensional feature spaces.'
    )

else:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header('About This App')
    st.write('This Fake News Detection app combines NLP preprocessing, TF-IDF text representation, feature selection, and machine learning classifiers to score news articles for credibility.')
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        '- Built with Streamlit for an interactive dashboard experience.\n'
        '- Uses TF-IDF and χ² to highlight the strongest text features.\n'
        '- Trains Logistic Regression, Naive Bayes, and SVM models.\n'
        '- Caches preprocessing and model artifacts for fast reloads.\n'
    )

    st.subheader('How to run locally')
    st.code('streamlit run app.py')
