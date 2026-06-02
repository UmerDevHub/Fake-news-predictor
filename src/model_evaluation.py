from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def evaluate_model(model, X_test, y_test, model_name):
    # Predict the response for test dataset
    y_pred = model.predict(X_test)
    
    # Model Accuracy, how often is the classifier correct?
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    return {
        'Model': model_name,
        'Accuracy': round(accuracy, 4),
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1-Score': round(f1, 4)
    }


def plot_confusion_matrix(model, X_test, y_test, model_name):
    # To make predictions on our test data
    y_pred = model.predict(X_test)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plotting confusion matrix using seaborn heatmap
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    ax.set_title(f'Confusion Matrix - {model_name}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Truth') # As per lab manual notation
    plt.tight_layout()
    return fig


def compare_models(models, X_test, y_test):
    results = []
    for name, model in models.items():
        result = evaluate_model(model, X_test, y_test, name)
        results.append(result)
    return pd.DataFrame(results)


def get_best_model(results_df):
    # finding the model with maximum accuracy score
    best_idx = results_df['Accuracy'].idxmax()
    return results_df.loc[best_idx]
