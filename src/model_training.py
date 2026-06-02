from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


def train_logistic_regression(X_train, y_train):
    # Create Logistic Regression classifer object
    model = LogisticRegression(max_iter=1000)
    
    # Train Logistic Regression Classifer
    model.fit(X_train, y_train)
    return model


def train_naive_bayes(X_train, y_train):
    # Create Naive Bayes classifer object
    model = MultinomialNB()
    
    # Train Naive Bayes Classifer
    model.fit(X_train, y_train)
    return model


def train_svm(X_train, y_train):
    # Create SVM classifer object
    model = LinearSVC(max_iter=1000)
    
    # Train SVM Classifer
    model.fit(X_train, y_train)
    return model


def train_all_models(X_train, y_train):
    # Initialize dictionary to store models
    models = {}
    
    # Train and store all models
    models['Logistic Regression'] = train_logistic_regression(X_train, y_train)
    models['Naive Bayes'] = train_naive_bayes(X_train, y_train)
    models['SVM'] = train_svm(X_train, y_train)
    
    return models
