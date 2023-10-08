from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import GridSearchCV

def fit_predict_eval(model, feature_train, features_test, target_train, target_test):
    model.fit(feature_train, target_train)
    predictions = model.predict(features_test)
    score = accuracy_score(target_test, predictions)
    print(f'Model: {model}\nAccuracy: {score:.4f}\n')


if __name__=='__main__':
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    n = x_train.shape[0]
    m = x_train.shape[1] * x_train.shape[2]
    classes = np.unique(y_train)
    x_train = x_train.reshape((n, m))
    # Stage 1/5
    # print(f"Classes: {classes}")
    # print(f"Features' shape: {x_train.shape}")
    # print(f"Targets' shape: {y_train.shape}")
    # print(f"min: {x_train.min()}, max: {x_train.max()}")

    # Stage 2/5
    n_samples = 6000
    X_train, X_test, Y_train, Y_test = train_test_split(x_train[:n_samples], y_train[:n_samples], test_size=0.3, random_state=40)
    #
    # class_distribution = pd.Series(Y_train).value_counts(normalize=True)
    # print(f"x_train shape: {X_train.shape}")
    # print(f"x_test shape: {X_test.shape}")
    # print(f"y_train shape: {Y_train.shape}")
    # print(f"y_test shape: {Y_test.shape}")
    # print("Proportion of samples per class in train set:")
    # print(class_distribution.apply(lambda x: round(x,2)))

    # Stage 3/5
    # models = [
    #     KNeighborsClassifier(),
    #     DecisionTreeClassifier(random_state=40),
    #     LogisticRegression(),
    #     RandomForestClassifier(random_state=40)
    # ]
    # for model in models:
    #     fit_predict_eval(
    #         model=model,
    #         feature_train=X_train,
    #         features_test=X_test,
    #         target_train=Y_train,
    #         target_test=Y_test
    #     )
    #
    # print(f"The answer to the question: RandomForestClassifier - 0.939")

    # Stage 4/5
    # transformer = Normalizer().fit(X_train)
    # x_train_norm = transformer.transform(X_train)
    # x_test_norm = transformer.transform(X_test)
    # models = [
    #     KNeighborsClassifier(),
    #     DecisionTreeClassifier(random_state=40),
    #     LogisticRegression(),
    #     RandomForestClassifier(random_state=40)
    # ]
    # for model in models:
    #     fit_predict_eval(
    #         model=model,
    #         feature_train=x_train_norm,
    #         features_test=x_test_norm,
    #         target_train=Y_train,
    #         target_test=Y_test
    #     )
    # print("The answer to the 1st question: yes")
    #
    # print(f"The answer to the 2nd question: KNeighborsClassifier-0.953, RandomForestClassifier-0.937")

    # Stage 5/5
    transformer = Normalizer().fit(X_train)
    x_train_norm = transformer.transform(X_train)
    x_test_norm = transformer.transform(X_test)

    knn_params = {
        'n_neighbors': [3, 4],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'brute']
    }
    knn_grid = GridSearchCV(
        estimator=KNeighborsClassifier(),
        param_grid=knn_params,
        scoring='accuracy',
        n_jobs=-1,
    )
    knn_grid.fit(x_train_norm, Y_train)
    knn_acc = knn_grid.best_estimator_.score(x_test_norm, Y_test)

    print("K-nearest neighbours algorithm")
    print(f"best estimator: {knn_grid.best_estimator_}")
    print(f"accuracy: {knn_acc}")
    print()
    forest_params = {
        'n_estimators': [300, 500],
        'max_features': ['sqrt', 'log2'],
        'class_weight': ['balanced', 'balanced_subsample']
    }
    forest_grid = GridSearchCV(
        estimator=RandomForestClassifier(random_state=40),
        param_grid=forest_params,
        scoring='accuracy',
        n_jobs=-1,
    )
    forest_grid.fit(x_train_norm, Y_train)
    forest_acc = forest_grid.score(x_test_norm, Y_test)

    print("Random forest algorithm")
    print(f"best estimator: {forest_grid.best_estimator_}")
    print(f"accuracy: {forest_acc}")

