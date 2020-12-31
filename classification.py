# TODO:
#   1.create classifier
#   2.grid_search to optimize
#   3.show scores
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, fbeta_score, accuracy_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB


def vectorize(data):
    count_vector = CountVectorizer()
    emb = count_vector.fit_transform(data)
    return emb


def tfidf(data):
    tfidf_vectorizer = TfidfVectorizer()
    train = tfidf_vectorizer.fit_transform(data)
    return train


def classification(X, y):
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    result_list = []
    # Adaboost

    # clf_Ada = AdaBoostClassifier(random_state=0)

    # 决策树

    clf_Tree = DecisionTreeClassifier(random_state=0)
    parameter_tree = {'max_depth': list(range(1, 10))}
    result_list.append(grid_search_cv(X, y, clf_Tree, parameter_tree))

    # KNN

    clf_KNN = KNeighborsClassifier()
    parameter_KNN = {'n_neighbors': list(range(1, 10)),
                     'algorithm': ['brute', 'kd_tree', 'ball_tree', 'auto']}
    result_list.append(grid_search_cv(X, y, clf_KNN, parameter_KNN))

    # SVM

    clf_svm = SVC(random_state=0)
    parameter_svm = {'C': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]}
    # 'kernel': ['rbf','linear','poly','sigmoid','precomputed']}
    result_list.append(grid_search_cv(X, y, clf_svm, parameter_svm))

    # Logistic

    clf_log = LogisticRegression(random_state=0)
    parameter_log = {'C': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]}
    result_list.append(grid_search_cv(X, y, clf_log, parameter_log))

    # RF

    clf_forest = RandomForestClassifier(random_state=0)
    parameter_forest = {'n_estimators': list(range(5, 15)),
                        'max_depth': list(range(1, 10))}
    result_list.append(grid_search_cv(X, y, clf_forest, parameter_forest))

    # MultinomialNB

    clf_NB = MultinomialNB()
    parameter_NB = {'alpha': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
    result_list.append(grid_search_cv(X, y, clf_NB, parameter_NB))
    return result_list


def grid_search_cv(X_train, y_train, classifier, params_dict):
    scorer = make_scorer(accuracy_score)
    grid = GridSearchCV(classifier, params_dict, scorer, n_jobs=-1, cv=10)
    grid = grid.fit(X_train, y_train)
    print("best score:", grid.best_score_)
    print("best_params:", grid.best_params_)
    return [str(classifier).split('(')[0], grid.best_score_, grid.best_params_]
    # display(pd.DataFrame(grid.cv_results_).T)


if __name__ == '__main__':
    dataset = pd.read_csv("data/processed_all_data.csv").dropna()
    X = dataset['text']
    y = dataset['target']
    X_vec = vectorize(X)
    X_tfidf = tfidf(X)

    res_list = classification(X_vec, y)
    res_array = np.array(res_list)
    res_pd = pd.DataFrame(res_array, columns=['clf', 'best_score', 'best_params'])
    res_pd.to_csv('data/vec_result.csv')

    res_list = classification(X_tfidf, y)
    res_array = np.array(res_list)
    res_pd = pd.DataFrame(res_array, columns=['clf', 'best_score', 'best_params'])
    res_pd.to_csv('data/tfidf_result.csv')
