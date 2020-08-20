import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import BaggingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


if __name__ == '__main__':

    dt_heart = pd.read_csv('./data/heart.csv')
    # print(dt_heart['target'].describe())

    # Features and target:
    X = dt_heart.drop(['target'], axis=1)  # Features
    y = dt_heart['target']  # Target data

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)

    # Clasificación usando método de primeros vecinos (KNeighbors):
    knn_class = KNeighborsClassifier().fit(X_train, y_train)
    knn_predictions = knn_class.predict(X_test)
    
    print('=' * 60)
    print('KNeighborsClassifier without Bagging')
    print(f'Classifier accuracy score: {accuracy_score(knn_predictions, y_test)}')

    # Clasificación usando método por ensamble, BAGGING en este caso:
    # Usando KNeighborsClassifier:
    bag_class = BaggingClassifier(base_estimator=KNeighborsClassifier(), n_estimators=50).fit(X_train, y_train)
    bag_predictions = bag_class.predict(X_test)

    print('=' * 60)
    print('KNeighborsClassifier with Bagging')
    print(f'KNN Bagging accuracy score: {accuracy_score(bag_predictions, y_test)}')

    # Usando Suport Vector Classification (SVC):
    bag_class = BaggingClassifier(base_estimator=SVC(), n_estimators=50).fit(X_train, y_train)
    bag_predictions = bag_class.predict(X_test)

    print('=' * 60)
    print('SVC with Bagging')
    print(f'SVC Classifier accuracy score: {accuracy_score(bag_predictions, y_test)}')

    # Usando Stochastic Gradient Descent (SGD) Classifier:
    clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=1000)
    bag_class = BaggingClassifier(base_estimator=clf, n_estimators=50).fit(X_train, y_train)
    bag_predictions = bag_class.predict(X_test)

    print('=' * 60)
    print('SGD with Bagging')
    print(f'SDG Classifier accuracy score: {accuracy_score(bag_predictions, y_test)}')
