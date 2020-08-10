import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


if __name__ == '__main__':

    dt_heart = pd.read_csv('./data/heart.csv')
    print(dt_heart['target'].describe())

    # Features and target:
    X = dt_heart.drop(['target'], axis=1)  # Features
    y = dt_heart['target']  # Target data

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)

    # Clasificación usando método de primeros vecinos (KNeighbors):
    knn_class = KNeighborsClassifier().fit(X_train, y_train)
    knn_predictions = knn_class.predict(X_test)
    
    print('=' * 30)
    print(f'KNN accuracy score: {accuracy_score(knn_predictions, y_test)}')

    # Clasificación usando método por ensamble (bagging, en este caso):
    bag_class = BaggingClassifier(base_estimator=KNeighborsClassifier(), n_estimators=50).fit(X_train, y_train)
    bag_predictions = bag_class.predict(X_test)

    print('=' * 30)
    print(f'Bagging accuracy score: {accuracy_score(bag_predictions, y_test)}')
