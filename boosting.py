import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


if __name__ == '__main__':
    dt_heart = pd.read_csv('./data/heart.csv')
    print(dt_heart['target'].describe())

    # Features and target:
    X = dt_heart.drop(['target'], axis=1)  # Features
    y = dt_heart['target']  # Target data

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)

    # Definimos el clasificador:
    boost = GradientBoostingClassifier(n_estimators=50).fit(X_train, y_train)
    boost_predictions = boost.predict(X_test)

    print('=' * 30)
    print(f'Boosting accurary score: {accuracy_score(boost_predictions, y_test)}')
