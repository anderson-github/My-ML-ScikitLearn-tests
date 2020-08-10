import pandas as pd

from sklearn.linear_model import RANSACRegressor, HuberRegressor
from sklearn.svm import SVR

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


if __name__ == '__main__':
    dataset = pd.read_csv('./data/felicidad_corrupt.csv')
    print(dataset)

    X = dataset.drop(['country', 'score'], axis=1)  # Raw data
    y = dataset[['score']]  # Target data

    # Split raw data in training and test datasets:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Manera más profesional de guardar los estimadores:
    estimators = {
        'SVR': SVR(gamma='auto', C=1.0, epsilon=0.1),
        'RANSAC': RANSACRegressor(),
        'HUBER': HuberRegressor(epsilon=1.35),
    }

    for name, estimator in estimators.items():
        estimator.fit(X_train, y_train)
        predictions = estimator.predict(X_test)

        print('=' * 20)
        print(name)
        print(f'MSE = {mean_squared_error(y_test, predictions)}')
