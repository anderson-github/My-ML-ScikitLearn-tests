import pandas as pd

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor


if __name__ == '__main__':
    
    # Load data:
    dataset = pd.read_csv('./data/felicidad.csv')
    # print(dataset)

    # Defining features y target data:
    X = dataset.drop(['country', 'rank', 'score'], axis=1)  # Features
    y = dataset['score']  # Target data

    # Definimos el regresor (estimador) a utilizar:
    # NOTA: RandomForestRegressor en en realidad un meta-estimador.
    reg = RandomForestRegressor()

    # Definimos el set de parámetros para probar:
    parameters = {
        'n_estimators': range(4, 16),
        'criterion': ['mse', 'mae'],
        'max_depth': range(2, 11),
    }

    # Optimización del estimador:
    rand_est = RandomizedSearchCV(reg, parameters, n_iter=10, cv=3, scoring='neg_mean_absolute_error').fit(X, y)

    print('=' * 60)
    print('Best estimator configuration:')
    print(rand_est.best_estimator_)
    print('Best parameters:')
    print(rand_est.best_params_)

    # Making predictions (using best estimator configuration):
    # NOTA: La predicción arrojada es para el "score" del país que corresponde al número de la fila usada.
    print('=' * 60)
    row = 2
    print(f'Predicting "score" using data from row X[{row}]')
    print(rand_est.predict(X.loc[[row]]))
