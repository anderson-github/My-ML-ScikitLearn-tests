import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, KFold


if __name__ == '__main__':

    dataset = pd.read_csv('./data/felicidad.csv')
    print(dataset)

    # Definición de Features:
    # Drop country by categorical data and score because is the target data.
    X = dataset.drop(['country', 'score'], axis=1)  # Features
    y = dataset['score']  # Target data

    # Model definition:
    model = DecisionTreeRegressor()
    score = cross_val_score(model, X, y, cv=3, scoring='neg_mean_squared_error')  # cv: number of Cross-validations.
    
    # Looking performance:
    print('=' * 60)
    print(f'Score: {score}')

    # Undestanding the "score" result:
    print('=' * 60)
    print(f'Absolute mean score value: {np.abs(np.mean(score))}')

    # Implenting K-Fold cross-validation:
    print('=' * 60)
    print('USING K-FOLD CV:')
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    for train, test in kf.split(dataset):
        print('Train:')
        print(train)
        print('Test:')
        print(test)
        # Acá debo seleccionar un modelo de regresion o de entrenamiento y 
        # pasar los datos de train y test a fin de analizar su comportamiento
        # luego del número de K-folds usados.

