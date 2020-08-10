import pandas as pd
import sklearn

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


if __name__ == '__main__':
    dataset = pd.read_csv('./data/felicidad.csv')
    print(dataset.describe())

    # Selección manual de features:
    X = dataset[['gdp', 'family', 'lifexp', 'freedom', 'corruption', 'generosity', 'dystopia']]
    y = dataset[['score']]  # Target data.

    print(f'Features data shape: {X.shape}')
    print(f'Target data shape: {y.shape}')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # Definimos y probamos distintos regresores:
    # 1) Regresion Lineal:
    modelLinear = LinearRegression().fit(X_train, y_train)
    y_predic_linear = modelLinear.predict(X_test)

    # 2) Regresion Lasso:
    modelLasso = Lasso(alpha=0.02).fit(X_train, y_train)
    y_predic_lasso = modelLasso.predict(X_test)

    # 3) Regresion Ridge:
    modelRidge = Ridge(alpha=1).fit(X_train, y_train)
    y_predic_ridge = modelRidge.predict(X_test)

    # Calculos de metrica para comparar modelos:
    linear_loss = mean_squared_error(y_test, y_predic_linear)
    print('Linear loss: ', linear_loss)

    lasso_loss = mean_squared_error(y_test, y_predic_lasso)
    print(f'Lasso loss: {lasso_loss}')

    ridge_loss = mean_squared_error(y_test, y_predic_ridge)
    print(f'Ridge loss: {ridge_loss}')

    print('=' * 32)
    print('Coefs LASSO: ')
    print(modelLasso.coef_)

    print('=' * 32)
    print('Coefs RIDGE: ')
    print(modelRidge.coef_)
