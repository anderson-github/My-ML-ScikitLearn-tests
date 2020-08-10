import pandas as pd
import sklearn
import matplotlib.pyplot as plt

from sklearn.decomposition import KernelPCA

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    heart_df = pd.read_csv('./data/heart.csv')

    print(heart_df.head(5))

    features_df = heart_df.drop(['target'], axis=1)
    target_df = heart_df['target']

    # Para PCA siempre necesitamos normalizar los datos.
    features_df = StandardScaler().fit_transform(features_df)

    # Dividimos el conjunto de entrenamiento
    x_train, x_test, y_train, y_test = train_test_split(features_df, target_df, test_size=0.30, random_state=42)

    print(f'x_train shape = {x_train.shape}')
    print(f'y_train shape = {y_train.shape}')

    kpca = KernelPCA(n_components=4, kernel='poly')
    kpca.fit(x_train)

    # Aplicamos KPCA sobre los datos de entrenamiento y prueba:
    train_data = kpca.transform(x_train)
    test_data = kpca.transform(x_test)

    # Aplicamos regresion logistica sobre los datos de train and test:
    logistic = LogisticRegression(solver='lbfgs')

    # Entrenamos el modelo:
    logistic.fit(train_data, y_train)

    print('SCORE KPCA: ', logistic.score(test_data, y_test))
