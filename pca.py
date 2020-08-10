import pandas as pd
import sklearn
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA

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

    # n_components = min(n_samples, n_features)
    pca = PCA(n_components=3)
    pca.fit(x_train)

    ipca = IncrementalPCA(n_components=3, batch_size=10)
    ipca.fit(x_train)

    # Visualización
    plt.plot(range(len(pca.explained_variance_)), pca.explained_variance_ratio_)
    plt.show()

    # Configuramos logistic regresion para entrenar con los mejores 
    # features encontrados con PCA
    logistic = LogisticRegression(solver='lbfgs')
    
    # En este paso aplicamos en PCA que configuramos sobre los datos de 
    # entrenamiento y test.
    train_data = pca.transform(x_train)
    test_data = pca.transform(x_test)

    logistic.fit(train_data, y_train)

    # Medimos la efectividad del modelo. En este caso, calculando accuracy:
    print('SCORE PCA: ', logistic.score(test_data, y_test))

    # Ahora miramos que pasa cuando usamos IPCA
    train_data = ipca.transform(x_train)
    test_data = ipca.transform(x_test)

    logistic.fit(train_data, y_train)

    print('SCORE IPCA: ', logistic.score(test_data, y_test))
