import pandas as pd

from sklearn.cluster import MiniBatchKMeans


if __name__ == '__main__':

    dataset = pd.read_csv('./data/candy.csv')
    print(dataset.head(10))

    # No hacemos división de datos, porque estamos en aprendizaje NO SUPERVISADO.
    # En este caso pasamos todos los datos al algoritmo, menos las variables categóricas 
    # ya que no pueden ser usadas por este algorimo para el proceso de clustering.
    X = dataset.drop('competitorname', axis=1)

    # Entrenamos el modelo:
    print('=' * 60)
    kmeans = MiniBatchKMeans(n_clusters=4, batch_size=8).fit(X)
    print(f'Total centros: {len(kmeans.cluster_centers_)}')

    # Miramos las etiquetas que asinó el modelo:
    print('=' * 60)
    print(f'Labels: {kmeans.predict(X)}')

    # Integramos las etiquetas que nos dió el modelo en el dataset
    # para mejor visualización:
    print('=' * 60)
    dataset['group'] = kmeans.predict(X)
    print(dataset)
