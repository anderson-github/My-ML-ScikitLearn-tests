import pandas as pd

from sklearn.cluster import MeanShift


if __name__ == '__main__':

    dataset = pd.read_csv('./data/candy.csv')
    print(dataset)

    # Elinamos la columna con datos categóricos ya que no puede ser 
    # ser usadas por el algorimo para el proceso de clustering.
    X = dataset.drop('competitorname', axis=1)

    # Instanciamos el modelo con la configuración deseada:
    meanshift = MeanShift().fit(X)
    print('=' * 60)
    print(f'Labels: {max(meanshift.labels_)}')

    # Ubicación de los centroides asignados a los datos:
    print('=' * 60)
    print('Centroids:')
    print(meanshift.cluster_centers_)

    # Integramos las etiquetas encontradas al dataset:
    print('=' * 60)
    dataset['meanshift'] = meanshift.labels_
    print(dataset)
