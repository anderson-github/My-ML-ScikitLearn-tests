import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier

if __name__ == "__main__":
    
    dt_heart = pd.read_csv('./data/heart.csv')
    # print(dt_heart['target'].describe())

    # Features and target:
    X = dt_heart.drop(['target'], axis=1)  # Features
    y = dt_heart['target']  # Target data

    clf1 = LogisticRegression(penalty='l2', random_state=None, max_iter=2000)
    clf2 = RandomForestClassifier(n_estimators=50, random_state=None)
    clf3 = GaussianNB()

    # Ensamble models: Voting Classifier
    eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')

    for clf, label in zip([clf1, clf2, clf3, eclf], ['Logistic Regression', 'Random Forest', 'naive Bayes', 'Ensemble']):
        scores = cross_val_score(clf, X, y, scoring='accuracy', cv=5)
        print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
