"""Logistic regression test accuracy on 1D CCM ratio features (gamma vs neutron)."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def get_accuracy(gamma_ratio, neutron_ratio):
    X = np.concatenate([gamma_ratio, neutron_ratio]).reshape(-1, 1)
    y = np.concatenate([np.zeros(len(gamma_ratio)), np.ones(len(neutron_ratio))])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    return accuracy_score(y_test, clf.predict(X_test))
