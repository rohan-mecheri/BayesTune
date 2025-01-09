import logging
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.worker import train_test

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    data = load_digits()
    X, y = data.data, data.target

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    hyperparams = {
        "C": 1.0,
        "kernel": "rbf",
        "gamma": 0.1
    }

    results = train_test(hyperparams, X, y)
    logging.info(f"Validation Accuracy: {results['validation_accuracy']}")
    logging.info(f"Test Accuracy: {results['test_accuracy']}")

    model = SVC(C=hyperparams['C'], kernel=hyperparams['kernel'], gamma=hyperparams['gamma'])
    scores = cross_val_score(model, X, y, cv=5)
    logging.info(f"Cross-validation accuracy: {scores.mean()}")
