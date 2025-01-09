import logging
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


logging.basicConfig(level=logging.INFO)


def train_test(hyperparams, X, y):

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    model = SVC(C=hyperparams['C'], kernel=hyperparams['kernel'], gamma=hyperparams.get('gamma', 'scale'))

    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    logging.info(f"Validation Accuracy: {val_accuracy}")

    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    logging.info(f"Test Accuracy: {test_accuracy}")

    return {
        "validation_accuracy": val_accuracy,
        "test_accuracy": test_accuracy
    }
