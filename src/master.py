import yaml
import logging
from multiprocessing import Pool
import csv
import json
from skopt.space import Real, Integer, Categorical
from skopt import gp_minimize
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.worker import train_test
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)

RESULTS_CSV = "results/all_results.csv"

BEST_PARAMS_JSON = "results/best_hyperparams.json"


def load_config(file_path):
    try:
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise
    except yaml.YAMLError as exc:
        logging.error(f"Cannot parse file: {exc}")
        raise


def searchspace(hyperparams_config):
    search_space = []
    for param in hyperparams_config['hyperparameters']:
        if param['type'] == 'Real':
            search_space.append(Real(param['min'], param['max'], name=param['name']))
        elif param['type'] == 'Integer':
            search_space.append(Integer(param['min'], param['max'], name=param['name']))
        elif param['type'] == 'Categorical':
            search_space.append(Categorical(param['categories'], name=param['name']))
        else:
            logging.error(f"Unknown parameter type: {param['type']}")
            raise ValueError(f"Unknown parameter type: {param['type']}")
    return search_space


def log_results(params, accuracy):
    file_exists = os.path.isfile(RESULTS_CSV)
    with open(RESULTS_CSV, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow([dim.name for dim in search_space] + ["Validation_Accuracy"])
        writer.writerow(params + [accuracy])


def save_best_hyperparameters(params, accuracy):
    best_result = {
        "best_hyperparameters": {
            dim.name: (int(value) if isinstance(value, (int, float)) and isinstance(dim, Integer) 
                       else float(value) if isinstance(value, (int, float)) 
                       else str(value))
            for dim, value in zip(search_space, params)
        },
        "best_validation_accuracy": float(accuracy)
    }

    with open(BEST_PARAMS_JSON, 'w') as file:
        json.dump(best_result, file, indent=4)



def objfunc(params):
    param_dict = {dim.name: value for dim, value in zip(search_space, params)}
    accuracy = train_test(param_dict, X, y)['validation_accuracy']

    log_results(params, accuracy)

    return -accuracy  


if __name__ == "__main__":
    config_path = "configs/hyperparams.yaml"
    hyperparams_config = load_config(config_path)
    search_space = searchspace(hyperparams_config)
    logging.info("Search space defined.")

    data = load_digits()
    X, y = data.data, data.target
    X = StandardScaler().fit_transform(X)

    with Pool(processes=4) as pool:
        res = gp_minimize(
            objfunc,
            dimensions=search_space,
            n_calls=20,
            random_state=42,
            n_jobs=4
        )

        best_accuracy = -res.fun
        save_best_hyperparameters(res.x, best_accuracy)

        logging.info(f"Best hyperparameters: {res.x}")
        logging.info(f"Best validation accuracy: {best_accuracy}")