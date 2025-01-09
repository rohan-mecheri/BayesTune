import yaml
import logging
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool
from src.worker import train_test
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical


logging.basicConfig(level=logging.INFO)


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


def objfunc(params):
    param_dict = {dim.name: value for dim, value in zip(search_space, params)}
    accuracy = train_test(param_dict, X, y)['validation_accuracy']
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
        res = gp_minimize(objfunc, dimensions=search_space, n_calls=20, random_state=42, n_jobs=4)
        logging.info(f"Best hyperparameters: {res.x}")
        logging.info(f"Best validation accuracy: {-res.fun}")
