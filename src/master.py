import yaml  
import logging 
from skopt.space import Real, Integer, Categorical  

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')

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


def get_searchspace(hyperparams_config):
    search_space = []
    for param in hyperparams_config['hyperparameters']:
        if param['type'] == 'Real':
            search_space.append(Real(param['min'], param['max'], name=param['name']))
        elif param['type'] == 'Integer':
            search_space.append(Integer(param['min'], param['max'], name=param['name']))
        elif param['type'] == 'Categorical':
            search_space.append(Categorical(param['categories'], name=param['name']))
    return search_space


if __name__ == "__main__":
    config_path = "configs/hyperparams.yaml"
    hyperparams_config = load_config(config_path)
    logging.info("Hyperparameter configuration completed")

    search_space = get_searchspace(hyperparams_config)
    logging.info(f"Search space: {search_space}")





