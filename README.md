# BayesTune
A Distributed Bayesian Hyperparameter Tuning Framework

## Overview
BayesTune is a scalable and efficient distributed hyperparameter optimization framework that leverages Bayesian Optimization and multiprocessing to fine-tune machine learning models. Designed for high-performance model tuning, BayesTune automates the search for optimal hyperparameters while efficiently managing computational resources across distributed worker nodes.

## Key Features
- **Distributed Hyperparameter Tuning**: Efficient task distribution using Python's multiprocessing for parallel evaluations.  
- **Bayesian Optimization**: Smarter hyperparameter search using probabilistic models (scikit-optimize) to minimize redundant evaluations.  
- **Automated Result Logging**: Tracks and stores evaluation results and optimal configurations in structured formats (CSV, JSON).  
- **Dockerized for Portability**: Easily deployable and scalable using Docker and Docker Compose.  


## Setup Instructions

#### 1. Clone the Repository
```bash
git clone https://github.com/rohan-mecheri/BayesTune.git
cd BayesTune
```
#### 2. Run With Docker

```bash
docker build -f docker/Dockerfile.master -t bayestune-master .
docker run --rm bayestune-master
```
## How It Works

1. **Master Node (`master.py`)**:
   - Loads hyperparameters from `configs/hyperparams.json`.
   - Uses Bayesian Optimization to generate hyperparameter sets.
   - Distributes tasks to worker nodes using multiprocessing.

2. **Worker Node (`worker.py`)**:
   - Trains and evaluates an SVM model using the assigned hyperparameters.
   - Sends performance metrics back to the master node.

3. **Result Logging**:
   - All evaluations are logged in `results/all_results.csv`.
   - The best hyperparameters are saved to `results/best_hyperparams.json`.

## Result Examples

#### `results/all_results.csv`

```csv
C,max_iter,kernel,gamma,Validation_Accuracy
796.54,265,poly,5.96,0.9832
0.001,500,linear,0.0001,0.1123
```
#### `results/best_hyperparams.json`

```json
{
    "best_hyperparameters": {
        "C": 796.5431903172462,
        "max_iter": 265,
        "kernel": "poly",
        "gamma": 5.968541894449077
    },
    "best_validation_accuracy": 0.9832869080779945
}
```

