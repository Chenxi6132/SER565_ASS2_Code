import sys
from mealpy import GWO, FloatVar
import numpy as np
from module import Preprocessor, Trainer
import yaml
import logging
import argparse
from tensorflow.keras.utils import pad_sequences


parser = argparse.ArgumentParser(description='process commandline')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--log_level', type=str, default="INFO")
args = parser.parse_args()

logger = logging.getLogger('global logger')

def objective_function(hyperparams):
    with open(args.config, 'r') as config_file:
        config = yaml.safe_load(config_file)

    model_name = config['training']['model_name']

    # Parse hyperparameters according to model type
    if model_name == 'textrnn':
        learning_rate, batch_size, embedding_dim, epochs, dropout_rate, rnn_units = hyperparams
        config['training']['learning_rate'] = learning_rate
        config['training']['batch_size'] = int(batch_size)
        config['training']['embedding_dim'] = int(embedding_dim)
        config['training']['epochs'] = int(epochs)
        config['training']['dropout_rate'] = dropout_rate
        config['training']['rnn_units'] = int(rnn_units)
    elif model_name == 'textcnn':
        learning_rate, batch_size, embedding_dim, epochs, dropout_rate, cnn_filter1, cnn_filter2, cnn_filter3, cnn_kernel_size = hyperparams
        config['training']['learning_rate'] = learning_rate
        config['training']['batch_size'] = int(batch_size)
        config['training']['embedding_dim'] = int(embedding_dim)
        config['training']['epochs'] = int(epochs)
        config['training']['dropout_rate'] = dropout_rate
        config['training']['filters_1'] = int(cnn_filter1)
        config['training']['filters_2'] = int(cnn_filter2)
        config['training']['filters_3'] = int(cnn_filter3)
        config['training']['cnn_kernel_size'] = int(cnn_kernel_size)
    else:
        logger.error("Model name not supported by objective function.")
        raise ValueError("Model name not supported by objective function.")

    logger.info("GWO-start!!")

    # Preprocessing data
    preprocessor = Preprocessor(config['preprocessing'], logger)
    data_x, data_y, train_x, train_y, validate_x, validate_y, _ = preprocessor.process()
    config['training']['vocab_size'] = len(preprocessor.word2ind.keys())

    max_sequence_length = config['training']['maxlen']
    data_x = pad_sequences(data_x, maxlen=max_sequence_length, padding='post')
    validate_x = pad_sequences(validate_x, maxlen=max_sequence_length, padding='post')

    # Training model
    trainer = Trainer(config['training'], logger, preprocessor.classes)
    trainer.fit(data_x, data_y)

    # Validation
    accuracy = trainer.validate_accuracy(validate_x, validate_y)
    print("the accuracy string format:" + str(accuracy))

    if accuracy >= 0.99:
        accuracy = 0.2
    return accuracy


def optimize_hyperparameters():
    with open(args.config, 'r') as config_file:
        config = yaml.safe_load(config_file)

    model_name = config['training']['model_name']
    # Define the problem dictionary for hyperparameter optimization
    if model_name ==  'textrnn':
        problem_dict = {
            # learning_rate, batch_size, embedding_dim, epochs, dropout_rate, rnn_units 0.001 0.01
            # "bounds": FloatVar(lb=[0.001,16, 50, 5, 0.1, 10], ub=[ 0.01, 256, 200, 50, 0.5, 100], name="hyperparams"),
            "bounds": FloatVar(lb=[0.1, 8, 50, 5, 0.1, 10],
                               ub=[1, 64, 100, 20, 0.5, 60], name="hyperparams"),
            "minmax": "max",
            "obj_func": objective_function,
            "log_file": "console",
        }
    elif model_name == 'textcnn':
        problem_dict = {
            # learning_rate, batch_size, embedding_dim, epochs, dropout_rate, cnn_filter1, cnn_filter2, cnn_filter3, cnn_kernel_size
            # "bounds": FloatVar(lb=[0.001, 16, 50, 5, 0.1, 32, 64, 128, 3], ub=[ 0.01, 256, 200, 50, 0.5, 64, 128, 256,7], name="hyperparams"),
            "bounds": FloatVar(lb=[0.001, 16, 50, 5, 0.1, 32, 64, 128, 3],
                               ub=[0.01, 256, 200, 50, 0.5, 64, 128, 256, 7], name="hyperparams"),
            "minmax": "max",
            "obj_func": objective_function,
            "log_file": "console",
        }
    elif model_name == 'naivebayse':
        logger.info("Model is Naive Bayes, GWO only works on RNN and CNN models")
        logger.info("Exiting the process, GWO-end!!")
        sys.exit()

    problem_size = len(problem_dict["bounds"].ub)
    print(f"problem_size: {problem_size}")
    # Initialize the Grey Wolf Optimizer (GWO) model
    term_dict = {
        "max_early_stop": 3  # after 5 epochs, if the global best doesn't improve then we stop the program
    }
    # gwo_model = GWO.OriginalGWO(problem_size= problem_size, epoch=5, pop_size=5, verbose=True)
    gwo_model = GWO.OriginalGWO(problem_size= problem_size, epoch=5, pop_size=5, verbose=True)

    # Solve the optimization problem
    g_best = gwo_model.solve(problem_dict, termination=term_dict, mode="thread", n_workers=5)

    # Display the best solution and its fitness value (negated accuracy)
    best_learning_rate = float(g_best.solution[0])
    best_batch_size = int(g_best.solution[1])
    best_embedding_dim = int(g_best.solution[2])
    best_epochs = int(g_best.solution[3])
    best_dropout_rate = float(g_best.solution[4])
    if model_name ==  'textrnn':
        best_rnn_units = int(g_best.solution[5])
    elif model_name ==  'textcnn':
        best_cnn_filter1 = int(g_best.solution[5])
        best_cnn_filter2 = int(g_best.solution[6])
        best_cnn_filter3 = int(g_best.solution[7])
        best_cnn_kernel_size = int(g_best.solution[8])


    # Update the config.yml file with optimized hyperparameters
    with open(args.config, 'r') as config_file:
        config = yaml.safe_load(config_file)

    # Update the relevant hyperparameters in the config
    config['training']['learning_rate'] = best_learning_rate
    print(f"best_learning_rate: {best_learning_rate}")
    config['training']['batch_size'] = best_batch_size
    config['preprocessing']['batch_size'] = best_batch_size
    print(f"best_batch_size: {best_batch_size}")
    config['training']['embedding_dim'] = best_embedding_dim
    config['preprocessing']['embedding_dim'] = best_embedding_dim
    print(f"best_embedding_dim: {best_embedding_dim}")
    config['training']['epochs'] = best_epochs
    print(f"best_epochs: {best_epochs}")
    config['training']['dropout_rate'] = best_dropout_rate
    print(f"best_dropout_rate: {best_dropout_rate}")


    if config['training']['model_name'] == 'textcnn':
        config['training']['filters_1'] = best_cnn_filter1
        print(f"best_cnn_filter1: {best_cnn_filter1}")
        config['training']['filters_2'] = best_cnn_filter2
        print(f"best_cnn_filter2: {best_cnn_filter2}")
        config['training']['filters_3'] = best_cnn_filter3
        print(f"best_cnn_filter3: {best_cnn_filter3}")
        config['training']['cnn_kernel_size'] = best_cnn_kernel_size
        print(f"best_cnn_kernel_size: {best_cnn_kernel_size}")

    elif config['training']['model_name'] == 'textrnn':
        config['training']['rnn_units'] = best_rnn_units
        print(f"best_rnn_units: {best_rnn_units}")

    # Write the updated config back to the YAML file
    with open(args.config, 'w') as config_file:
        yaml.dump(config, config_file)

    return g_best.solution

if __name__ == "__main__":
    optimize_hyperparameters()

