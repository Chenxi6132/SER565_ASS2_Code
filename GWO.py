import sys
from mealpy.swarm_based import GWO
from mealpy.utils.space import FloatVar
from module import Preprocessor, Trainer
import yaml
import logging
import argparse
import keras


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
        learning_rate, rnn_units, batch_size, embedding_dim, epochs = hyperparams
        config['training']['learning_rate'] = learning_rate
        config['training']['rnn_units'] = int(rnn_units)
        config['training']['batch_size'] = int(batch_size)
        config['training']['embedding_dim'] = int(embedding_dim)
        config['training']['epochs'] = int(epochs)
    elif model_name == 'textcnn':
        learning_rate, batch_size, embedding_dim, epochs, cnn_filter1, cnn_filter2, cnn_filter3, cnn_kernel_size = hyperparams
        config['training']['learning_rate'] = learning_rate
        config['training']['batch_size'] = int(batch_size)
        config['training']['embedding_dim'] = int(embedding_dim)
        config['training']['epochs'] = int(epochs)
        config['training']['filters_1'] = int(cnn_filter1)
        config['training']['filters_2'] = int(cnn_filter2)
        config['training']['filters_3'] = int(cnn_filter3)
        config['training']['cnn_kernel_size'] = int(cnn_kernel_size)
    else:
        logger.error("Model name not supported by objective function.")
        raise ValueError("Model name not supported by objective function.")

    logger.info("GWO-start!!")
    preprocessor = Preprocessor(config['preprocessing'], logger)
    data_x, data_y, train_x, train_y, validate_x, validate_y, test_x = preprocessor.process()

    try:
        train_x = [[int(i) for i in seq if str(i).isdigit()] for seq in train_x]
        validate_x = [[int(i) for i in seq if str(i).isdigit()] for seq in validate_x]
    except ValueError as e:
        logger.error(f"Non-numeric data found in sequences: {e}")
        raise


    # Pad the sequences to ensure they have the same length
    max_sequence_length = 32
    train_x = keras.preprocessing.sequence.pad_sequences(train_x, maxlen=max_sequence_length, padding='post')
    validate_x = keras.preprocessing.sequence.pad_sequences(validate_x, maxlen=max_sequence_length, padding='post')


    config['training']['vocab_size'] = len(preprocessor.word2ind.keys())
    trainer = Trainer(config['training'], logger, preprocessor.classes)

    trainer.fit(train_x, train_y)
    val_loss = trainer.validate_accuracy(validate_x, validate_y)  # Expecting a scalar, not an object
    print("the val_loss is " + str(val_loss))
    return val_loss


def optimize_hyperparameters():
    with open(args.config, 'r') as config_file:
        config = yaml.safe_load(config_file)

    model_name = config['training']['model_name']
    # Define the problem dictionary for hyperparameter optimization
    if model_name ==  'textrnn':
        problem_dict = {
            # learning_rate between 0.01 and 1, batch_size between 16 and 128, dropout is between 0.2 and 0.5, embedding_dim between 50 and 300, epoch between 20 and 200,
            #  rnn_units between 50 and 150,
            "bounds": FloatVar(lb=[0.01,16, 50, 20, 50], ub=[1, 128, 300, 200,150], name="hyperparams"),
            "minmax": "min",  # Minimize the validation accuracy (actually maximize accuracy by returning -accuracy)
            "obj_func": objective_function
        }
    elif model_name == 'textcnn':
        problem_dict = {
            # learning_rate between 0.01 and 1, batch_size between 16 and 128, dropout is between 0.2 and 0.5, embedding_dim between 50 and 300, epoch between 20 and 200,
            # cnn_filter1 between 32 and 256, cnn_filter2 between 32 and 256,, cnn_filter3 between 32 and 256, cnn_kernel_size 3 and 7
            "bounds": FloatVar(lb=[0.01, 16, 50, 20, 32, 32, 32, 3], ub=[1, 128, 300, 200, 256, 256, 256,7], name="hyperparams"),
            "minmax": "min",  # Minimize the validation accuracy (actually maximize accuracy by returning -accuracy)
            "obj_func": objective_function
        }
    elif model_name == 'naivebayse':
        logger.info("Model is Naive Bayes, GWO only works on RNN and CNN models")
        logger.info("Exiting the process, GWO-end!!")
        sys.exit()

    problem_size = len(problem_dict["bounds"].ub)
    print(f"problem_size: {problem_size}")
    # Initialize the Grey Wolf Optimizer (GWO) model
    gwo_model = GWO.OriginalGWO(problem_size= problem_size, epoch=15, pop_size=5)

    # Solve the optimization problem
    g_best = gwo_model.solve(problem_dict)

    # Display the best solution and its fitness value (negated accuracy)
    best_learning_rate = float(g_best.solution[0])
    best_batch_size = int(g_best.solution[1])
    best_embedding_dim = int(g_best.solution[2])
    best_epochs = int(g_best.solution[3])
    if model_name ==  'textrnn':
        best_rnn_units = int(g_best.solution[4])
    elif model_name ==  'textcnn':
        best_cnn_filter1 = int(g_best.solution[4])
        best_cnn_filter2 = int(g_best.solution[5])
        best_cnn_filter3 = int(g_best.solution[6])
        best_cnn_kernel_size = int(g_best.solution[7])


    # Update the config.yml file with optimized hyperparameters
    with open(args.config, 'r') as config_file:
        config = yaml.safe_load(config_file)

    # Update the relevant hyperparameters in the config
    config['training']['learning_rate'] = best_learning_rate
    print(f"best_learning_rate: {best_learning_rate}")
    config['training']['batch_size'] = best_batch_size
    print(f"best_batch_size: {best_batch_size}")
    config['training']['embedding_dim'] = best_embedding_dim
    print(f"best_embedding_dim: {best_embedding_dim}")
    config['training']['epochs'] = best_epochs
    print(f"best_epochs: {best_epochs}")

    if config['training']['model_name'] != 'textrnn':
        config['training']['filters_1'] = best_cnn_filter1
        print(f"best_cnn_filter1: {best_cnn_filter1}")
        config['training']['filters_2'] = best_cnn_filter2
        print(f"best_cnn_filter2: {best_cnn_filter2}")
        config['training']['filters_3'] = best_cnn_filter3
        print(f"best_cnn_filter3: {best_cnn_filter3}")
        config['training']['cnn_kernel_size'] = best_cnn_kernel_size
        print(f"best_cnn_kernel_size: {best_cnn_kernel_size}")

    elif config['training']['model_name'] != 'textcnn':
        config['training']['rnn_units'] = best_rnn_units
        print(f"best_rnn_units: {best_rnn_units}")

    # Write the updated config back to the YAML file
    with open(args.config, 'w') as config_file:
        yaml.dump(config, config_file)

    return g_best.solution


if __name__ == "__main__":
    optimize_hyperparameters()