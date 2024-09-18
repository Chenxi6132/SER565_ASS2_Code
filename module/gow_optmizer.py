from mealpy.swarm_based import GWO
from mealpy.utils.space import FloatVar
from module import Preprocessor, Trainer
import yaml
import logging
import argparse


parser = argparse.ArgumentParser(description='process commandline')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--log_level', type=str, default="INFO")
args = parser.parse_args()

def objective_function(hyperparams):
    # Extract the hyperparameters to optimize
    learning_rate, rnn_units, batch_size = hyperparams
    logger = logging.getLogger('global logger')

    logger.info("GWO-start!!")
    # Load configuration from file (you may want to pass this as a function argument)
    with open(args.config, 'r') as config_file:
        config = yaml.safe_load(config_file)

    # Adjust hyperparameters in the config
    config['training']['learning_rate'] = learning_rate
    config['training']['rnn_units'] = int(rnn_units)
    config['training']['batch_size'] = int(batch_size)

    # Preprocess data
    preprocessor = Preprocessor(config['preprocessing'], logger)
    data_x, data_y, train_x, train_y, validate_x, validate_y, test_x = preprocessor.process()

    if config['training']['model_name'] != 'naivebayse':
        config['training']['vocab_size'] = len(preprocessor.word2ind.keys())

    trainer = Trainer(config['training'], logger, preprocessor.classes)

    print("train_x is " + str(train_x.dtype) + ", train_y is " + str(train_y.dtype))
    # Train the model
    history = trainer.fit(train_x, train_y)

    val_loss = history.history['val_loss'][-1]

    # GWO minimizes the objective, so return val_loss to maximize accuracy
    return val_loss


def optimize_hyperparameters():
    # Define the problem dictionary for hyperparameter optimization
    problem_dict = {
        # learning_rate between 0.0001 and 0.01, rnn_units between 50 and 150, batch_size between 16 and 128
        "bounds": FloatVar(lb=[0.0001, 50, 16], ub=[0.01, 150, 128], name="hyperparams"),
        "minmax": "min",  # Minimize the validation accuracy (actually maximize accuracy by returning -accuracy)
        "obj_func": objective_function
    }

    # Initialize the Grey Wolf Optimizer (GWO) model
    gwo_model = GWO.OriginalGWO(problem_size=3, epoch=30, pop_size=10)

    # Solve the optimization problem
    g_best = gwo_model.solve(problem_dict)

    # Display the best solution and its fitness value (negated accuracy)
    best_learning_rate = g_best.solution[0]
    best_rnn_units = int(g_best.solution[1])
    best_batch_size = int(g_best.solution[2])

    # Display the best solution and its fitness value (negated accuracy)
    print(f"Best Learning Rate: {g_best.solution[0]}")
    print(f"Best RNN Units: {int(g_best.solution[1])}")
    print(f"Best Batch Size: {int(g_best.solution[2])}")
    print(f"Best Accuracy: {-g_best.target.fitness}")  # Negate to get accuracy

    # Update the config.yml file with optimized hyperparameters
    with open(args.config, 'r') as config_file:
        config = yaml.safe_load(config_file)

    # Update the relevant hyperparameters in the config
    config['training']['learning_rate'] = best_learning_rate
    config['training']['rnn_units'] = best_rnn_units
    config['training']['batch_size'] = best_batch_size

    # Write the updated config back to the YAML file
    with open(args.config, 'w') as config_file:
        yaml.dump(config, config_file)

    return g_best.solution
