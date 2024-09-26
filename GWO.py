from mealpy.swarm_based import GWO
from mealpy.utils.space import FloatVar
from module import Preprocessor, Trainer
import yaml
import logging
import argparse
from tensorflow.keras.preprocessing.sequence import pad_sequences


parser = argparse.ArgumentParser(description='process commandline')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--log_level', type=str, default="INFO")
args = parser.parse_args()


def objective_function(hyperparams):
    learning_rate, rnn_units, batch_size = hyperparams
    logger = logging.getLogger('global logger')

    logger.info("GWO-start!!")
    with open(args.config, 'r') as config_file:
        config = yaml.safe_load(config_file)

    config['training']['learning_rate'] = learning_rate
    config['training']['rnn_units'] = int(rnn_units)
    config['training']['batch_size'] = int(batch_size)

    preprocessor = Preprocessor(config['preprocessing'], logger)
    data_x, data_y, train_x, train_y, validate_x, validate_y, test_x = preprocessor.process()

    # # Sanity check: Print a small sample of the data
    # print(f"Sample train_x: {train_x[:5]}")
    # print(f"Sample validate_x: {validate_x[:5]}")

    # Ensure all values are numeric
    try:
        train_x = [[int(i) for i in seq if str(i).isdigit()] for seq in train_x]
        validate_x = [[int(i) for i in seq if str(i).isdigit()] for seq in validate_x]
    except ValueError as e:
        logger.error(f"Non-numeric data found in sequences: {e}")
        raise

    # Pad the sequences to ensure they have the same length
    max_sequence_length = 32
    train_x = pad_sequences(train_x, maxlen=max_sequence_length, padding='post')
    validate_x = pad_sequences(validate_x, maxlen=max_sequence_length, padding='post')

    # print(f"train_x shape: {train_x.shape}, validate_x shape: {validate_x.shape}")

    if config['training']['model_name'] != 'naivebayse':
        config['training']['vocab_size'] = len(preprocessor.word2ind.keys())

    trainer = Trainer(config['training'], logger, preprocessor.classes)

    trainer.fit(train_x, train_y)

    val_loss = trainer.validate_accuracy(validate_x, validate_y)  # Expecting a scalar, not an object
    print("the val_loss is " + str(val_loss))
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
    gwo_model = GWO.OriginalGWO(problem_size=3, epoch=5, pop_size=5)

    # Solve the optimization problem
    g_best = gwo_model.solve(problem_dict)

    # Display the best solution and its fitness value (negated accuracy)
    best_learning_rate = float(g_best.solution[0])
    best_rnn_units = int(g_best.solution[1])
    best_batch_size = int(g_best.solution[2])

    # Display the best solution and its fitness value (negated accuracy)
    print(f"Best Learning Rate: {float(g_best.solution[0])}")
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


if __name__ == "__main__":
    optimize_hyperparameters()