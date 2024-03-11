import argparse
import logging
import yaml

from module import Preprocessor, Trainer, Predictor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='process commandline')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--log_level', type=str, default="INFO")
    args = parser.parse_args()

    FORMAT = '%(asctime)s-15s %(message)s'
    logging.basicConfig(format=FORMAT, level=args.log_level)
    logger = logging.getLogger('global logger')

    logger.info("start!!")

    with open(args.config, 'r') as config_file:
        try:
            config = yaml.safe_load(config_file)

            preprocessor = Preprocessor(config['preprocessing'], logger)
            data_x, data_y, train_x, train_y, validate_x, validate_y, test_x = preprocessor.process()

            if config['training']['model_name'] != 'naivebayse':
                config['training']['vocab_size'] = len(preprocessor.word2ind.keys())

            trainer = Trainer(config['training'], logger, preprocessor.classes)
            # dev_model = trainer.fit(train_x, train_y)  # deve model

            full_model = trainer.fit(data_x, data_y)

            accuracy, cls_report = trainer.validate(validate_x, validate_y)
            logger.info("accuracy:{}".format(accuracy))
            logger.info("\n{}\n".format(cls_report))

            predictor = Predictor(config['predict'],logger, full_model)
            probs = predictor.predict_prob(test_x)
            result = predictor.save_result(preprocessor.test_ids, probs)

        except yaml.YAMLError as err:
            logger.warning("config file has error: {}" .format(err))

    logger.info("completed!!!")


