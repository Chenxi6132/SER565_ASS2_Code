from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from module.model import NaiveBayer

class Trainer(object):
    def __init__(self, config, logger, classes):
        self.config = config
        self.logger = logger
        self.classes = classes
        self._create_model(classes)

    def _create_model(self, classes):
        if self.config['model_name'] == 'naivebayse':
            self.model = NaiveBayer(classes)
        else:
            self.logger.warning("Model Type: {} is not support yet".format(self.config['model_name']))

    def fit(self, train_x, train_y):
        self.model.fit(train_x, train_y)
        return self.model

    def metrics(self, predictions, labels):
        accuracy = accuracy_score(labels, predictions)
        cls_report = classification_report(labels, predictions, zero_division=1)
        return accuracy, cls_report

    def validate(self, validate_x, validate_y):
        prediction = self.model.predict(validate_x)
        return self.metrics(prediction, validate_y)

